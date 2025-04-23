import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from skimage import metrics
from torchvision.transforms import InterpolationMode
import cfg  # data_shape, model_path 등이 정의되어 있다고 가정

# --- geometry/util 함수들은 원본 그대로 ---
class Point:
    def __init__(self, x=0, y=0):
        self.x, self.y = x, y

class Line:
    def __init__(self, p1, p2):
        self.p1, self.p2 = p1, p2

def GetLinePara(line):
    line.a = line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x * line.p2.y - line.p2.x * line.p1.y

def GetCrossPoint(l1, l2):
    GetLinePara(l1); GetLinePara(l2)
    d = l1.a * l2.b - l2.a * l1.b
    p = Point()
    p.x = (l1.b * l2.c - l2.b * l1.c) / d
    p.y = (l1.c * l2.a - l2.c * l1.a) / d
    return p

def cross_point(quad):
    p1 = Point(*quad[0]); p3 = Point(*quad[2])
    p2 = Point(*quad[1]); p4 = Point(*quad[3])
    return GetCrossPoint(Line(p1,p3), Line(p2,p4)).x, GetCrossPoint(Line(p1,p3), Line(p2,p4)).y

def tensor2jpg_np(tensor: np.ndarray, W: int, H: int):
    # tensor: (C, H', W') float32, normalized([-1,1])
    # 1) denormalize back to [0,255]
    img = (tensor * 0.5 + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 2) (C, H', W') -> (H', W', C)
    img = np.transpose(img, (1, 2, 0))
    # 3) RGB -> BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 4) 원래 H, W로 리사이즈
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    return img

def tensor2mask_np(tensor: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    마스크(mask) 출력용.
    tensor: (1, h', w') float32, 값은 [0,1] 혹은 [-1,1] if normalized
    """
    # 1) 값이 [-1,1]이라면 [0,1]으로 복원
    t = tensor
    if t.min() < 0:
        t = (t * 0.5 + 0.5)
    # 2) [0,1] -> [0,255]
    m = (t * 255.0).clip(0, 255).astype(np.uint8)[0]  # shape (h', w')
    # 3) 리사이즈할 땐 계단 현상 최소화를 위해 nearest
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    # 4) 3채널로 복제 (디스플레이용)
    return np.repeat(m[:, :, None], 3, axis=2)

# --- 품질 지표(원본 그대로) ---
def cal_mse(src, tar):
    return metrics.mean_squared_error(src/255.0, tar/255.0)

def cal_psnr(src, tar):
    return metrics.peak_signal_noise_ratio(src, tar, data_range=255)

def cal_ssim(src, tar):
    return metrics.structural_similarity(src, tar, data_range=255, multichannel=True) * 100


def cal_Width_Height(Poly):
    tl, tr, br, bl = Poly
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    return max(int(widthA), int(widthB)), max(int(heightA), int(heightB)),\
           (widthA+widthB)/2, (heightA+heightB)/2

def expand_roi(Poly, img_width=512, img_height=512, ratio=0.2):
    c_x, c_y = cross_point(Poly)
    expand_poly = np.zeros((4,2), dtype="float32")
    for i, (x,y) in enumerate(Poly):
        dx, dy = x-c_x, y-c_y
        dis = np.hypot(dx, dy) * ratio
        ang = abs(np.arctan2(dy, dx))
        ex = x +  dis*np.cos(ang) * np.sign(dx)
        ey = y +  dis*np.sin(ang) * np.sign(dy)
        expand_poly[i] = [np.clip(ex, 0, img_width-1), np.clip(ey, 0, img_height-1)]
    return expand_poly

def four_point_transform(image, pts):
    quad = pts.astype(np.float32)
    maxW, maxH, meanW, meanH = cal_Width_Height(quad)
    if meanW >= meanH:
        H_target = maxH
    else:
        quad = quad[[1,2,3,0]]
        H_target = maxW
    R = 0.2 if H_target >= 20 else (0.3 if H_target>=15 else (0.8 if H_target>=10 else 1.0))
    exp_quad = expand_roi(quad, image.shape[1], image.shape[0], ratio=R)
    maxW, maxH, _, _ = cal_Width_Height(exp_quad)
    dst_rect = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(exp_quad, dst_rect)
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    return warped, exp_quad, dst_rect

# --- ONNX 추론 파트 ---
def load_onnx_model(path):
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])

def preprocess_np(img: np.ndarray):
    """
    1) 원본 H0,W0 저장
    2) resize_H = min(H0, cfg.data_shape[0]), resize_W = min(W0, cfg.data_shape[1])
    3) BGR->RGB, [0,1], 리사이즈, normalize->[-1,1], C×H×W
    4) data_shape 크기로 zero-pad
    """
    H0, W0 = img.shape[:2]
    # 1. resize 크기 결정
    resize_H = H0 if H0 <= cfg.data_shape[0] else cfg.data_shape[0]
    resize_W = W0 if W0 <= cfg.data_shape[1] else cfg.data_shape[1]

    # 2. 리사이즈 전처리
    arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    arr = cv2.resize(arr, (resize_W, resize_H), interpolation=cv2.INTER_CUBIC)
    arr = (arr - 0.5) / 0.5         # normalize to [-1,1]
    arr = np.transpose(arr, (2,0,1))  # C,H,W

    # 3. 패딩
    C, H_t, W_t = 3, cfg.data_shape[0], cfg.data_shape[1]
    pad = np.zeros((C, H_t, W_t), dtype=np.float32)
    pad[:, :resize_H, :resize_W] = arr

    return pad, H0, W0, resize_H, resize_W


def model_prediction_onnx(session, input_img: np.ndarray):
    # pad, H0, W0 = preprocess_np(input_img)
    # inp_name = session.get_inputs()[0].name
    # out0, out1 = session.run(None, {inp_name: pad[np.newaxis, ...]})
    # # slice back to original
    # out_im = out0[0][:, :pad.shape[1], :pad.shape[2]]
    # out_mask = out1[0][:, :pad.shape[1], :pad.shape[2]]
    # img_out  = tensor2jpg_np(out_im, W0, H0)
    # mask_out = tensor2mask_np(out_mask, W0, H0)
    # # 디스플레이: 원본, 마스크, 결과 수직 연결
    # orig_vis = tensor2jpg_np(np.transpose(input_img.astype(np.float32)/255.0 *1.0 ,(2,0,1)), W0, H0)  # 입력을 다시 0-1->RGB로
    # disp = np.vstack([orig_vis, mask_out, img_out])
    # return img_out, mask_out, disp
    """
    ONNX로 infer 후,
    - output_padding 에서 [:, :resize_H, :resize_W] slice
    - tensor2jpg_np, tensor2mask_np 로 원본 H0×W0 크기로 복원
    """
    pad, H0, W0, resize_H, resize_W = preprocess_np(input_img)
    inp_name = session.get_inputs()[0].name
    out0, out1 = session.run(None, {inp_name: pad[np.newaxis, ...]})

    # 1) slice back to resized ROI
    out_im   = out0[0][:, :resize_H, :resize_W]   # (3, resize_H, resize_W)
    out_mask = out1[0][:, :resize_H, :resize_W]   # (1, resize_H, resize_W)

    # 2) 원본 ROI 크기(H0,W0)로 각각 리사이즈
    img_out  = tensor2jpg_np(out_im,  W0, H0)
    mask_out = tensor2mask_np(out_mask, W0, H0)

    # 3) 디스플레이용: 원본, 마스크, 결과 수직 결합
    disp = np.vstack([input_img, mask_out, img_out])

    return img_out, mask_out, disp

def comp_back_persp(dst: np.ndarray,
                    output: np.ndarray,
                    img_height: int,
                    img_width: int,
                    rect: np.ndarray,
                    expand_quad: np.ndarray,
                    poly_pts: np.ndarray) -> np.ndarray:
    """
    dst          : 원본 이미지 (H×W×3, BGR)
    output       : 모델이 생성한 패치 이미지 (h'×w'×3, BGR)
    img_height   : 원본 높이 H
    img_width    : 원본 너비 W
    rect         : four_point_transform 에서 사용된 dst_rect (4×2 float32)
    expand_quad  : four_point_transform 에서 확장된 쿼드 (4×2 float32)
    poly_pts     : 원본 bbox 정수 pts (4×2 int32)
    """
    # 1) 패치를 원래 위치로 투영
    M = cv2.getPerspectiveTransform(rect, expand_quad.astype(np.float32))
    text_erased_patch = cv2.warpPerspective(
        output, M, (img_width, img_height), flags=cv2.INTER_CUBIC)

    # 2) 폴리곤 영역 마스크 생성
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    poly = poly_pts.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255)

    # 3) 경계 부드럽게 하기 위해 팽창
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel)

    # 4) 채널 차원 맞추고 합성
    mask = mask[:, :, None]  # H×W×1
    composed = np.where(mask > 0, text_erased_patch, dst)

    return composed


def inference_onnx(session, src_img_dir, src_txt_dir, save_path):
    txts = sorted([f for f in os.listdir(src_txt_dir) if f.endswith('.txt')],
                  key=lambda x: int(''.join(filter(str.isdigit, x))))
    for i, fn in enumerate(txts):
        lines = open(os.path.join(src_txt_dir, fn)).read().splitlines()
        idx = fn.replace('.txt','').replace('gt_','').replace('res_','')
        try:
            img = cv2.imread(os.path.join(src_img_dir, idx +'.jpg'))
            img_clone = img.copy()
        except Exception:
            img = cv2.imread(os.path.join(src_img_dir, idx +'.png'))
        img_clone = img.copy()
        dst = img.copy()
        mask_img = img.copy()
        h_img, w_img = img.shape[:2]
        k = 1
        for gt in lines:
            parts = gt.split(',')
            n = len(parts)//2
            pts = np.array(list(map(float, parts[:2*n])), dtype=np.int32).reshape(-1,2)
            if n==4:
                crop, exp_quad, rect = four_point_transform(dst, pts.astype(np.float32))
                output, o_mask, display = model_prediction_onnx(session, crop)
                dst = comp_back_persp(dst, output, h_img, w_img, rect, exp_quad, pts)
                mask_img = comp_back_persp(
                    mask_img, o_mask, h_img, w_img, rect, exp_quad, pts)
            else:
                print("잘못된 bbox:", gt)
            cv2.namedWindow('img_clone', cv2.WINDOW_NORMAL)
            cv2.imshow('img_clone', img_clone)
            cv2.imshow('dst', dst)
            cv2.imshow('mask_img', mask_img)
            windows_name = f'part_{k}'
            cv2.imshow(windows_name, display)
            k += 1
        cv2.waitKey()
        for ii in range(k,1):
            cv2.destroyWindow(f'part_{ii}')                        
        cv2.imwrite(os.path.join(save_path, idx+'.png'), dst)


def quality_metric(img_path, label_path):
    imgs = sorted([f for f in os.listdir(img_path) if f.lower().endswith(('.png','.jpg'))],
                  key=lambda x: int(''.join(filter(str.isdigit,x))))
    psnrs, ssims, mses = [], [], []
    for nm in imgs:
        im = cv2.imread(os.path.join(img_path, nm))
        lbl = cv2.imread(os.path.join(label_path, nm.replace('.png','.jpg'))) \
              or cv2.imread(os.path.join(label_path, nm))
        psnrs.append(cal_psnr(im, lbl))
        ssims.append(cal_ssim(im, lbl))
        mses.append(cal_mse(im, lbl))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(mses))

# if __name__ == "__main__":
#     sess = load_onnx_model(cfg.model_path)
#     inference_onnx(sess, cfg.src_img_dir, cfg.src_txt_dir, cfg.save_path)
