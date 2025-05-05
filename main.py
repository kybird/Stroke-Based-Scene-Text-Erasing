import os
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import logging # 로깅 추가
import cfg  # 설정 파일 임포트
from onnx_port import (
    load_onnx_model,
    four_point_transform,
    model_prediction_onnx,
    comp_back_persp
)

# --- 로거 초기화 ---
logger = logging.getLogger(__name__)
# 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 생성 (터미널에 출력)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 로그 포맷터 생성
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 로거에 핸들러 추가
logger.addHandler(console_handler)

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- ONNX 모델 로드 ---
# 실제 모델 경로 확인 필요 (cfg.py에 정의되어 있지 않다면 직접 지정)
MODEL_PATH = "model.onnx" # 기본값, 필요시 수정
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

onnx_session = load_onnx_model(MODEL_PATH)
DATA_SHAPE = cfg.data_shape # cfg 모듈에서 data_shape 가져오기

@app.post("/erase")
async def erase_text(
    image: UploadFile = File(...),
    ocr_data: str = Form(...)
):
    """
    이미지와 OCR 데이터를 받아 텍스트를 지운 이미지를 반환합니다.

    - **image**: 처리할 이미지 파일 (jpg, png 등)
    - **ocr_data**: 텍스트 영역 좌표 데이터. 각 줄은 8개의 좌표 숫자로 시작합니다.
                   (예: "158,128,411,128,411,181,158,181,Footpath\n443,128,501,128,501,169,443,169,To")
    """
    try:
        # 1. 이미지 읽기
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="잘못된 이미지 파일입니다.")
        
        h_img, w_img = img.shape[:2]
        dst = img.copy() # 최종 결과 이미지 저장용

        # 2. OCR 데이터 파싱 및 처리
        lines = ocr_data.strip().split('\n')
        for line in lines:
            parts = line.strip().split(',')
            try:
                # 좌표는 최소 8개 필요
                coords = list(map(float, parts[:8]))
                if len(coords) != 8:
                    print(f"경고: 좌표 개수가 8개가 아닌 줄 건너뜀: {line}")
                    continue
                
                pts = np.array(coords, dtype=np.int32).reshape(4, 2)

                # 3. 각 텍스트 영역 처리 (onnx_port.py 함수 사용)
                # four_point_transform은 float32 입력을 기대할 수 있음
                crop, exp_quad, rect = four_point_transform(dst, pts.astype(np.float32))

                # model_prediction_onnx 호출 시 data_shape 전달 (onnx_port.py 수정 안 했으므로 제거)
                # output, _, _ = model_prediction_onnx(onnx_session, crop, DATA_SHAPE)
                # onnx_port.py 수정 안 했으므로 cfg.data_shape를 내부적으로 사용함
                output, _, _ = model_prediction_onnx(onnx_session, crop) 

                # comp_back_persp 함수로 결과 합성
                dst = comp_back_persp(dst, output, h_img, w_img, rect, exp_quad, pts)

            except ValueError:
                print(f"경고: 좌표 파싱 오류 발생 줄 건너뜀: {line}")
                continue
            except Exception as e:
                print(f"처리 중 오류 발생: {e}")
                # 특정 영역 처리 실패 시 계속 진행할지, 에러를 반환할지 결정 필요
                # 여기서는 일단 계속 진행
                continue

        # 4. 결과 이미지 저장 및 반환
        # 임시 파일 사용 (보안 및 정리 용이)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            cv2.imwrite(temp_file.name, dst)
            temp_file_path = temp_file.name
        
        # FileResponse는 파일을 자동으로 닫음. 백그라운드 태스크로 파일 삭제 추가
        return FileResponse(temp_file_path, media_type="image/png", filename="result.png")

    except HTTPException as http_exc:
        raise http_exc # FastAPI 예외는 그대로 전달
    except Exception as e:
        logger.exception(e) # 오류 로깅
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # uvicorn main:app --reload --host 0.0.0.0 --port 8000
    # 위 명령어로 실행하는 것을 권장
    uvicorn.run(app, host="0.0.0.0", port=8000)
