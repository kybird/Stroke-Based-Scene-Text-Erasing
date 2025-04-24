import requests
import os

# --- 설정 ---
API_URL = "http://localhost:8000/erase"
# 테스트할 이미지 파일 경로 (프로젝트 루트 기준)
IMAGE_PATH = "example/images/img_0.jpg"
# 테스트할 OCR 텍스트 파일 경로 (프로젝트 루트 기준)
OCR_TEXT_PATH = "example/txts/gt_img_0.txt"
# 결과 이미지를 저장할 파일 이름
OUTPUT_IMAGE_PATH = "result_api_test.png"

# --- 파일 존재 여부 확인 ---
if not os.path.exists(IMAGE_PATH):
    print(f"오류: 이미지 파일을 찾을 수 없습니다 - {IMAGE_PATH}")
    exit()
if not os.path.exists(OCR_TEXT_PATH):
    print(f"오류: OCR 텍스트 파일을 찾을 수 없습니다 - {OCR_TEXT_PATH}")
    exit()

# --- OCR 데이터 읽기 ---
try:
    with open(OCR_TEXT_PATH, 'r', encoding='utf-8') as f:
        ocr_data = f.read()
except Exception as e:
    print(f"오류: OCR 텍스트 파일 읽기 실패 - {e}")
    exit()

# --- API 요청 보내기 ---
print(f"API 서버({API_URL})에 요청을 보냅니다...")
print(f"- 이미지: {IMAGE_PATH}")
print(f"- OCR 데이터 파일: {OCR_TEXT_PATH}")

try:
    with open(IMAGE_PATH, 'rb') as img_file:
        files = {'image': (os.path.basename(IMAGE_PATH), img_file)}
        data = {'ocr_data': ocr_data}
        
        response = requests.post(API_URL, files=files, data=data)

    # --- 응답 처리 ---
    if response.status_code == 200:
        # 성공 시, 응답 내용을 이미지 파일로 저장
        try:
            with open(OUTPUT_IMAGE_PATH, "wb") as f:
                f.write(response.content)
            print(f"성공! 결과 이미지를 '{OUTPUT_IMAGE_PATH}'로 저장했습니다.")
        except Exception as e:
            print(f"오류: 결과 이미지 저장 실패 - {e}")
    else:
        # 실패 시, 상태 코드와 오류 메시지 출력
        print(f"오류 발생: HTTP 상태 코드 {response.status_code}")
        try:
            # 오류 응답이 JSON 형태일 수 있음 (FastAPI 기본 오류 형식)
            error_detail = response.json()
            print(f"서버 오류 메시지: {error_detail}")
        except requests.exceptions.JSONDecodeError:
            # JSON이 아닌 경우, 텍스트로 출력
            print(f"서버 응답 내용: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"오류: API 요청 실패 - {e}")
    print("서버가 실행 중인지, URL이 올바른지 확인하세요.")
except Exception as e:
    print(f"알 수 없는 오류 발생: {e}")
