original implementation is https://github.com/tzm-tora/Stroke-Based-Scene-Text-Erasing

환경설정

```
python -m venv .env
.\.env\Scripts\activate
pip install -r requirements.txt
```


목표
onnx 로 변환후

서비스 가능한 형태로 변경

### 체크리스트

- [x] onnx 변환
- [ ] 추론함수 포팅 
- [ ] 결과 확인
- [ ] 서버