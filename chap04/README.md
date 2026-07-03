# 4장 - GPT-4o를 이용한 AI 이미지 분석가

| 파일 | 설명 |
|---|---|
| `vision_basic.py` / `.ipynb` | GPT Vision으로 이미지 설명·비교. URL 이미지 설명, 로컬 이미지 설명(base64 인코딩), 두 이미지 비교, 고해상도 이미지의 해상도 한계 테스트 |
| `image_quiz.py` / `.ipynb` | 이미지를 보고 영어 리스닝 퀴즈를 생성하는 파이프라인. 이미지 → GPT로 퀴즈 텍스트/영단어 사전 생성 → TTS로 음성 파일까지 자동 생성 |
| `data.zip` | 실습에 사용한 실제 사진들(`images/`). 압축 해제해서 `data/` 폴더에 두고 사용 |

`image_quiz_eng.md`/`.json`, `data/audio/*.mp3` 등 `image_quiz.py` 실행 결과물은 스크립트를 돌리면 재생성되므로 저장소에 포함하지 않음.
