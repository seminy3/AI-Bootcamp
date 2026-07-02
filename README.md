# 강원대학교 AI부트캠프 중급과정

여름 계절학기 AI 부트캠프 중급과정 실습 코드. OpenAI API 대신 [OpenRouter](https://openrouter.ai)를 사용.

## 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv streamlit pymupdf pillow pytz yfinance langchain langchain-openai pydantic tabulate
```

`.env` 파일에 OpenRouter API 키 설정 (git에 커밋되지 않음):

```
OPENAI_API_KEY=sk-or-v1-...
```

## 챕터별 진행 상황

| 챕터 | 내용 | 상태 |
|---|---|---|
| 2장 | GPT 기본 호출, 멀티턴 챗봇, Streamlit 챗봇 | ✅ |
| 3장 | PDF 논문 요약기 | ✅ |
| 4장 | GPT Vision 이미지 분석, 리스닝 퀴즈(TTS) 생성 | ✅ |
| 5장 | Function Calling (시간 조회) | ✅ |
| 6장 | Function Calling + 스트리밍 (주식 정보) | ✅ |
| 7장 | LangChain 기본/세션 관리/LCEL/도구/에이전트 | ✅ |
| 8장 | 인터넷 검색 챗봇 만들기 | ⬜ |
| 9장 | RAG / 음성을 텍스트로 정리하기 | ⬜ |
| 11장 | 로컬 DeepSeek-R1 모델 사용하기 | ⬜ |

## 실행 방법

일반 스크립트:

```bash
.venv/bin/python chap0N/파일명.py
```

Streamlit 앱:

```bash
.venv/bin/streamlit run chap0N/파일명.py
```
