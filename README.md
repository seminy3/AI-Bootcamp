# 강원대학교 AI부트캠프 중급과정

여름 계절학기 AI 부트캠프 중급과정 실습 코드. OpenAI API 대신 [OpenRouter](https://openrouter.ai)를 사용.

각 챕터 폴더의 `README.md`에 해당 챕터 파일들의 상세 설명이 있습니다.

## 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv streamlit pymupdf pillow pytz yfinance langchain langchain-openai pydantic tabulate \
    ddgs beautifulsoup4 tavily-python youtube-search youtube-transcript-api langchain-community \
    pandas langchain_chroma pypdf langgraph langchain-ollama jupyter nbformat
```

`.env` 파일에 API 키 설정 (git에 커밋되지 않음):

```
OPENAI_API_KEY=sk-or-v1-...        # OpenRouter (2~13장 공통)
TAVILY_API_KEY=tvly-...            # 8장 Tavily 검색
```

9장은 OpenRouter 오디오 모델을 쓰므로 별도 설치가 필요 없고, 11장은 [Ollama](https://ollama.com) 설치 후 `ollama pull deepseek-r1:8b`가 필요합니다.

PDF/음성/이미지 등 원본 데이터 파일은 용량 문제로 각 챕터의 `data.zip`으로만 저장소에 포함되어 있습니다 (3, 4, 9, 10, 13장). 실행 전 압축을 해제하세요:

```bash
cd chap0N && unzip data.zip -d data && cd ..
```

실행 결과물(요약, 퀴즈, STT 결과 등)은 저장소에 포함하지 않으며, 각 스크립트를 실행하면 다시 생성됩니다.

## 챕터별 진행 상황

| 챕터 | 내용 | 상태 |
|---|---|---|
| 2장 | GPT 기본 호출, 멀티턴 챗봇, Streamlit 챗봇 | ✅ |
| 3장 | PDF 논문 요약기 | ✅ |
| 4장 | GPT Vision 이미지 분석, 리스닝 퀴즈(TTS) 생성 | ✅ |
| 5장 | Function Calling (시간 조회) | ✅ |
| 6장 | Function Calling + 스트리밍 (주식 정보) | ✅ |
| 7장 | LangChain 기본/세션 관리/LCEL/도구/에이전트 | ✅ |
| 8장 | 인터넷 검색 챗봇 만들기 (DuckDuckGo/Tavily/YouTube) | ✅ |
| 9장 | 음성을 텍스트로 정리하기 (OpenRouter 오디오 모델, 화자 구분) | ✅ |
| 10장 | RAG (검색 증강 생성) | ✅ |
| 11장 | 로컬 DeepSeek-R1 모델 사용하기 | ✅ |
| 12장 | LangGraph 기본/메모리/도구 에이전트 | ✅ |
| 13장 | LangGraph 멀티에이전트 RAG | ✅ |

## 실행 방법

일반 스크립트:

```bash
.venv/bin/python chap0N/파일명.py
```

Streamlit 앱:

```bash
.venv/bin/streamlit run chap0N/파일명.py
```

주피터 노트북 (Streamlit 앱을 제외한 모든 스크립트는 `.ipynb` 버전도 함께 제공):

```bash
.venv/bin/jupyter notebook
```
