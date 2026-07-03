# 11장 - 로컬에서 딥시크(DeepSeek)-R1 모델 사용하기

| 파일 | 설명 |
|---|---|
| `deepseek_simple_chatbot.py` / `.ipynb` | Ollama로 로컬에 받은 `deepseek-r1:8b` 모델을 `ChatOllama`로 사용하는 터미널 챗봇. 답변 전 `<think>...</think>` 추론 과정을 걸러내고 실제 답변만 대화 기록에 저장 |
| `retriever.py` / `.ipynb` | 10장에서 만든 벡터 DB(`chap10/chroma_store`)를 재사용하되, 답변 생성 LLM만 GPT에서 로컬 딥시크로 교체한 RAG 모듈 (임베딩은 그대로 OpenRouter 사용) |
| `rag_deepseek.py` | 로컬 딥시크 기반 RAG Streamlit 챗봇 (Streamlit 앱이라 노트북 없음) |

> 사전 준비: [Ollama](https://ollama.com) 설치 후 `ollama pull deepseek-r1:8b` 필요 (약 5.2GB)
