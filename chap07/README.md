# 7장 - 랭체인(LangChain)으로 에이전트 만들기

| 파일 | 설명 |
|---|---|
| `langchain_basic.py` / `.ipynb` | `ChatOpenAI`로 LangChain의 가장 기본적인 모델 호출 (`invoke`) |
| `langchain_session_history.py` / `.ipynb` | `RunnableWithMessageHistory` + `session_id`로 세션별 대화 기록 자동 관리 (직접 messages를 append하지 않아도 됨) |
| `lcel_chain.py` / `.ipynb` | LCEL(`|` 파이프)로 프롬프트-모델-파서를 연결하는 체인, `with_structured_output`으로 Pydantic 모델 기반 구조화된 출력 |
| `langchain_tools.py` / `.ipynb` | `@tool` 데코레이터로 함수를 도구로 등록하고 `bind_tools`로 모델에 연결, 재귀적으로 도구 호출을 처리하는 에이전트 |
| `streamlit_langchain_agent.py` | 위 도구 호출 에이전트 + 스트리밍을 결합한 Streamlit 챗봇 (Streamlit 앱이라 노트북 없음) |
