# 5장 - 펑션 콜링(Function Calling)

| 파일 | 설명 |
|---|---|
| `gpt_functions.py` / `.ipynb` | GPT에게 등록할 함수 정의. `get_current_time(timezone)`과 이를 설명하는 JSON 스키마(`tools`) |
| `what_time_is_it_terminal.py` / `.ipynb` | 함수 호출을 지원하는 터미널 챗봇. GPT가 `tool_calls`를 요청하면 실제 함수를 실행하고 결과를 다시 GPT에 전달해 최종 답변을 받는 흐름 |
| `streamlit_function_calling.py` | 위 챗봇을 Streamlit 웹 UI로 옮긴 버전 (Streamlit 앱이라 노트북 없음) |
