# 6장 - GPT로 경제 이야기하기 (주식 정보 + 스트리밍)

| 파일 | 설명 |
|---|---|
| `gpt_functions.py` / `.ipynb` | 5장 함수에 주식 관련 함수 3종 추가: `get_yf_stock_info`, `get_yf_stock_history`, `get_yf_stock_recommendations` (yfinance 사용) + 확장된 `tools` 스키마 |
| `stock_info_streamlit.py` | 스트리밍 응답 + 함수 호출을 결합한 Streamlit 주식 정보 챗봇. 스트리밍 중 조각난 `tool_calls` 청크를 재조립하는 `tool_list_to_tool_obj` 함수가 핵심 (Streamlit 앱이라 노트북 없음) |
