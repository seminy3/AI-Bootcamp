# 2장 - GPT API 시작하기

| 파일 | 설명 |
|---|---|
| `gpt_basic.py` / `.ipynb` | OpenRouter를 통해 GPT를 가장 기본적인 방식으로 호출. 클라이언트 생성부터 `chat.completions.create` 한 번 호출까지 |
| `multi_turn.py` / `.ipynb` | 터미널에서 대화를 주고받는 멀티턴 챗봇. `messages` 리스트에 대화 기록을 계속 append 하는 방식으로 문맥 유지 |
| `streamlit_chatbot.py` | 위 멀티턴 챗봇을 Streamlit 웹 UI로 옮긴 버전. `st.session_state`로 대화 기록 관리 (Streamlit 앱이라 노트북 없음, `streamlit run streamlit_chatbot.py`로 실행) |
