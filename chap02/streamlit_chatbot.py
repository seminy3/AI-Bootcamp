import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# .venv/bin/streamlit run chap02/streamlit_chatbot.py 실행시키는 방법 

# 사이드바에 API 클라이언트 설정 부분을 넣음
with st.sidebar:
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

st.title("💬 Chatbot")

# st.session_state는 스트림릿이 새로고침돼도 값을 유지해주는 저장소
# "messages"가 아직 없으면(=첫 실행이면) 초기 인사말로 시작
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]

# 지금까지 쌓인 대화 기록을 화면에 전부 그려줌
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 화면 하단의 채팅 입력창. 사용자가 뭔가 입력하고 엔터를 치면 그 값이 prompt에 담김
if prompt := st.chat_input():
    if not client:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # 사용자 입력을 대화 기록에 추가하고 화면에도 바로 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 지금까지의 전체 대화 기록을 AI에게 보내서 응답 받기 (멀티턴과 동일한 원리)
    response = client.chat.completions.create(model="openai/gpt-4o", messages=st.session_state.messages)
    msg = response.choices[0].message.content

    # AI 응답도 대화 기록에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
