from gpt_functions import get_current_time, tools
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import streamlit as st

load_dotenv()

with st.sidebar:
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def get_ai_response(messages, tools=None):
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
        tools=tools,
    )
    return response


st.title("💬 Chatbot (Function Calling)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
    ]

# system 메시지는 화면에 안 보이게, user/assistant만 화면에 표시
for msg in st.session_state.messages:
    if msg["role"] in ("assistant", "user"):
        st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    ai_response = get_ai_response(st.session_state.messages, tools=tools)
    ai_message = ai_response.choices[0].message

    tool_calls = ai_message.tool_calls

    if tool_calls:
        st.session_state.messages.append(ai_message)

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id
            arguments = json.loads(tool_call.function.arguments)

            if tool_name == "get_current_time":
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": get_current_time(timezone=arguments['timezone']),
                })

        ai_response = get_ai_response(st.session_state.messages, tools=tools)
        ai_message = ai_response.choices[0].message

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_message.content,
    })

    st.chat_message("assistant").write(ai_message.content)
