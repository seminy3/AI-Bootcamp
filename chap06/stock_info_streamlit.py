from gpt_functions import get_current_time, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations, tools
from collections import defaultdict
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


def get_ai_response(messages, tools=None, stream=True):
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
        tools=tools,
        stream=stream,  # 스트리밍: 답변을 통째로 기다리지 않고 조각(chunk) 단위로 실시간 전송받음
    )
    if stream:
        for chunk in response:
            yield chunk
    else:
        return response


def tool_list_to_tool_obj(tool_calls_chunks):
    """스트리밍 중에는 tool_calls도 여러 조각으로 쪼개져서 온다 (함수명 한 글자씩, 인자 한 글자씩 등).
    이 조각들을 인덱스(index) 기준으로 다시 하나로 합쳐서 완전한 tool_call 딕셔너리로 복원함."""
    tool_calls_dict = defaultdict(lambda: {"id": None, "function": {"arguments": "", "name": None}, "type": None})

    for chunk in tool_calls_chunks:
        if chunk.id is not None:
            tool_calls_dict[chunk.index]["id"] = chunk.id
        if chunk.function.name is not None:
            tool_calls_dict[chunk.index]["function"]["name"] = chunk.function.name
        tool_calls_dict[chunk.index]["function"]["arguments"] += chunk.function.arguments
        if chunk.type is not None:
            tool_calls_dict[chunk.index]["type"] = chunk.type

    return {"tool_calls": list(tool_calls_dict.values())}


def call_function(tool_name, arguments):
    if tool_name == "get_current_time":
        return get_current_time(timezone=arguments['timezone'])
    elif tool_name == "get_yf_stock_info":
        return get_yf_stock_info(ticker=arguments['ticker'])
    elif tool_name == "get_yf_stock_history":
        return get_yf_stock_history(ticker=arguments['ticker'], period=arguments['period'])
    elif tool_name == "get_yf_stock_recommendations":
        return get_yf_stock_recommendations(ticker=arguments['ticker'])


st.title("💬 Chatbot (주식 정보)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "너는 사용자를 도와주는 주식 상담사야."},
    ]

for msg in st.session_state.messages:
    if msg["role"] in ("assistant", "user"):
        st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    ai_response = get_ai_response(st.session_state.messages, tools=tools)

    content = ""
    tool_calls_chunks = []

    # 스트리밍 응답을 실시간으로 화면에 타이핑하듯 출력
    with st.chat_message("assistant").empty():
        for chunk in ai_response:
            content_chunk = chunk.choices[0].delta.content
            if content_chunk:
                content += content_chunk
                st.markdown(content)

            if chunk.choices[0].delta.tool_calls:
                tool_calls_chunks += chunk.choices[0].delta.tool_calls

    tool_obj = tool_list_to_tool_obj(tool_calls_chunks)
    tool_calls = tool_obj["tool_calls"]

    if tool_calls:
        st.session_state.messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls,
        })

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_call_id = tool_call["id"]
            arguments = json.loads(tool_call["function"]["arguments"])

            func_result = call_function(tool_name, arguments)

            st.session_state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": func_result,
            })

        # 함수 실행 결과를 바탕으로 GPT에게 최종 답변을 다시 스트리밍으로 요청
        ai_response = get_ai_response(st.session_state.messages, tools=tools)
        content = ""
        with st.chat_message("assistant").empty():
            for chunk in ai_response:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    content += content_chunk
                    st.markdown(content)

    st.session_state.messages.append({"role": "assistant", "content": content})
