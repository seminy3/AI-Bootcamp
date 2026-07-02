import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime
import pytz
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)


@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수"""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f'{timezone} ({location}) 현재시각 {now}'
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"


@tool
def get_yf_stock_history(ticker: str, period: str) -> str:
    """주식 종목의 가격 데이터를 조회하는 함수"""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    return history.to_markdown()


tools = [get_current_time, get_yf_stock_history]
tool_dict = {"get_current_time": get_current_time, "get_yf_stock_history": get_yf_stock_history}
llm_with_tools = llm.bind_tools(tools)


def get_ai_response(messages):
    """스트리밍(stream)으로 답을 받으면서, 필요하면 도구도 호출하는 재귀 제너레이터"""
    response = llm_with_tools.stream(messages)

    gathered = None
    for chunk in response:
        yield chunk
        gathered = chunk if gathered is None else gathered + chunk

    if gathered.tool_calls:
        st.session_state.messages.append(gathered)

        for tool_call in gathered.tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call)
            st.session_state.messages.append(tool_msg)

        for chunk in get_ai_response(st.session_state.messages):
            yield chunk


st.title("💬 GPT-4o Langchain Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다. tools를 사용할 수 있다."),
        AIMessage("How can I help you?"),
    ]

for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            st.chat_message("tool").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    response = get_ai_response(st.session_state.messages)

    result = st.chat_message("assistant").write_stream(response)
    st.session_state.messages.append(AIMessage(result))
