from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
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


@tool  # @tool 데코레이터를 사용하면 일반 함수를 랭체인 도구로 바로 등록할 수 있음 (5장의 tools 딕셔너리를 직접 안 써도 됨)
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수

    Args:
        timezone (str): 타임존 (예: 'Asia/Seoul') 실제 존재하는 타임존이어야 함
        location (str): 지역명. 타임존이 모든 지명에 대응되지 않기 때문에 이후 llm 답변 생성에 사용됨
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"


@tool
def get_yf_stock_history(ticker: str, period: str) -> str:
    """주식 종목의 가격 데이터를 조회하는 함수"""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    return history.to_markdown()


# 도구를 tools 리스트와 tool_dict에 등록
tools = [get_current_time, get_yf_stock_history]
tool_dict = {"get_current_time": get_current_time, "get_yf_stock_history": get_yf_stock_history}

# 도구를 모델에 바인딩: 이걸 해줘야 모델이 필요할 때 도구를 사용해서 답을 만들 수 있음
llm_with_tools = llm.bind_tools(tools)


def get_ai_response(messages):
    """도구 호출이 필요하면 실제로 호출하고, 그 결과를 반영한 최종 답변까지 재귀적으로 만들어 반환"""
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)

        return get_ai_response(messages)  # 함수 실행 결과를 반영해서 다시 답변 생성

    return response


if __name__ == '__main__':
    messages = [
        SystemMessage("너는 사용자의 질문에 답변을 하기 위해 tools를 사용할 수 있다."),
        HumanMessage("부산은 지금 몇시야? 그리고 테슬라는 한달 전에 비해 주가가 올랐나 내렸나?"),
    ]

    response = get_ai_response(messages)
    print('----')
    print(response.content)
