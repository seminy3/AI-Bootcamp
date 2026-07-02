from langchain_core.chat_history import InMemoryChatMessageHistory  # 메모리에 대화 기록을 저장하는 클래스
from langchain_core.runnables.history import RunnableWithMessageHistory  # 메시지 기록을 자동으로 관리해주는 래퍼
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

# 세션 ID별 대화 기록을 저장할 딕셔너리 (세션 = 대화방이라고 생각하면 됨)
store = {}


def get_session_history(session_id: str):
    """해당 세션 ID의 대화 기록이 없으면 새로 만들고, 있으면 그대로 반환"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 모델에 "대화 기록을 자동으로 관리해주는 기능"을 씌운 것
# 매번 messages 리스트를 직접 append 안 해도, session_id만 같으면 알아서 이전 대화를 기억함
with_message_history = RunnableWithMessageHistory(model, get_session_history)


if __name__ == '__main__':
    config = {"configurable": {"session_id": "abc2"}}  # 이 session_id로 대화방을 구분

    response = with_message_history.invoke(
        [HumanMessage(content="안녕? 난 세민이야.")],
        config=config,
    )
    print(response.content)

    # 같은 세션 ID로 다시 물어보면 이전 대화를 기억함
    response = with_message_history.invoke(
        [HumanMessage(content="내 이름이 뭐지?")],
        config=config,
    )
    print(response.content)

    # 다른 세션 ID("abc3")로 물어보면 완전히 새로운 대화라 이름을 모름
    config2 = {"configurable": {"session_id": "abc3"}}
    response = with_message_history.invoke(
        [HumanMessage(content="내 이름이 뭐지?")],
        config=config2,
    )
    print(response.content)
