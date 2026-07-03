from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)


class State(TypedDict):
    """
    State 클래스는 TypedDict를 상속받습니다.

    속성:
        messages (Annotated[list[str], add_messages]): 메시지들은 "list" 타입을 가집니다.
        'add_messages' 함수는 이 상태 키가 어떻게 업데이트되어야 하는지를 정의합니다.
        (이 경우, 메시지를 덮어쓰는 대신 리스트에 추가합니다)
    """
    messages: Annotated[list[str], add_messages]


# StateGraph 클래스를 사용하여 State 타입의 그래프를 생성
graph_builder = StateGraph(State)


def generate(state: State):
    """주어진 상태를 기반으로 챗봇의 응답 메시지를 생성"""
    return {"messages": [model.invoke(state["messages"])]}


graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

if __name__ == '__main__':
    # 1) 그래프에 첫 메시지 입력
    response = graph.invoke({"messages": ["안녕하세요! 저는 최세민 입니다"]})
    print(type(response))
    print(response)

    # 2) 이전 대화 내용에 새 메시지를 추가해서 이어가기 (그래프 자체에는 메모리가 없음)
    response["messages"].append("제 이름을 아시나요?")
    print("\n===== 메시지 추가 후 응답 =====")
    print(graph.invoke(response))

    # 3) 스트림 방식으로 출력하기
    print("\n===== 스트리밍 출력 =====")
    inputs = {"messages": [("human", "한국과 일본의 관계에 대해 자세히 알려줘")]}
    for chunk, _ in graph.stream(inputs, stream_mode="messages"):
        print(chunk.content, end="")
    print()
