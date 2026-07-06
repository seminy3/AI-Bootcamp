from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List, Union
from utils import save_state, get_outline, save_outline
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# 현재 폴더 경로 찾기 (랭그래프 이미지 저장 및 결과 파일 저장 경로로 활용)
absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)

# 모델 초기화 (OpenAI API 대신 OpenRouter 사용)
llm = ChatOpenAI(
    model="openai/gpt-4o",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)


# 상태 정의
class State(TypedDict):
    messages: List[Union[AnyMessage, str]]


# 목차를 작성하는 노드(agent)
def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")

    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.

        --------------------------------
        - 지난 목차: {outline}
        --------------------------------
        - 이전 대화 내용: {messages}
        """
    )

    # 시스템 프롬프트와 모델을 연결
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]              # 상태에서 메시지를 가져옴
    outline = get_outline(current_path)       # 저장된 목차를 가져옴

    inputs = {
        "messages": messages,
        "outline": outline,
    }

    # 목차 작성 (스트림으로 출력하면서 gathered에 모으기)
    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end='')

    print()

    save_outline(current_path, gathered)      # 목차 저장

    # 메시지 추가
    content_strategist_message = "[Content Strategist] 목차 작성 완료"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    return {"messages": messages}


# 사용자와 대화할 노드(agent): communicator
def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서,
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다.

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.
        outline: {outline}
        --------------------------------
        messages: {messages}
        """
    )

    # 시스템 프롬프트와 모델을 연결
    system_chain = communicator_system_prompt | llm

    messages = state["messages"]              # 상태에서 메시지를 가져옴

    inputs = {
        "messages": messages,
        "outline": get_outline(current_path),
    }

    # 스트림되는 메시지를 출력하면서, gathered에 모으기
    gathered = None

    print('\nAI\t: ', end='')
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    messages.append(gathered)

    return {"messages": messages}


# 상태 그래프 정의
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("communicator", communicator)

# Edges
graph_builder.add_edge(START, "content_strategist")
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

# 그래프 도식화하고 PNG 파일을 만듦
graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))


if __name__ == '__main__':
    # 상태 초기화
    state = State(
        messages=[
            SystemMessage(
                f"""
                너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가팀이다.
                사용자가 사용하는 언어로 대화하라.

                현재시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.
                """
            )
        ],
    )

    while True:
        user_input = input("\nUser\t: ").strip()

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(user_input))
        state = graph.invoke(state)

        print('\n----------------------------------- MESSAGE COUNT\t', len(state["messages"]))

        save_state(current_path, state)       # 현재 state 내용 저장
