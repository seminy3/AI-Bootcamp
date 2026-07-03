from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 로컬 Ollama에 받아둔 딥시크-R1 모델을 사용 (API 키 불필요, 완전히 로컬 실행)
llm = ChatOllama(model="deepseek-r1:8b")

messages = [
    SystemMessage("너는 사용자를 도와주는 상담사야."),
]

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    messages.append(HumanMessage(user_input))

    # 스트림으로 응답을 받아 바로바로 출력 (일반 invoke는 딥시크의 추론 과정 때문에 응답이 느림)
    response = llm.stream(messages)
    ai_message = None
    for chunk in response:
        print(chunk.content, end="")
        if ai_message is None:
            ai_message = chunk
        else:
            ai_message += chunk
    print('')

    # 딥시크는 답변 전에 <think>...</think> 추론 과정을 포함할 수 있으므로, 있다면 실제 답변만 골라내 저장
    if "</think>" in ai_message.content:
        message_only = ai_message.content.split("</think>")[1].strip()
    else:
        message_only = ai_message.content.strip()
    messages.append(AIMessage(message_only))
