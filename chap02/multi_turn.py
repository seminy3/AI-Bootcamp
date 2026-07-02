from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


# messages(대화 기록)를 넘겨받아서 AI 응답을 받아오는 함수로 따로 뺌
def get_ai_response(messages):
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        temperature=0.9,
        messages=messages,  # 지금까지의 모든 대화 기록을 통째로 전달
    )
    return response.choices[0].message.content


# 대화 기록을 저장하는 리스트. 여기에 계속 메시지가 쌓임
messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
]

while True:  # 사용자가 "exit"을 입력할 때까지 무한 반복
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    # 방금 사용자가 입력한 내용을 대화 기록에 추가
    messages.append({"role": "user", "content": user_input})

    # 지금까지 쌓인 대화 기록 전체를 AI에게 보내서 응답 받기
    ai_response = get_ai_response(messages)

    # AI의 답변도 대화 기록에 추가해야 다음 질문에서 "기억"하는 것처럼 동작함
    messages.append({"role": "assistant", "content": ai_response})

    print("AI: " + ai_response)
