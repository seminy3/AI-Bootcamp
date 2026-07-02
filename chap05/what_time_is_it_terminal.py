from gpt_functions import get_current_time, tools
from openai import OpenAI
from dotenv import load_dotenv
import os
import json  # GPT가 JSON 형태의 문자열로 반환하는 함수 인자를 읽기 위한 라이브러리

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def get_ai_response(messages, tools=None):
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
        tools=tools,  # 사용 가능한 도구 목록 전달
    )
    return response


messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
]

while True:
    user_input = input("사용자\t: ")

    if user_input == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    ai_response = get_ai_response(messages, tools=tools)
    ai_message = ai_response.choices[0].message

    tool_calls = ai_message.tool_calls  # AI가 호출하고 싶어하는 함수 목록

    if tool_calls:
        # 사용자가 "서울, 뉴욕, 런던 시간 알려줘"처럼 여러 개를 한 번에 물어보면
        # tool_calls 안에 함수 호출 요청이 여러 개 들어있을 수 있어서 for문으로 전부 처리
        messages.append(ai_message)

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id
            arguments = json.loads(tool_call.function.arguments)  # 문자열을 딕셔너리로 변환

            if tool_name == "get_current_time":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": get_current_time(timezone=arguments['timezone']),  # 실제 함수 실행 결과
                })

        # 함수 실행 결과를 바탕으로 GPT에게 최종 답변을 다시 요청
        ai_response = get_ai_response(messages, tools=tools)
        ai_message = ai_response.choices[0].message

    messages.append(ai_message)

    content = ai_message.content or "(응답 없음)"
    print("AI\t: " + content)
