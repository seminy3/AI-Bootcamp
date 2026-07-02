from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ChatOpenAI가 랭체인에서 오픈AI(호환) 모델을 감싸는 클래스
# 아래처럼 model만 바꾸면 다른 회사 모델로도 쉽게 교체할 수 있는 게 랭체인의 핵심 장점
model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",  # 필수! OpenRouter 사용
    api_key=os.getenv('OPENAI_API_KEY'),
)

if __name__ == '__main__':
    # invoke: 프롬프트를 한 번 던지고 답 하나를 받는 가장 기본적인 호출 방식
    response = model.invoke([HumanMessage(content="안녕? 나는 세민이야.")])
    print(response)
    print('----')
    print(response.content)
