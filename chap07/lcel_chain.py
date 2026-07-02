from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

# {중괄호} 부분이 나중에 실제 값으로 채워지는 프롬프트 템플릿
system_template = "너는 {story}에 나오는 {character_a} 역할이다. 그 캐릭터에 맞게 사용자와 대화하라."
human_template = "안녕? 저는 {character_b}입니다. 오늘 시간 괜찮으시면 {activity} 같이 할까요?"

prompt_template = ChatPromptTemplate([
    ("system", system_template),
    ("user", human_template),
])

parser = StrOutputParser()

# LCEL: "|"(파이프)로 프롬프트 -> 모델 -> 파서를 순서대로 연결. 데이터가 왼쪽에서 오른쪽으로 흐름
chain = prompt_template | model | parser


class Adlib(BaseModel):
    """스토리 설정과 사용자 입력에 반응하는 대사를 만드는 클래스"""
    answer: str = Field(description="스토리 설정과 사용자와의 대화 기록에 따라 생성된 대사")
    main_emotion: Literal["기쁨", "분노", "슬픔", "공포", "냉소", "불쾌", "중립"] = Field(description="대사의 주요 감정")
    main_emotion_intensity: float = Field(description="대사의 주요 감정의 강도 (0.0 ~ 1.0)")


# 모델이 Adlib 클래스 형태(감정, 강도 등)로 정확히 답하도록 강제하는 구조화된 출력
structured_llm = model.with_structured_output(Adlib)
adlib_chain = prompt_template | structured_llm


if __name__ == '__main__':
    inputs = {
        "story": "미녀와 야수",
        "character_a": "미녀",
        "character_b": "개스톤",
        "activity": "저녁",
    }

    print("===== 일반 체인 (문자열 답변) =====")
    print(chain.invoke(inputs))

    print("\n===== 구조화된 출력 체인 (감정 분석 포함) =====")
    print(adlib_chain.invoke(inputs))
