from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

# 타빌리는 덕덕고보다 안정적이지만 유료(첫 1000회 API 콜은 무료)
client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

QUERY = "현대자동차 미국 시장 2025년 전망"

# Step 1. 질문에 대한 자료를 타빌리로 검색
content = client.search(
    QUERY,
    include_raw_content=True,
    search_depth="advanced",
)["results"]

print("===== 검색 결과 =====")
print(content)

# Step 2. 검색 결과를 근거로 기사를 작성하도록 프롬프트 구성
prompt = [
    {
        "role": "system",
        "content": (
            "당신은 신문기사를 쓰는 기자 AI입니다.\n"
            "당신은 주어진 정보를 바탕으로 객관적이고 체계적으로 작성된 기사를 써야 합니다.\n"
        ),
    },
    {
        "role": "user",
        "content": (
            f'정보: """{content}"""\n\n'
            f'위의 정보를 사용하여, 다음 질문에 대해 자세한 보고서를 한국어로 작성하세요: "{QUERY}"\n'
            '- 신문기사 형식을 사용하되, MLA를 준수하는 markdown 문법을 사용해주세요.\n'
            '- 활용한 자료는 출처를 명시해주세요.'
        ),
    },
]

# Step 3. GPT로 기사 생성
report = model.invoke(prompt).content

print("\n===== 생성된 기사 =====")
print(report)
