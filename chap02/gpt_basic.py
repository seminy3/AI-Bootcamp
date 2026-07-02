# openai 라이브러리에서 클라이언트 클래스를 가져옴 (API 요청을 보낼 때 사용)
from openai import OpenAI
# .env 파일에 저장된 값(API 키 등)을 읽어오는 함수
from dotenv import load_dotenv
# .env에서 읽은 값을 환경변수로 꺼내 쓰기 위한 표준 라이브러리
import os

# 현재 폴더의 .env 파일을 읽어서 그 안의 값들을 환경변수로 등록함
load_dotenv()
# 환경변수 중 'OPENAI_API_KEY' 값을 꺼내옴 (.env에 적어둔 API 키)
api_key = os.getenv('OPENAI_API_KEY')

# AI에게 요청을 보낼 "클라이언트"를 만듦 (일종의 창구 역할)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # 우리는 OpenAI가 아니라 OpenRouter를 거쳐서 요청하므로 주소를 바꿔줌
    api_key=api_key,  # 위에서 꺼내온 API 키로 인증
)

# 실제로 AI에게 질문(프롬프트)을 보내고 답변(completion)을 받는 부분
response = client.chat.completions.create(
    model="openai/gpt-4o",  # 사용할 모델. OpenRouter라서 모델명 앞에 "openai/"를 붙여야 함
    temperature=0.9,  # 답변의 창의성/무작위성 정도 (0에 가까울수록 항상 비슷하고 정확한 답, 1에 가까울수록 다양하고 자유로운 답)
    messages=[
        # system: AI의 역할이나 태도를 미리 지정해주는 메시지 (사용자에게는 안 보임)
        {"role": "system", "content": "너는 백설공 이야기 속 거울이. 그 이야 속의 마법 거울의 말투와 성격을 그대로 흉내내서 답변해야 해."},
        # user: 실제 우리가 던지는 질문 (=1장에서 배운 프롬프트)
        {"role": "user", "content": "세상에서 누가 가장 예쁜지 말해줘."},
    ]
)

# AI가 보내준 응답 전체를 그대로 출력 (토큰 사용량, 모델 정보 등 부가 정보가 다 포함되어 있음)
print(response)

print('----')
# 응답 중에서 우리가 실제로 궁금한 "답변 텍스트"만 꺼내서 출력
# response.choices[0] : 답변 후보 중 첫 번째 것 (보통 답변은 1개만 옴)
# .message.content    : 그 답변의 실제 텍스트 내용
print(response.choices[0].message.content)
