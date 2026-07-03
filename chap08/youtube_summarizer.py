from youtube_search import YoutubeSearch
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import os

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

QUERY = "인공지능 최신 뉴스"

# 1) 유튜브에서 영상 검색
videos = YoutubeSearch(QUERY, max_results=5).to_dict()

# 2) 영상 길이가 60분 이하인 것만 남기기 (자막이 너무 길면 요약이 부정확해짐)
print('총 영상 수:', len(videos))
videos = [v for v in videos if len(v['duration'].split(':')) < 3]
print('60분 이하 영상 수:', len(videos))

# 3) 각 영상의 자막(자막이 없으면 빈 리스트) 불러오기
for v in videos:
    v['video_url'] = 'http://youtube.com' + v['url_suffix']
    loader = YoutubeLoader.from_youtube_url(
        v['video_url'],
        language=['ko', 'en'],  # 자막 언어 우선순위
    )
    v['content'] = loader.load()

# 4) 요약 프롬프트 구성 (context에 자막 문서를 그대로 넣음)
prompt = ChatPromptTemplate.from_messages(
    [("system", "다음 영상에 대한 요약을 한국어로 만들어줘 :\n\n{context}")]
)
chain = create_stuff_documents_chain(model, prompt)

# 5) 모든 영상에 대해 순서대로 요약 생성
for v in tqdm(videos):
    if v['content']:
        v['summary'] = chain.invoke({"context": v['content']})
    else:
        v['summary'] = "자막이 없어 요약할 수 없습니다."

print("\n===== 유튜브 검색 + 자막 요약 결과 =====")
for v in videos:
    print(f"제목: {v['title']}")
    print(f"링크: {v['video_url']}")
    print(f"요약: {v['summary']}")
    print('-' * 50)
