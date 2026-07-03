import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

QUESTION = "2025년 현대자동차 미국 시장 전망은 어떻게 되나요?"


def search_with_retry(search_tool, query, retries=3):
    """ddgs 라이브러리가 LibreSSL 환경에서 간헐적으로 TLS 버전 오류를 내므로 재시도"""
    last_error = None
    for _ in range(retries):
        try:
            return search_tool.invoke(query)
        except ValueError as e:
            last_error = e
    raise last_error


# 1) 덕덕고 기본 검색: 무료이고 사용자 데이터를 수집하지 않는다는 게 장점
search = DuckDuckGoSearchResults(results_separator=';\n')
docs = search_with_retry(search, QUESTION)
print("===== 1. 기본 검색 결과 =====")
print(docs)

# 2) API wrapper로 검색 옵션(지역, 기간, 뉴스 소스 등) 지정
wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time="w")
news_search = DuckDuckGoSearchResults(
    api_wrapper=wrapper,
    source="news",  # 뉴스 소스로만 한정
    results_separator=';\n',
)
news_docs = search_with_retry(news_search, QUESTION)
print("\n===== 2. 뉴스 소스 한정 검색 =====")
print(news_docs)


def get_article_text(url: str) -> str:
    """주어진 URL에서 기사 본문 텍스트만 추출"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        article = soup.find('article')
        if article:
            return article.get_text(strip=True)

        div = soup.find('div', id="CmAdContent")
        if div:
            return div.get_text(strip=True)

        return "기사 내용을 찾을 수 없습니다."
    except requests.exceptions.RequestException as e:
        return f"URL을 가져오는 중 오류 발생: {e}"


# 3) 검색 결과에서 링크만 추출
links = []
for doc in news_docs.split(';\n'):
    link = doc.split('link:')[-1].strip()
    links.append(link)

print("\n===== 3. 추출된 링크 =====")
print(links)

# 4) 각 링크에서 기사 본문 가져오기
articles = []
for link in links:
    article_text = get_article_text(link)
    articles.append(article_text)

print("\n===== 4. 기사 본문 =====")
for link, text in zip(links, articles):
    print(f"URL: {link}")
    print(f"내용: {text[:200]}...")
    print('-' * 50)

# 5) 검색 결과(context)를 바탕으로 GPT가 답변 생성 (RAG 패턴)
question_answering_prompt = ChatPromptTemplate([
    ("system", "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}"),
    MessagesPlaceholder(variable_name="messages"),
])
document_chain = question_answering_prompt | model

chat_history = InMemoryChatMessageHistory()
chat_history.add_user_message(QUESTION)

answer = document_chain.invoke({
    "messages": chat_history.messages,
    "context": docs,
})
chat_history.add_ai_message(answer)

print("\n===== 5. 검색 기반 답변 =====")
print(answer.content)
