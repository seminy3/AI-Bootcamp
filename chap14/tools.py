from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
from dotenv import load_dotenv
import json
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)
output_dir = os.path.join(current_path, 'data')   # 검색 결과(JSON) 저장 폴더

# data 폴더가 없을 경우 자동 생성
os.makedirs(output_dir, exist_ok=True)

# 오픈AI Embedding 설정 (OpenAI API 대신 OpenRouter 사용)
embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
)

# 크로마 DB 저장 경로 설정
persist_directory = f"{current_path}/data/chroma_store"

# Chroma 객체 생성
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)


@tool
def web_search(query: str):
    """
    주어진 query에 대해 웹검색을 하고, 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        list: 검색 결과
    """
    client = TavilyClient()

    content = client.search(
        query,
        search_depth="advanced",
        include_raw_content=True,
    )

    results = content["results"]

    for result in results:
        if result["raw_content"] is None:
            try:
                result["raw_content"] = load_web_page(result["url"])
            except Exception as e:
                print(f"Error loading page: {result['url']}")
                print(e)
                result["raw_content"] = result["content"]

    # 검색 결과를 JSON 파일로 저장
    file_name = f"resources_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.json"
    resources_json_path = os.path.join(output_dir, file_name)

    with open(resources_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results, resources_json_path   # 검색 결과와 JSON 파일 경로 반환


def load_web_page(url: str):
    """WebBaseLoader로 URL의 본문을 읽어와 공백을 정리한 텍스트로 반환한다."""
    loader = WebBaseLoader(url, verify_ssl=False)

    content = loader.load()
    raw_content = content[0].page_content.strip()

    while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
        raw_content = raw_content.replace('\n\n\n', '\n\n')
        raw_content = raw_content.replace('\t\t\t', '\t\t')

    return raw_content


def web_page_to_document(web_page):
    """검색 결과(dict) 하나를 랭체인 Document 객체로 변환한다."""
    # raw_content와 content 중 정보가 많은 것을 page_content로 한다.
    if len(web_page['raw_content']) > len(web_page['content']):
        page_content = web_page['raw_content']
    else:
        page_content = web_page['content']

    document = Document(
        page_content=page_content,
        metadata={
            'title': web_page['title'],
            'source': web_page['url'],
        }
    )

    return document


def web_page_json_to_documents(json_file):
    """JSON 파일에 저장된 검색 결과들을 Document 객체 리스트로 변환한다."""
    with open(json_file, "r", encoding='utf-8') as f:
        resources = json.load(f)

    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Document 객체들을 청크 단위로 자른다."""
    print('Splitting documents...')
    print(f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할합니다.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(documents)

    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits


def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    """새로운 Document들만 청킹해 Chroma DB에 저장한다(중복 URL은 건너뜀)."""
    print("Documents를 Chroma DB에 저장합니다.")
    # documents의 url 가져오기
    urls = [document.metadata['source'] for document in documents]
    # 이미 vectorstore에 저장된 urls 가져오기
    stored_metadatas = vectorstore._collection.get()['metadatas']
    stored_web_urls = [metadata['source'] for metadata in stored_metadatas]
    # 새로운 urls만 남기기
    new_urls = set(urls) - set(stored_web_urls)
    # 새로운 urls에 대한 documents만 남기기
    new_documents = []

    for document in documents:
        if document.metadata['source'] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    # 새로운 documents를 Chroma DB에 저장
    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # 크로마 DB에 저장
    if splits:
        vectorstore.add_documents(splits)
    else:
        print("No new urls to process")


def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    """json 파일에서 documents를 만들고, 그 documents들을 Chroma DB에 저장한다."""
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@tool
def retrieve(query: str, top_k: int = 5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs


if __name__ == '__main__':
    # 1) 웹 검색 → JSON 저장
    results, resources_json_path = web_search.invoke({"query": "2025년 한국 경제 전망"})
    print(f"검색 결과 저장 경로: {resources_json_path}")

    # 2) JSON → Document 변환 → 청킹 → Chroma DB에 저장
    add_web_pages_json_to_chroma(resources_json_path)

    # 3) 벡터 검색으로 확인
    retrieved_docs = retrieve.invoke({"query": "한국 경제 위험 요소"})
    print(retrieved_docs)
