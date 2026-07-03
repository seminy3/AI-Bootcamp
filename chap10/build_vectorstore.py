from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, 'data', '2040_seoul_plan.pdf')
PERSIST_DIR = os.path.join(BASE_DIR, 'chroma_store')

# OpenRouter도 임베딩 엔드포인트를 지원하므로 동일한 클라이언트 설정 사용
embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)


def build():
    # 1) PDF 읽기
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()

    # 2) 청킹 (1000자 단위, 100자 오버랩)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(data)
    print(f'총 청크 수: {len(splits)}')

    # 3) 오버랩 처리: 각 청크 뒤에 다음 청크의 앞부분을 덧붙여 문맥 단절 완화
    for i in range(len(splits) - 1):
        splits[i].page_content += "\n" + splits[i + 1].page_content[:100]

    # 4) 임베딩 후 크로마 DB에 저장 (한 번 만들어두면 재사용 가능)
    if not os.path.exists(PERSIST_DIR):
        print("Creating new Chroma store")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=PERSIST_DIR,
        )
    else:
        print("Loading existing Chroma store")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding,
        )

    return vectorstore


if __name__ == '__main__':
    vectorstore = build()

    retriever = vectorstore.as_retriever(k=3)
    docs = retriever.invoke("서울시의 환경 정책에 대해 궁금해")

    print("\n===== 유사 청크 검색 결과 =====")
    for d in docs:
        print(d.page_content[:200])
        print('------')
