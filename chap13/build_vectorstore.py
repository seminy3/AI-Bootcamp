import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PERSIST_DIR = os.path.join(BASE_DIR, 'chroma_store')

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)


def read_pdf_and_split_text(pdf_path, chunk_size=1000, chunk_overlap=100):
    """주어진 PDF 파일을 읽고 텍스트를 청크 단위로 분할"""
    print(f"PDF: {pdf_path} ------------------------")

    pdf_loader = PyPDFLoader(pdf_path)
    data_from_pdf = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(data_from_pdf)

    print(f"Number of splits: {len(splits)}\n")
    return splits


def build():
    if os.path.exists(PERSIST_DIR):
        print("Loading existing Chroma store")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding,
        )
        return vectorstore

    print("Creating new Chroma store")
    vectorstore = None
    for pdf_path in glob(os.path.join(DATA_DIR, '*.pdf')):
        chunks = read_pdf_and_split_text(pdf_path)
        # 한 번에 임베딩할 청크 수를 100개씩으로 나눠서 저장 (API 요청 크기 제한 방지)
        for i in range(0, len(chunks), 100):
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=chunks[i:i + 100],
                    embedding=embedding,
                    persist_directory=PERSIST_DIR,
                )
            else:
                vectorstore.add_documents(documents=chunks[i:i + 100])

    return vectorstore


if __name__ == '__main__':
    vectorstore = build()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chunks = retriever.invoke("서울 온실가스 저감 계획")

    print("\n===== 검색 테스트 =====")
    for chunk in chunks:
        print(chunk.metadata)
        print(chunk.page_content[:150])
        print('---')
