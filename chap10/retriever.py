from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, 'chroma_store')

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

print("Loading existing Chroma store")
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever(k=3)

# 검색된 문서(context)를 바탕으로 답변하는 체인
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

# 대화 맥락(대명사 등)을 반영해 질문을 명확한 한 문장으로 바꾸는 체인 (질의 확장)
query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "기존의 대화 내용을 활용하여 사용자의 아래 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환하라. "
            "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 바꿔라.:\n\n{query}",
        ),
    ]
)
query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()
