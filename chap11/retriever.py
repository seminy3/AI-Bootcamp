from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 10장에서 만든 벡터 DB(2040 서울도시기본계획)를 그대로 재사용
PERSIST_DIR = os.path.join(BASE_DIR, '..', 'chap10', 'chroma_store')

# 임베딩은 OpenRouter(GPT) 사용, 답변 생성만 로컬 딥시크-R1 사용
embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENAI_API_KEY'),
)

llm = ChatOllama(model="deepseek-r1:8b")

print("Loading existing Chroma store")
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever(k=3)

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
document_chain = create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()

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
