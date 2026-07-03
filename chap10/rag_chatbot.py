import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import retriever

st.title("💬 RAG 도시 정책 챗봇 (2040 서울도시기본계획)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 문서에 기반해 답변하는 도시 정책 전문가야."),
    ]

for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        # SystemMessage는 화면에 표시하지 않음

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # 1) 대화 맥락을 반영해 질문을 명확하게 확장 (질의 확장)
    augmented_query = retriever.query_augmentation_chain.invoke({
        "messages": st.session_state["messages"],
        "query": prompt,
    })

    # 2) 확장된 질문으로 관련 문서(청크) 검색
    docs = retriever.retriever.invoke(f"{prompt}\n{augmented_query}")

    # 3) 검색된 문서를 근거로 답변 스트리밍 생성
    def get_ai_response(messages, docs):
        response = retriever.document_chain.stream({
            "messages": messages,
            "context": docs,
        })
        for chunk in response:
            yield chunk  # create_stuff_documents_chain은 기본적으로 문자열을 그대로 yield함

    with st.chat_message("assistant"):
        result = st.write_stream(get_ai_response(st.session_state["messages"], docs))

    st.session_state.messages.append(AIMessage(result))
