import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import retriever

st.title("💬 DeepSeek-R1 랭체인 채팅")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 문서에 기반해 답변하는 도시 정책 전문가야."),
        AIMessage("무엇을 도와드릴까요?"),
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

    # 1) 대화 맥락을 반영해 질문을 명확하게 확장 (10장과 동일한 질의 확장 패턴)
    augmented_query = retriever.query_augmentation_chain.invoke({
        "messages": st.session_state["messages"],
        "query": prompt,
    })

    # 2) 확장된 질문으로 관련 문서(청크) 검색
    docs = retriever.retriever.invoke(f"{prompt}\n{augmented_query}")

    # 3) 검색된 문서를 근거로 로컬 딥시크-R1이 답변 생성 (스트리밍)
    with st.spinner(f"AI가 답변을 준비 중입니다... '{augmented_query}'"):
        response = retriever.document_chain.stream({
            "messages": st.session_state["messages"],
            "context": docs,
        })
        result = st.chat_message("assistant").write_stream(response)

    st.session_state.messages.append(AIMessage(result))
