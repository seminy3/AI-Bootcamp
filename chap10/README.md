# 10장 - RAG (검색 증강 생성)

| 파일 | 설명 |
|---|---|
| `build_vectorstore.py` / `.ipynb` | `data/2040_seoul_plan.pdf`를 읽어 청킹(1000자, 100자 오버랩) → 임베딩(`text-embedding-3-large`) → 크로마(Chroma) 벡터 DB로 저장. 최초 1회 실행 필요 |
| `retriever.py` / `.ipynb` | `rag_chatbot.py`가 불러다 쓰는 모듈. 벡터스토어/리트리버, 문서 기반 답변 체인(`document_chain`), 대화 맥락 기반 질의 확장 체인(`query_augmentation_chain`) 정의 |
| `rag_chatbot.py` | 질의 확장 + 문서 검색 + 스트리밍 답변을 결합한 RAG Streamlit 챗봇 (Streamlit 앱이라 노트북 없음) |
| `data.zip` | 실습용 원본 문서(2040 서울도시기본계획 PDF). 압축 해제해서 `data/` 폴더에 두고 사용 |
| `chroma_store/` | `build_vectorstore.py` 실행 시 자동 생성되는 벡터 DB. 용량이 커서 저장소에 올리지 않음(`.gitignore`) — 직접 스크립트를 실행해 생성 |
