# 13장 - 랭그래프를 활용한 멀티에이전트 RAG 만들기

| 파일 | 설명 |
|---|---|
| `build_vectorstore.py` / `.ipynb` | `data/` 안의 여러 PDF(서울·뉴욕 도시계획)를 한 번에 읽어 청킹·임베딩 후 하나의 크로마 벡터 DB로 통합 구축 |
| `multi_agent_rag.py` / `.ipynb` | 멀티에이전트 RAG 그래프. ① 질문을 `vectorstore`(RAG 필요) / `casual_talk`(일상 대화)로 자동 라우팅, ② 검색된 문서가 실제로 관련 있는지 yes/no로 평가해 필터링(`grade_documents`), ③ 필터링된 문서로 최종 답변 생성 |
| `data.zip` | 서울 도시계획 문서(2040 서울도시기본계획 PDF, chap10과 동일 파일). 압축 해제해서 `data/` 폴더에 두고 사용 |
| `chroma_store/` | `build_vectorstore.py` 실행 시 자동 생성되는 통합 벡터 DB. 저장소에 올리지 않음(`.gitignore`) |

**뉴욕시 도시계획 문서(OneNYC 2050 Strategic Plan)는 저장소에 포함하지 않음** — 용량이 커서(약 50MB) zip에도 넣지 않았음. 실행하려면 "OneNYC 2050 Strategic Plan PDF"를 검색해 `nyc.gov` 공식 출처에서 직접 다운로드한 뒤 `data/OneNYC.pdf`로 저장해야 함.
