# 12장 - 랭그래프(LangGraph)

| 파일 | 설명 |
|---|---|
| `langgraph_basic.py` / `.ipynb` | `StateGraph`로 State/Node/Edge 기본 구조를 만드는 가장 단순한 그래프. `invoke`, 메시지 이어붙이기, `stream` 방식 출력까지 |
| `langgraph_memory.py` / `.ipynb` | `MemorySaver` 체크포인터 + `thread_id`로 대화 내용을 자동으로 기억하는 터미널 챗봇 |
| `langgraph_tools_agent.py` / `.ipynb` | `bind_tools` + `BasicToolNode` + 조건부 엣지(`route_tools`)로 구현한 도구 호출 에이전트(시간 조회, 웹 검색). 하나의 시스템 프롬프트로 "검색 → 목차 작성 → 재검색 → 기사 작성"을 반복 수행하는 심층 분석 기사 작성 테스트 포함 |

각 노트북의 "그래프 구조 시각화" 셀을 실행하면 `display(Image(graph.get_graph().draw_mermaid_png()))`로 노드/엣지 다이어그램을 바로 볼 수 있음.
