# 14장 - 랭그래프로 목차를 작성하는 멀티에이전트 만들기

사용자와 대화하며 책의 목차(outline)를 작성하는 멀티에이전트를 단계적으로 만든다.
`supervisor`가 상황을 판단해 목차를 쓰는 `content_strategist`와 사용자와 대화하는 `communicator` 중 누구에게 일을 시킬지 결정하고, 웹 검색·RAG로 자료를 보강한다.

| 파일 | 설명 |
|---|---|
| `utils.py` | `save_state`(대화·작업 이력을 `data/state.json`으로 저장), `get_outline`/`save_outline`(목차를 `data/outline.md`로 읽고 씀) |
| `models.py` | supervisor의 라우팅 판단을 구조화하는 Pydantic `Task` 모델 (`agent`/`done`/`description`/`done_at`) |
| `book_writer_basic.py` / `.ipynb` | **1단계.** `content_strategist → communicator` 로만 이어지는 단순 그래프. 목차를 작성하고 사용자에게 진행상황을 보고 |
| `book_writer_supervisor.py` / `.ipynb` | **2단계.** `supervisor`를 추가. `with_structured_output(Task)`로 다음 agent를 정확히 선택하고, `add_conditional_edges`로 라우팅. `task_history`로 작업 이력 관리 |
| `tools.py` / `.ipynb` | **3단계.** Tavily 웹 검색(`web_search`) → JSON 저장 → 랭체인 `Document` 변환 → 청킹 → Chroma 벡터 DB 저장, 그리고 벡터 검색(`retrieve`) 도구 |

실행 가능한 세 파일은 `.ipynb` 버전도 제공한다. `utils.py`·`models.py`는 다른 파일이 `import`해서 쓰는 부품 모듈이라 `.py`로만 둔다. 노트북은 `__file__`이 없으므로 경로를 `os.getcwd()` 기준으로 잡으니 **반드시 `chap14` 폴더에서 열어 실행**한다.

## 실행

루트 README의 환경 설정을 마치고 `.env`에 `OPENAI_API_KEY`(OpenRouter), `TAVILY_API_KEY`를 설정한 뒤:

```bash
# 1단계: 기본 2-에이전트
.venv/bin/python chap14/book_writer_basic.py

# 2단계: supervisor 라우팅
.venv/bin/python chap14/book_writer_supervisor.py

# 3단계: 웹 검색 + RAG 벡터 DB 구축/검색
.venv/bin/python chap14/tools.py
```

`book_writer_*.py`는 터미널에서 대화형으로 실행된다. `HYBE와 JYP 경영 전략 비교하는 책 써줘`처럼 입력하면 목차가 작성되고, `exit`/`quit`/`q`로 종료한다.

## 참고

- 교재는 OpenAI API(`gpt-4o`, `text-embedding-3-large`)와 `sec01/sec02/sec03` 폴더 구조를 쓰지만, 이 저장소는 다른 챕터와 동일하게 **OpenRouter**(`openai/gpt-4o` + `base_url`)를 사용하고 파일을 챕터 폴더에 평평하게 둔다.
- 실행하면 `data/`에 `state.json`, `outline.md`, `resources_*.json`, `chroma_store/`가 생성된다. 모두 재생성 가능하므로 저장소에는 커밋하지 않는다(`.gitignore`).
- 새로 실행해 처음부터 목차를 만들려면 `data/`의 `outline.md`(및 필요시 `state.json`)를 삭제한 뒤 실행한다
