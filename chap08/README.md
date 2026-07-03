# 8장 - 인터넷 검색 챗봇 만들기

| 파일 | 설명 |
|---|---|
| `duckduckgo_search.py` / `.ipynb` | 무료 DuckDuckGo 검색(`DuckDuckGoSearchResults`)으로 기본 검색·뉴스 소스 한정 검색 → 검색 결과 링크에서 BeautifulSoup으로 기사 본문 스크래핑 → 검색 결과 기반 GPT 답변(RAG 패턴). `search_with_retry`로 ddgs 라이브러리의 간헐적 TLS 오류에 대비한 재시도 로직 포함 |
| `tavily_search.py` / `.ipynb` | 유료(1000회 무료) Tavily 검색 API로 더 안정적인 검색 결과를 얻고, 출처를 명시한 신문기사 형식 보고서를 GPT로 생성 |
| `youtube_summarizer.py` / `.ipynb` | 유튜브 검색(`YoutubeSearch`) → 영상 자막 로드(`YoutubeLoader`) → GPT로 한국어 요약 (유튜브 IP 차단 시 일정 시간 후 재시도 필요) |
