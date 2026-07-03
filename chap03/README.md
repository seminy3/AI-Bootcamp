# 3장 - GPT API를 활용한 업무 자동화 (PDF 논문 요약)

| 파일 | 설명 |
|---|---|
| `paper_summarizer.py` / `.ipynb` | PDF 논문을 텍스트로 추출(`pdf_to_text`, PyMuPDF로 머리말/꼬리말 제거) → GPT로 정해진 포맷에 맞춰 요약(`summarize_txt`) → 전체 파이프라인(`summarize_pdf`) |
| `data.zip` | 요약 실습에 사용한 원본 논문 PDF (부트캠프 제공 자료). 압축 해제해서 `data/` 폴더에 두고 사용 |

`summary.txt` 등 실행 결과물은 스크립트를 돌리면 재생성되므로 저장소에 포함하지 않음.
