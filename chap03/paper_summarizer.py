from openai import OpenAI
from dotenv import load_dotenv
import os
import pymupdf

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


def pdf_to_text(pdf_file_path: str) -> str:
    """PDF를 열어 텍스트로 변환하되, 페이지 위/아래(헤더·푸터)는 제외하고 본문만 추출"""
    doc = pymupdf.open(pdf_file_path)

    header_height = 80  # 페이지 상단에서 이만큼(px)은 헤더로 보고 제외
    footer_height = 80  # 페이지 하단에서 이만큼(px)은 푸터로 보고 제외
    full_text = ''

    for page in doc:
        rect = page.rect  # 페이지 크기(가로/세로) 가져오기
        # clip=(왼쪽, 위, 오른쪽, 아래) 범위 안의 텍스트만 추출 -> 헤더/푸터 영역은 건너뜀
        text = page.get_text(clip=(0, header_height, rect.width, rect.height - footer_height))
        full_text += text + '\n-----------------------------------\n'

    # 전처리된 텍스트를 파일로도 저장해둠 (결과 확인용)
    txt_file_path = pdf_file_path.replace('.pdf', '_pre.txt')
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    return txt_file_path


def summarize_txt(file_path: str) -> str:
    """텍스트 파일을 읽어서 GPT에게 정해진 포맷으로 요약을 요청"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # GPT에게 역할과 출력 포맷을 명확히 지정하는 시스템 프롬프트
    system_prompt = f'''
너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고,
주요 내용을 요약하라.

작성해야 하는 포맷은 다음과 같다.

# 제목

## 저자의 문제 인식 및 주장 (15문장 이내)

## 저자 소개

================ 이하 텍스트 ================

{txt}
'''

    response = client.chat.completions.create(
        model="openai/gpt-4o",
        temperature=0.1,  # 요약처럼 정확성이 중요한 작업은 낮게 설정
        messages=[
            {"role": "system", "content": system_prompt},
        ]
    )

    return response.choices[0].message.content


def summarize_pdf(pdf_file_path: str, output_file_path: str):
    """PDF 경로만 넣으면 텍스트 추출 -> 요약 -> 파일 저장까지 한번에 처리"""
    txt_file_path = pdf_to_text(pdf_file_path)
    summary = summarize_txt(txt_file_path)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(summary)


if __name__ == '__main__':
    # 실행 위치(작업 디렉터리)가 어디든 상관없이, 이 파일 기준으로 data 폴더를 찾도록 함
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    pdf_file_path = os.path.join(data_dir, "과정기반 작물모형을 이용한 구축.pdf")
    output_file_path = os.path.join(data_dir, "summary.txt")

    summarize_pdf(pdf_file_path, output_file_path)
    print("PDF 텍스트 추출 및 요약본 저장이 모두 완료되었습니다.")
