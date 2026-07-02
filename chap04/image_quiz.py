from glob import glob
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import json

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'images')
AUDIO_DIR = os.path.join(BASE_DIR, 'data', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_quiz(image_path: str, n_trial: int = 0, max_trial: int = 3):
    """이미지 하나를 보고 토익 리스닝 스타일의 4지선다 퀴즈를 GPT가 생성.
    GPT가 가끔 원하는 포맷("Listening:")을 안 지킬 때가 있어서, 지킬 때까지 재시도(최대 3회)."""
    if n_trial >= max_trial:
        raise Exception("Failed to generate a quiz.")

    base64_image = encode_image(image_path)

    quiz_prompt = """
제공된 이미지를 바탕으로, 다음과 같은 양식으로 퀴즈를 만들어주세요.
정답은 1~4 중 하나만 해당하도록 출제하세요.
토익 리스닝 문제 스타일로 문제를 만들어주세요.
아래는 예시입니다.
----- 예시 -----

Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
- (1) 베이커리에서 사람들이 빵을 사고 있는 모습이 담겨 있습니다.
- (2) 맨 앞에 서 있는 사람은 빨간색 셔츠를 입고 있습니다.
- (3) 기차를 타기 위해 줄을 서 있는 사람들이 있습니다.
- (4) 점원은 노란색 티셔츠를 입고 있습니다.

Listening: Which of the following descriptions of the image is incorrect?
- (1) It shows people buying bread at a bakery.
- (2) The person standing at the front is wearing a red shirt.
- (3) There are people lining up to take a train.
- (4) The clerk is wearing a yellow T-shirt.

정답: (4) 점원은 노란색 티셔츠가 아닌 파란색 티셔츠를 입고 있습니다.
(주의: 정답은 1~4 중 하나만 선택되도록 출제하세요.)
======
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": quiz_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=messages,
        )
    except Exception as e:
        print("failed\n" + str(e))
        return image_quiz(image_path, n_trial + 1)

    content = response.choices[0].message.content

    if "Listening:" in content:
        return content, True
    else:
        return image_quiz(image_path, n_trial + 1)


def text_to_speech(text: str, output_path: str, voice: str = "Eve"):
    """텍스트를 음성 mp3 파일로 변환 (듣기평가 음성 생성용)"""
    response = client.audio.speech.create(
        model="x-ai/grok-voice-tts-1.0",
        voice=voice,
        input=text,
        response_format="mp3",
    )
    response.write_to_file(output_path)


def build_listening_quiz():
    """images 폴더의 모든 jpg를 돌면서 퀴즈를 만들고, 영어 문제 부분만 뽑아 mp3 음성까지 생성"""
    txt = ''
    eng_dict = []
    no = 1

    for image_path in glob(os.path.join(IMAGES_DIR, '*.jpg')):
        q, is_succeed = image_quiz(image_path)

        if not is_succeed:
            continue

        divider = f'## 문제 {no}\n\n'
        txt += divider

        filename = os.path.basename(image_path)
        txt += f'![image]({filename})\n\n'
        txt += q + '\n\n-----------------------\n\n'

        # "Listening:" 부터 "정답:" 사이의 영어 문제만 추출
        eng = q.split('Listening: ')[1].split('정답:')[0].strip()
        eng_dict.append({'no': no, 'eng': eng, 'img': filename})

        no += 1

    # 마크다운 파일로 저장 (사람이 눈으로 확인하는 용도)
    with open(os.path.join(IMAGES_DIR, 'image_quiz_eng.md'), 'w', encoding='utf-8') as f:
        f.write(txt)

    # JSON 파일로 저장 (TTS에서 프로그램으로 읽기 위한 용도)
    with open(os.path.join(IMAGES_DIR, 'image_quiz_eng.json'), 'w', encoding='utf-8') as f:
        json.dump(eng_dict, f, ensure_ascii=False, indent=4)

    # 영어 문제 하나하나를 mp3 음성 파일로 변환
    for q in eng_dict:
        text_to_speech(q['eng'], os.path.join(AUDIO_DIR, f"{q['no']}.mp3"))
        print(f"{q['no']}.mp3 생성 완료")

    return eng_dict


if __name__ == '__main__':
    result = build_listening_quiz()
    print(f"\n총 {len(result)}개의 듣기 문제와 음성 파일이 생성되었습니다.")
