"""음성을 텍스트로 변환하기 (STT) - OpenRouter 오디오 채팅 모델 사용

OpenRouter는 Whisper 전용 엔드포인트(client.audio.transcriptions)를 지원하지 않지만,
오디오를 입력으로 받는 멀티모달 채팅 모델(gemini-2.5-flash 등)에 오디오를 넣고
"받아써줘"라고 요청하면 STT가 된다. 로컬 모델·별도 API 키가 필요 없다.
"""
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 오디오 입력을 지원하는 멀티모달 모델 (받아쓰기 품질이 로컬 whisper-tiny보다 훨씬 좋음)
AUDIO_MODEL = "google/gemini-2.5-flash"


def audio_to_text(audio_file_path: str, instruction: str) -> str:
    """오디오 파일을 채팅 모델에 넣어 지시(instruction)대로 텍스트를 받아온다."""
    with open(audio_file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    audio_format = os.path.splitext(audio_file_path)[1].lstrip(".")  # 'mp3', 'wav' 등

    response = client.chat.completions.create(
        model=AUDIO_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
            ],
        }],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    audio_path = os.path.join(BASE_DIR, "data", "lsy_audio_2023_58s.mp3")

    print("===== 1. 받아쓰기 (한국어 그대로) =====")
    print(audio_to_text(audio_path, "이 오디오 내용을 한국어로 그대로 받아써줘. 받아쓴 내용만 출력."))

    print("\n===== 2. 영어로 번역 =====")
    print(audio_to_text(audio_path, "이 오디오 내용을 영어로 번역해서 받아써줘. 번역한 내용만 출력."))
