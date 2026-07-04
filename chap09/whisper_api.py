"""음성을 텍스트로 변환하기 - OpenRouter의 Whisper API 사용

OpenRouter는 OpenAI 정품과 요청 형식이 다르다:
  - 정품 OpenAI: 파일을 multipart/form-data로 업로드 (client.audio.transcriptions)
  - OpenRouter : JSON 바디에 base64 오디오를 담아 전송
그래서 OpenAI SDK로는 호출이 안 되고, requests로 직접 HTTP 요청을 보내야 한다.

Whisper는 '받아쓰기 전용' 모델이라 화자 구분·타임스탬프는 제공하지 않는다.
(그 기능은 stt_with_speakers.py 의 Gemini 오디오 모델 참고)
"""
import os
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# 번역 단계에서 쓸 채팅 클라이언트 (Whisper는 한국어→영어 번역 엔드포인트가 없음)
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def whisper_transcribe(audio_file_path: str) -> str:
    """OpenRouter의 Whisper API로 음성을 텍스트로 변환한다."""
    with open(audio_file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    audio_format = os.path.splitext(audio_file_path)[1].lstrip(".")  # 'mp3', 'wav' 등

    response = requests.post(
        f"{BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "openai/whisper-large-v3",
            "input_audio": {"data": audio_b64, "format": audio_format},
        },
    )
    response.raise_for_status()
    result = response.json()
    # usage에 처리 시간과 비용이 함께 온다
    usage = result.get("usage", {})
    if usage:
        print(f"(처리: {usage.get('seconds')}초, 비용: ${usage.get('cost')})")
    return result["text"]


def translate_to_english(text: str) -> str:
    """받아쓴 텍스트를 GPT로 영어 번역한다 (Whisper 번역 엔드포인트가 없어 텍스트 번역으로 대체)."""
    r = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": f"다음을 자연스러운 영어로 번역해줘. 번역문만 출력:\n\n{text}"}],
    )
    return r.choices[0].message.content


if __name__ == "__main__":
    audio_path = os.path.join(BASE_DIR, "data", "lsy_audio_2023_58s.mp3")

    print("===== 1. Whisper API 받아쓰기 =====")
    text = whisper_transcribe(audio_path)
    print(text)

    print("\n===== 2. 영어 번역 =====")
    print(translate_to_english(text))
