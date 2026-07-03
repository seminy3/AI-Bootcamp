"""화자 구분 + 타임스탬프까지 한 번에 (회의록 만들기)

OpenRouter의 오디오 채팅 모델(gemini-2.5-flash)에 대화 음성을 넣으면,
"누가(화자) / 언제(시작~종료 시각) / 뭐라고(텍스트)"를 한 번의 호출로 정리해준다.
기존에는 로컬 Whisper(STT) + pyannote(화자 분리) 두 모델을 따로 돌려야 했지만,
이제 API 호출 한 번으로 끝난다.
"""
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import json
import re
import pandas as pd

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

AUDIO_MODEL = "google/gemini-2.5-flash"

INSTRUCTION = """이 오디오는 여러 사람의 대화입니다.
각 발언을 화자별로 구분하고, 시작/종료 시각을 초 단위 숫자(소수점 허용)로 표기해서 JSON 배열로만 출력하세요.
형식: [{"start": 1.7, "end": 49.8, "speaker": "화자A", "text": "..."}]
JSON 외의 다른 텍스트나 코드블록 표시는 절대 쓰지 마세요."""


def transcribe_with_speakers(audio_file_path: str, output_csv_path: str) -> pd.DataFrame:
    """대화 음성을 화자·시각·텍스트로 정리해 DataFrame과 CSV로 반환한다."""
    with open(audio_file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    audio_format = os.path.splitext(audio_file_path)[1].lstrip(".")

    response = client.chat.completions.create(
        model=AUDIO_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
            ],
        }],
    )

    # 혹시 ```json ... ``` 코드블록으로 감싸서 오면 제거하고 파싱
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.M).strip()
    data = json.loads(content)

    df = pd.DataFrame(data, columns=["start", "end", "speaker", "text"])
    df.to_csv(output_csv_path, index=False)
    return df


if __name__ == "__main__":
    audio_path = os.path.join(BASE_DIR, "data", "싼기타_비싼기타.mp3")
    output_path = os.path.join(BASE_DIR, "data", "싼기타_비싼기타_result.csv")

    df = transcribe_with_speakers(audio_path, output_path)
    print(f"총 발언 수: {len(df)}\n")
    print(df.to_string(index=False, max_colwidth=50))
