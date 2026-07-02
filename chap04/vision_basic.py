from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def encode_image(image_path: str) -> str:
    """로컬 이미지 파일을 base64 문자열로 변환 (URL이 없는 내 컴퓨터 사진을 GPT에 보낼 때 사용)"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_image_by_url(image_url: str, question: str = "이 이미지에 대해 설명해주세요.") -> str:
    """인터넷에 있는 이미지(URL)를 GPT에게 설명해달라고 요청"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content


def _to_data_url(image_path: str) -> str:
    """확장자에 맞는 MIME 타입을 붙여서 data URL 문자열을 만듦 (jpg/png 둘 다 지원)"""
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{encode_image(image_path)}"


def describe_image_by_path(image_path: str, question: str = "이 이미지에 대해 설명해주세요.") -> str:
    """내 컴퓨터에 있는 로컬 이미지 파일을 GPT에게 설명해달라고 요청 (base64로 인코딩해서 전달)"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": _to_data_url(image_path)}},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content


def compare_images(image_path_1: str, image_path_2: str, question: str) -> str:
    """로컬 이미지 두 장을 함께 보여주고 비교 질문을 요청"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": _to_data_url(image_path_1)}},
                {"type": "image_url", "image_url": {"url": _to_data_url(image_path_2)}},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'images')

    print("===== ① URL 이미지 설명 =====")
    image_url = "https://images.unsplash.com/photo-1736264335247-8ec5664c8328?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    print(describe_image_by_url(image_url))

    print("\n===== ② 로컬 이미지 설명 (망원동 베이커리) =====")
    print(describe_image_by_path(os.path.join(IMAGES_DIR, "mangwon_bakery.jpg")))

    print("\n===== ③ 두 카페 비교 (선릉 테라로사 vs 로컬스티치 테라로사) =====")
    print(compare_images(
        os.path.join(IMAGES_DIR, "seolleung_terrarosa.jpg"),
        os.path.join(IMAGES_DIR, "local_stitch_terrarosa.jpg"),
        "두 카페의 차이점을 설명해주세요.",
    ))

    print("\n===== ④ GPT Vision 한계 테스트 (OECD 연구개발비 그래프, 해상도별 비교) =====")
    question = "첫번째는 2021년 데이터이고, 두번째는 2022년 데이터입니다. 이 데이터에 대해 설명해주세요. 어떤 변화가 있었나요?"

    print("--- large 해상도 ---")
    print(compare_images(
        os.path.join(IMAGES_DIR, "oecd_rnd_2021_large.png"),
        os.path.join(IMAGES_DIR, "oecd_rnd_2022.png"),
        question,
    ))

    print("\n--- medium(낮은) 해상도 ---")
    print(compare_images(
        os.path.join(IMAGES_DIR, "oecd_rnd_2021_medium.png"),
        os.path.join(IMAGES_DIR, "oecd_rnd_2022.png"),
        question,
    ))
