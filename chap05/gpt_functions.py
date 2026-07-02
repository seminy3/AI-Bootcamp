from datetime import datetime
import pytz


def get_current_time(timezone: str = 'Asia/Seoul'):
    """주어진 타임존의 현재 날짜/시간을 문자열로 반환.
    GPT는 스스로 현재 시각을 알 수 없기 때문에, 이 함수를 GPT 대신 호출해서 결과를 알려줘야 함."""
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    now_timezone = f'{now} {timezone}'
    print(now_timezone)
    return now_timezone


# GPT에게 "이런 함수가 있다"고 알려주는 설명서(딕셔너리).
# GPT는 이 설명을 보고, 사용자 질문에 시간 정보가 필요하면 이 함수를 호출하라고 요청함.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "해당 타임존의 날짜와 시간을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "현재 날짜와 시간을 반환할 타임존을 입력하세요. (예: Asia/Seoul)",
                    },
                },
                "required": ["timezone"],
            },
        },
    },
]

if __name__ == '__main__':
    get_current_time('America/New_York')
