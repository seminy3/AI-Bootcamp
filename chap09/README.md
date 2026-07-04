# 9장 - 음성을 텍스트로 정리하기

음성 파일을 텍스트로 변환(STT)하고, 화자를 구분해 회의록 형태로 정리한다. OpenRouter 키 하나로 처리하는 **최신 방식**과, 로컬에서 직접 모델을 구동하는 **교재 원본 방식**을 모두 다룬다.

## 1. Whisper API로 받아쓰기 (추천)

OpenAI의 음성 인식 전용 모델 **Whisper**(`openai/whisper-large-v3`)를 OpenRouter로 호출한다. 로컬 모델(torch·pyannote) 설치나 별도 API 키 없이 받아쓰기 품질이 매우 높다.

| 파일 | 설명 |
|---|---|
| `whisper_api.py` / `.ipynb` | Whisper API로 음성 → 텍스트 변환 + 영어 번역. OpenRouter는 요청 형식이 정품 OpenAI와 달라(파일 업로드 X, JSON+base64 O) SDK 대신 `requests`로 직접 호출한다. 번역은 Whisper에 번역 엔드포인트가 없어 받아쓴 텍스트를 GPT로 번역 |

> Whisper는 '받아쓰기 전용'이라 **화자 구분·타임스탬프는 제공하지 않는다.** 그 기능은 아래 2번 참고.

## 2. 화자 구분 + 타임스탬프 (회의록 만들기)

Whisper가 못 하는 "누가 / 언제 / 뭐라고"를, 오디오를 이해하는 멀티모달 채팅 모델 **Gemini**(`google/gemini-2.5-flash`)로 한 번에 처리한다.

| 파일 | 설명 |
|---|---|
| `stt_with_speakers.py` / `.ipynb` | 2인 이상 대화를 화자 구분 + 시작/종료 시각과 함께 JSON으로 받아 CSV(회의록)로 저장. 받아쓰기 + 화자 분리를 **API 호출 한 번**으로 해결 |

<br>

## 3. 로컬 모델 활용하기 (교재 원본 방식)

인터넷 연결이나 외부 API 의존도를 낮추고 로컬 환경에서 직접 모델을 구동하여 음성을 처리하는 방식이다.

| 파일 | 설명 |
|---|---|
| `whisper_local.py` / `.ipynb` | HuggingFace transformers의 로컬 Whisper(`whisper-tiny`) 모델로 음성 파일을 텍스트로 변환한다. 청크별 타임스탬프를 표로 정리해 CSV로 저장 |
| `speaker_diarization.py` / `.ipynb` | `pyannote/speaker-diarization-3.1`로 화자를 분리한다. 결과를 RTTM 파일로 저장하고, 연속된 동일 화자의 발화를 하나로 묶어 CSV로 정리 (HuggingFace 토큰 필요, `pyannote.audio 4.x` 기준 `token=` 인자 사용) |
| `merge_speaker_text.py` / `.ipynb` | 위 두 결과(Whisper의 발화 텍스트 + pyannote의 화자 구간)를 시간 겹침 기준으로 매칭해, "누가 뭐라고 말했는지" 최종 합친 CSV를 생성 |

<br>

## 공통 데이터

| 파일 | 설명 |
|---|---|
| `data.zip` | 실습용 원본 오디오(`lsy_audio_2023_58s.mp3` 1인, `싼기타_비싼기타.mp3` 2인). 압축 해제해서 `data/` 폴더에 두고 사용 |

> **참고**: 교재 원본은 로컬 Whisper(transformers) + pyannote(화자 분리) + HuggingFace 토큰 + ffmpeg + torch를 썼지만, OpenRouter의 Whisper API와 Gemini 오디오 모델을 쓰면 이 무거운 로컬 의존성 없이 더 간단하게 연동할 수 있다. STT 결과 CSV·RTTM 파일은 실행 시 재생성되므로 저장소에 포함하지 않는다.
