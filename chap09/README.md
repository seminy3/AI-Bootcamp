# 9장 - 음성을 텍스트로 정리하기

이 장에서는 음성 파일을 텍스트로 변환(STT)하고 화자를 분리하는 두 가지 방법을 배웁니다. HuggingFace와 로컬 라이브러리를 활용하는 전통적인 방식과, OpenRouter의 오디오 입력 채팅 모델(`google/gemini-2.5-flash`)을 활용해 복잡한 의존성 없이 API 호출 한 번으로 처리하는 최신 방식을 모두 다룹니다.

## 1. OpenRouter 오디오 모델 활용하기 (추천)
별도의 로컬 모델 설치나 추가 API 키 없이, 기존 OpenRouter 키 하나로 음성 인식부터 화자 구분, 회의록 정리까지 한 번에 처리합니다. 무거운 라이브러리를 설치하지 않아도 되며 받아쓰기 품질이 우수합니다.

| 파일 | 설명 |
|---|---|
| `stt_basic.py` / `.ipynb` | 오디오를 채팅 모델에 넣어 받아쓰기(STT) 및 영어 번역을 수행합니다. OpenRouter에서 지원하지 않는 `client.audio.transcriptions`(Whisper 전용) 대신 `input_audio` 블록으로 오디오를 전달합니다. |
| `stt_with_speakers.py` / `.ipynb` | 2인 이상의 대화를 화자 구분 및 시작/종료 시각과 함께 JSON 구조로 받아와 CSV(회의록)로 저장합니다. STT와 화자 분리를 **API 호출 한 번**으로 해결합니다. |

<br>

## 2. 로컬 모델 활용하기 (기존 방식)
인터넷 연결이나 외부 API 의존도를 낮추고 로컬 환경에서 직접 모델을 구동하여 음성을 처리하는 방식입니다.

| 파일 | 설명 |
|---|---|
| `whisper_local.py` / `.ipynb` | HuggingFace transformers의 로컬 Whisper(`whisper-tiny`) 모델로 음성 파일을 텍스트로 변환합니다. 청크별 타임스탬프를 표로 정리해 CSV로 저장합니다. |
| `speaker_diarization.py` / `.ipynb` | `pyannote/speaker-diarization-3.1`을 사용하여 화자를 분리합니다. 결과를 RTTM 파일로 저장하고, 연속된 동일 화자의 발화를 하나로 묶어 CSV로 정리합니다. (HuggingFace 토큰 필요) |
| `merge_speaker_text.py` / `.ipynb` | 위의 두 결과(Whisper의 발화 텍스트 + pyannote의 화자 구간)를 시간 겹침 기준으로 매칭하여, "누가 뭐라고 말했는지" 최종 합친 CSV를 생성합니다. |

<br>

## 공통 데이터 및 참고사항

| 파일 | 설명 |
|---|---|
| `data.zip` | STT 및 화자분리 실습·테스트용 원본 오디오 파일(`lsy_audio_2023_58s.mp3` 1인 발화, `싼기타_비싼기타.mp3` 2인 대화). 압축을 해제하여 `data/` 폴더에 두고 사용합니다. |

> **참고**: 교재 원본은 로컬 Whisper + pyannote + HuggingFace 토큰 + ffmpeg + torch 등 무겁고 복잡한 로컬 의존성을 사용하지만, OpenRouter의 오디오 모델을 활용하면 이러한 환경 구축 스트레스 없이 더 뛰어난 품질로 연동할 수 있습니다. 실습 스크립트 실행 시 생성되는 STT 결과 CSV 및 RTTM 파일은 재생성 가능하므로 저장소(Git)에는 포함하지 않습니다.

