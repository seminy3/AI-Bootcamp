# 9장 - 음성을 텍스트로 정리하기

| 파일 | 설명 |
|---|---|
| `whisper_local.py` / `.ipynb` | HuggingFace transformers의 로컬 Whisper(`whisper-tiny`) 모델로 음성 파일을 텍스트로 변환. 청크별 타임스탬프를 표로 정리해 CSV로 저장 (OpenRouter가 Whisper API를 지원하지 않아 API 대신 로컬 모델 사용) |
| `speaker_diarization.py` / `.ipynb` | `pyannote/speaker-diarization-3.1`로 화자 분리. 결과를 RTTM 파일로 저장하고, 연속된 동일 화자 발화를 하나로 묶어 CSV로 정리 (HuggingFace 토큰 필요) |
| `merge_speaker_text.py` / `.ipynb` | 위 두 결과(발화 텍스트 + 화자 구간)를 시간 겹침 기준으로 매칭해 "누가 뭐라고 말했는지" 합친 최종 CSV 생성 |
| `data.zip` | STT/화자분리 테스트용 원본 오디오(`lsy_audio_2023_58s.mp3`, `싼기타_비싼기타.mp3`). 압축 해제해서 `data/` 폴더에 두고 사용 |

STT/화자분리 결과 CSV·RTTM 파일은 스크립트를 돌리면 재생성되므로 저장소에 포함하지 않음.
