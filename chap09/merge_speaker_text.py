import os
import pandas as pd
from whisper_local import whisper_stt
from speaker_diarization import diarize_to_rttm, rttm_to_grouped_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def match_speaker(text_start, text_end, df_speakers):
    """발화 텍스트 구간과 가장 많이 겹치는 화자를 찾아서 반환"""
    best_speaker = None
    best_overlap = 0

    for _, row in df_speakers.iterrows():
        overlap = min(text_end, row['end']) - max(text_start, row['start'])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = row['speaker_id']

    return best_speaker


def transcribe_with_speakers(audio_file_path: str, output_csv_path: str):
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    rttm_path = os.path.join(BASE_DIR, 'data', f'{base_name}.rttm')
    speaker_csv_path = os.path.join(BASE_DIR, 'data', f'{base_name}_rttm.csv')
    text_csv_path = os.path.join(BASE_DIR, 'data', f'{base_name}_text.csv')

    # 1) 화자 분리 (이미 결과가 있으면 재사용)
    if not os.path.exists(speaker_csv_path):
        diarize_to_rttm(audio_file_path, rttm_path)
        df_speakers = rttm_to_grouped_csv(rttm_path, speaker_csv_path)
    else:
        df_speakers = pd.read_csv(speaker_csv_path)

    # 2) 음성을 텍스트로 변환
    _, df_text = whisper_stt(audio_file_path, text_csv_path)

    # 3) 각 발화 텍스트 구간에 가장 많이 겹치는 화자를 매칭
    df_text['speaker_id'] = df_text.apply(
        lambda row: match_speaker(row['start'], row['end'], df_speakers), axis=1
    )

    df_text.to_csv(output_csv_path, index=False, sep='|')
    return df_text


if __name__ == '__main__':
    audio_path = os.path.join(BASE_DIR, 'data', '싼기타_비싼기타.mp3')
    output_path = os.path.join(BASE_DIR, 'data', '싼기타_비싼기타_merged.csv')

    df = transcribe_with_speakers(audio_path, output_path)
    print(df)
