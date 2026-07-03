import os
import pandas as pd
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pyannote.audio import Pipeline

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


def diarize_to_rttm(audio_file_path: str, rttm_file_path: str):
    """화자 분리 실행 후 결과를 RTTM 파일로 저장"""
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGING_FACE_TOKEN,
    )

    diarization = pipeline(audio_file_path)

    with open(rttm_file_path, "w", encoding='utf-8') as rttm:
        diarization.write_rttm(rttm)

    return diarization


def rttm_to_grouped_csv(rttm_file_path: str, csv_file_path: str):
    """RTTM 파일을 읽어 같은 화자의 연속 발화끼리 묶은 뒤 CSV로 저장"""
    df_rttm = pd.read_csv(
        rttm_file_path,
        sep=' ',
        header=None,
        names=['type', 'file', 'chnl', 'start', 'duration', 'C1', 'C2', 'speaker_id', 'C3', 'C4'],
    )

    # 발화 시작 + 길이를 더해 끝난 시간(end) 계산
    df_rttm['end'] = df_rttm['start'] + df_rttm['duration']

    # 연속된 발화를 같은 번호로 묶기 위한 number 컬럼
    df_rttm["number"] = None
    df_rttm.at[0, "number"] = 0
    for i in range(1, len(df_rttm)):
        if df_rttm.at[i, "speaker_id"] != df_rttm.at[i - 1, "speaker_id"]:
            df_rttm.at[i, "number"] = df_rttm.at[i - 1, "number"] + 1
        else:
            df_rttm.at[i, "number"] = df_rttm.at[i - 1, "number"]

    # 같은 화자끼리 묶어서 시작/끝 시간 정리
    df_grouped = df_rttm.groupby("number").agg(
        start=pd.NamedAgg(column='start', aggfunc='min'),
        end=pd.NamedAgg(column='end', aggfunc='max'),
        speaker_id=pd.NamedAgg(column='speaker_id', aggfunc='first'),
    )
    df_grouped["duration"] = df_grouped["end"] - df_grouped["start"]
    df_grouped = df_grouped.reset_index(drop=True)

    df_grouped.to_csv(csv_file_path, sep=',', index=False)
    return df_grouped


if __name__ == '__main__':
    audio_path = os.path.join(BASE_DIR, 'data', '싼기타_비싼기타.mp3')
    rttm_path = os.path.join(BASE_DIR, 'data', '싼기타_비싼기타.rttm')
    csv_path = os.path.join(BASE_DIR, 'data', '싼기타_비싼기타_rttm.csv')

    diarize_to_rttm(audio_path, rttm_path)
    df_grouped = rttm_to_grouped_csv(rttm_path, csv_path)
    print(df_grouped)
