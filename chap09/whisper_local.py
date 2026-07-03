import os
import torch
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def whisper_to_dataframe(result, output_file_path):
    """위스퍼 결과(청크별 타임스탬프+텍스트)를 표(DataFrame)로 정리하고 CSV로 저장"""
    start_end_text = []

    for chunk in result["chunks"]:
        start = chunk["timestamp"][0]
        end = chunk["timestamp"][1]
        text = chunk["text"].strip()
        start_end_text.append([start, end, text])

    df = pd.DataFrame(start_end_text, columns=["start", "end", "text"])
    df.to_csv(output_file_path, index=False, sep="|")
    return df


def whisper_stt(audio_file_path: str, output_file_path: str = "./output.csv"):
    """로컬에 다운로드한 위스퍼 모델로 음성 파일을 텍스트로 변환"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-tiny"  # 가벼운 모델로 CPU에서도 빠르게 테스트 가능

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,  # 청크별로 타임스탬프 반환
        chunk_length_s=10,       # 입력 오디오 10초씩 나누기
        stride_length_s=2,       # 2초씩 겹치도록 청크 나누기
    )

    result = pipe(audio_file_path)
    df = whisper_to_dataframe(result, output_file_path)

    return result, df


if __name__ == '__main__':
    audio_path = os.path.join(BASE_DIR, 'data', 'lsy_audio_2023_58s.mp3')
    output_path = os.path.join(BASE_DIR, 'data', 'lsy_audio_2023_58s.csv')

    result, df = whisper_stt(audio_path, output_path)
    print(df)
