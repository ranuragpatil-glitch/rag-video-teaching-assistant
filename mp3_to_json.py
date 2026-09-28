import json
import os

import whisper

AUDIO_DIR = "audios"
OUTPUT_DIR = "jsons"
MODEL_SIZE = "small"
LANGUAGE = "hindi"  # language spoken in the audio
TASK = "translate"  # "translate" -> English text, "transcribe" -> same language as audio


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = whisper.load_model(MODEL_SIZE)

    for file in sorted(os.listdir(AUDIO_DIR)):
        if not file.lower().endswith(".mp3"):
            continue

        print(f"Transcribing {file}")
        result = model.transcribe(
            audio=os.path.join(AUDIO_DIR, file),
            language=LANGUAGE,
            task=TASK,
            word_timestamps=False,
        )

        chunks = [
            {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
            for segment in result["segments"]
        ]

        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()
