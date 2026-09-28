import requests
import os
import json
import pandas as pd
import joblib


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    if not r.ok:
        raise RuntimeError(f"Ollama embedding request failed ({r.status_code}): {r.text}")
    response = r.json()
    if not isinstance(response, dict) or "embeddings" not in response:
        detail = response.get("error", response) if isinstance(response, dict) else response
        raise RuntimeError(f"Unexpected Ollama embedding response: {detail}")
    return response["embeddings"]


def main():
    jsons = sorted(f for f in os.listdir("jsons") if f.endswith(".json"))  # List all the jsons
    my_dicts = []
    chunk_id = 0

    for json_file in jsons:
        with open(f"jsons/{json_file}") as f:
            content = json.load(f)
        print(f"Creating Embeddings for {json_file}")
        chunks = content["chunks"] if isinstance(content, dict) else content
        embeddings = create_embedding([c['text'] for c in chunks])

        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = chunk_id
            chunk['embedding'] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk)
       

    df = pd.DataFrame.from_records(my_dicts)
    joblib.dump(df, 'embedding.joblib')


if __name__ == "__main__":
    main()

