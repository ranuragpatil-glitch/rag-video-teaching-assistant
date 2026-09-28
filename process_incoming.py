import joblib
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

from preprocessed_json import create_embedding

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2"
TOP_RESULTS = 5


def inference(prompt):
    r = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    if not r.ok:
        raise RuntimeError(f"Ollama generate request failed ({r.status_code}): {r.text}")

    data = r.json()
    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return data["response"]


def main():
    df = joblib.load("embedding.joblib")

    incoming_query = input("Ask a Question: ")
    question_embedding = create_embedding([incoming_query])[0]

    # Find similarity of the question embedding with every chunk embedding
    similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()
    max_indx = similarities.argsort()[::-1][:TOP_RESULTS]
    new_df = df.iloc[max_indx]
    result_columns = [column for column in ("title", "number", "start", "end", "text") if column in new_df.columns]

    prompt = f'''Here are relevant video subtitle chunks with available timestamps in seconds and text:

{new_df[result_columns].to_json(orient="records")}
---------------------------------
"{incoming_query}"
Answer questions related to these chunks in a natural way and cite the relevant timestamp.
The video title and number are not included in the available data, so do not guess them. If the question
is unrelated, say you can only answer questions related to the course.
'''
    with open("prompt.txt", "w") as f:
        f.write(prompt)

    response = inference(prompt)
    print("\nAnswer:\n")
    print(response)

    with open("response.txt", "w") as f:
        f.write(response)


if __name__ == "__main__":
    main()
