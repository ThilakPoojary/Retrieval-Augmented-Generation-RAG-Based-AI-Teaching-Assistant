import requests
import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_embeddings(text_list):
    f = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = f.json()["embeddings"]
    return embedding



df = joblib.load('embeddings.joblib')



incoming_query = input("Ask a question: ")
question_embedding = create_embeddings(incoming_query)[0]


similarity = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarity)


max_idx = similarity.argsort()[::-1][0:3]
print(max_idx)


new_df = df.loc[max_idx]
print(new_df)


for index, item in new_df.iterrows():
    print(index, item["text"], item["start"], item["end"])


prompt = f'''I am teaching about AI, you can predict which chunk is in which video

{new_df[["number","chunk_id","start","end","text"]].to_json(orient="records")}



"{incoming_query}"

**3. INSTRUCTIONS & OUTPUT FORMAT:**
* **Analyze:** Use ONLY the provided CONTEXT to formulate your answer.
* **Answer:** Address the USER QUERY directly.
* **Source Citation:** After your answer, you MUST provide a concise list of sources.
* **Format for Citations:** For each relevant chunk, provide the **Video Number**, the **Start Time**, and the **End Time** where the content is discussed.

User asked these questions related to video chunks, you have to answer where and how much content is taught in which video and at what timestamp and guide the user to go to the particular video and say in which video this content exists. If the user asks any unrelated questions then you can guide them to ask questions related to the course only.
f the USER QUERY cannot be answered using the provided CONTEXT, state clearly that the information is not in the course material and ask the user to refine their question.
'''



with open("prompt.txt", "w") as f:
    f.write(prompt)



def inference(prompts):
    r = requests.post("http://localhost:11434/api/generate", json={
        ##"model": "deepseek-r1",
        "model": "mistral",
        "prompt": prompts,
        "stream": False,
        
    })
    response = r.json()
    print(response)
    return response
    


response = inference(prompt)['response']
print(response)



with open("response.txt", "w") as f:
    f.write(response)