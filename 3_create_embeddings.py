import requests
import os
import joblib
import json
import pandas as pd
import  numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings(text_list):
    f=requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input":text_list})

    embedding=f.json()["embeddings"]
    return embedding


json_files=os.listdir("jsons")

my_dicts=[]

chunk_id=0



for js in json_files:
    with open(f"jsons/{js}") as r:
        content=json.load(r)

    embedding=create_embeddings([c['text'] for c in content['chunks']])
    print(f"Embedding created for ----> {js}")

    for i,chunk in enumerate(content['chunks']):
        chunk['chunk_id']=chunk_id
        chunk['embedding']=embedding[i]
        chunk_id+=1

        my_dicts.append(chunk)
        

print(my_dicts)
        

        
    
    
    
data=pd.DataFrame.from_records(my_dicts)
data


joblib.dump(data,'embeddings.joblib')