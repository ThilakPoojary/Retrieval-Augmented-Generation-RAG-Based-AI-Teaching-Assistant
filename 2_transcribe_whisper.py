import whisper
import os
import json

model=whisper.load_model("small")

files=os.listdir("audios")
chunks=[]

for audio in files:

    number=audio.split(" ")[0]
    
    result=model.transcribe(audio=f"audios/{audio}",
                            language="hi",
                            task="translate",
                           )

    
    for segment in result["segments"]:
        chunks.append({"number":number,"start":segment["start"],"end":segment["end"],"text":segment["text"]})

    chunks_with_metadata={"chunks":chunks,"text":result["text"]}

    
    with open(f"jsons/{audio}.json","w",encoding="utf-8") as f:
        json.dump(chunks_with_metadata,f)