import os
import subprocess
import json
import requests
import pandas as pd
import numpy as np
import whisper
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename  # <-- THE FIX IS HERE

app = Flask(__name__)
# YOU MUST CHANGE THIS KEY
app.secret_key = "your_super_secret_key_123" 

# --- Folders ---
os.makedirs("videos", exist_ok=True)
os.makedirs("audios", exist_ok=True)
os.makedirs("jsons", exist_ok=True)

# --- Constants (From your original model) ---
EMBEDDING_API_URL = "http://localhost:11434/api/embed"
LLM_API_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"
TOP_K = 3

# --- Load Models ---
print("Loading Whisper model...")
try:
    whisper_model = whisper.load_model("small")
    print("Whisper model loaded.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

# --- Global App State ---
# We store the data in memory, not just the transcript
current_df = None
current_embeddings = None
current_video_name = None

# === HELPER FUNCTIONS (From your original model) ===

def create_embeddings_api(text_list):
    """Generates embeddings via the local API."""
    try:
        f = requests.post(EMBEDDING_API_URL, json={
            "model": EMBEDDING_MODEL,
            "input": text_list
        })
        f.raise_for_status()
        embedding = f.json()["embeddings"]
        return embedding
    except Exception as e:
        flash(f"Error connecting to embedding model: {e}", "error")
        print(f"Error creating embedding: {e}")
        return None

def inference_api(prompt_text):
    """Sends a prompt to the local LLM and gets a response."""
    try:
        r = requests.post(LLM_API_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt_text,
            "stream": False
        })
        r.raise_for_status()
        response = r.json()
        return response['response']
    except Exception as e:
        flash(f"Error connecting to LLM: {e}", "error")
        print(f"Error during LLM inference: {e}")
        return None

# === Flask Routes ===

@app.route("/")
def home():
    """Renders the main page."""
    return render_template(
        "index.html", 
        video_name=current_video_name
    )


@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Handles video upload AND runs the full pipeline 
    (Extract, Transcribe, AND Embed).
    """
    global current_df, current_embeddings, current_video_name

    if "file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    video_path = os.path.join("videos", filename)
    audio_path = os.path.join("audios", filename.rsplit(".", 1)[0] + ".mp3")
    json_path = os.path.join("jsons", filename + ".json")
    
    file.save(video_path)

    # --- Step 1: Extract Audio (Your code) ---
    print("Step 1: Extracting audio...")
    try:
        subprocess.run(["ffmpeg", "-i", video_path, audio_path, "-y"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        flash(f"FFMPEG Error: {e.stderr.decode()}", "error")
        return redirect(url_for("home"))

    # --- Step 2: Transcribe (Your code) ---
    print("Step 2: Transcribing...")
    if whisper_model is None:
        flash("Whisper model not loaded.", "error")
        return redirect(url_for("home"))
        
    result = whisper_model.transcribe(audio_path, language="hi", task="translate")
    text = result["text"]
    chunks = [{"number": filename, "start": s["start"], "end": s["end"], "text": s["text"]}
              for s in result["segments"]]
    
    # Save the JSON (Good for debugging)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "text": text}, f, ensure_ascii=False, indent=2)

    # --- Step 3: Create Embeddings (Your original logic) ---
    print("Step 3: Creating embeddings...")
    try:
        text_list = [c['text'] for c in chunks]
        embeddings = create_embeddings_api(text_list)
        
        if embeddings is None:
             # Error was already flashed by the helper function
            return redirect(url_for("home"))

        my_dicts = []
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            chunk['embedding'] = embeddings[i]
            my_dicts.append(chunk)
            
        data = pd.DataFrame.from_records(my_dicts)
        all_embeddings_matrix = np.vstack(data['embedding'])
        
        # --- SUCCESS: Store in memory ---
        current_df = data
        current_embeddings = all_embeddings_matrix
        current_video_name = filename
        
        print("Pipeline complete. Data is ready in memory.")
        flash(f"Successfully processed '{filename}'. Ready to ask.", "success")
        
    except Exception as e:
        flash(f"Error during embedding: {e}", "error")

    return redirect(url_for("home"))


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Handles a question using YOUR original RAG pipeline.
    (This is your 4_query_answering.py logic)
    """
    # Check if data is loaded
    if current_df is None or current_embeddings is None:
        flash("Please upload and process a video first.", "error")
        return redirect(url_for("home"))

    data = request.form
    question = data.get("question")
    if not question:
        flash("Please provide a question", "error")
        return redirect(url_for("home"))

    print(f"Handling query: {question}")
    
    # --- Step 4: Query Logic ---
    # 1. Embed the question
    question_embedding = create_embeddings_api([question])
    if question_embedding is None:
        return redirect(url_for("home"))
    
    # 2. Find relevant chunks
    similarity = cosine_similarity(current_embeddings, question_embedding).flatten()
    max_idx = similarity.argsort()[::-1][0:TOP_K]
    new_df = current_df.loc[max_idx]

    # 3. Build the prompt (Your original prompt)
    context_json = new_df[["number", "chunk_id", "start", "end", "text"]].to_json(orient="records")
    
    prompt = f'''I am teaching about AI, you can predict which chunk is in which video

    {context_json}

    "{question}"
    User asked these questions related to video chunks, you have to answer where and how much content is taught in which video and at what timestamp and guide the user to go to the particular video and say in which video this content exists. If the user asks any unrelated questions then you can guide them to ask questions related to the course only.
    '''
    
    # 4. Get generative answer
    answer = inference_api(prompt)
    if answer is None:
        return redirect(url_for("home"))
        
    # 5. Format sources
    sources = []
    for _, item in new_df.iterrows():
        start_time = pd.to_datetime(item["start"], unit='s').strftime('%M:%S')
        end_time = pd.to_datetime(item["end"], unit='s').strftime('%M:%S')
        sources.append({
            "number": item['number'],
            "timestamp": f"{start_time} - {end_time}",
            "text": item['text']
        })

    # 6. Render the page with the answer
    return render_template(
        "index.html",
        video_name=current_video_name,
        answer=answer,
        sources=sources,
        query=question
    )

if __name__ == "__main__":
    # Make sure FFMPEG is installed
    # Make sure your Ollama server is running at http://localhost:11434
    if whisper_model is None:
        print("Exiting: Whisper model failed to load.")
    else:
        app.run(host="0.0.0.0", port=7860, debug=True)