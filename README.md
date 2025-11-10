# 🚀 AI Video Chat Pipeline 🤖

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-black?logo=flask&logoColor=white)
![Whisper](https://img.shields.io/badge/OpenAI_Whisper-grey?logo=openai&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-black?logo=ollama&logoColor=white)

This project provides a full-stack web application to "chat" with your videos. It runs a complete Retrieval-Augmented Generation (RAG) pipeline locally, allowing you to upload a video, have it transcribed, and then ask questions about its content.

## ✨ The Pipeline

The application works by running your original Python scripts as a single, efficient process:

**`Upload`** ➔ **`1. FFMPEG`** (Extract Audio) ➔ **`2. Whisper`** (Transcribe) ➔ **`3. bge-m3`** (Embed Chunks) ➔ **`Ready`**

When you ask a question:

**`Query`** ➔ **`bge-m3`** (Embed Query) ➔ **`Cosine Similarity`** (Find Chunks) ➔ **`llama3.2`** (Generate Answer) ➔ **`Display`**

---

## 🔧 Tech Stack

```diff
# --- Backend ---
! Flask: For the web server and UI.
! Werkzeug: For secure file handling.

# --- Audio & Transcription ---
! ffmpeg: The core dependency for audio extraction.
! openai-whisper: For highly accurate transcription and translation.

# --- RAG & AI ---
! Ollama: For serving local language models.
! bge-m3: State-of-the-art model for text embeddings.
! llama3.2: Generative model for intelligent Q&A.

# --- Data & Vector Math ---
! Pandas & NumPy: For managing and searching the embedding data.
! Scikit-learn: For calculating cosine similarity.
```

---

## ⚡️ Setup & Installation

This project relies on **external, heavy dependencies**. Do not skip these steps.

### 1. Prerequisites (Crucial!)

```diff
- 1. Install FFMPEG
# This project CANNOT work without FFMPEG.
# Download it from ffmpeg.org and add it to your system's PATH.
# To verify, open a terminal and type:
ffmpeg -version

- 2. Install Ollama
# Download and install Ollama from [https://ollama.com/](https://ollama.com/)
# After installing, run the server (it usually runs in the background).
```

### 2. Project Installation

```diff
+ 1. Clone the repository
git clone https://your-repo-url/here.git
cd your-project-directory

+ 2. Create a Python Virtual Environment
python -m venv venv
source venv/bin/activate   # On Windows: .\venv\Scripts\activate

+ 3. Install all required Python packages
pip install flask pandas numpy scikit-learn joblib requests openai-whisper werkzeug
```

### 3. Download the AI Models

This is the final setup step. Your Ollama server must be running.

```diff
! 1. Pull the Embedding Model (bge-m3)
ollama pull bge-m3

! 2. Pull the LLM (llama3.2)
ollama pull llama3.2

# You can verify the models are installed by typing:
ollama list
```

---

## ▶️ How to Run

You're all set!

```diff
+ 1. Start the Flask App
# (Make sure your Ollama server is already running in the background)
python app.py

+ 2. Open Your Browser
# Navigate to the local address shown in your terminal, usually:
http://localhost:5000
```

Now you can upload a video, wait for it to process, and start asking questions!
