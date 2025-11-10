import os

print("STEP 1: Extracting audio...")
os.system("python 1_extract_audio.py")

print("STEP 2: Transcribing with Whisper...")
os.system("python 2_transcribe_whisper.py")

print("STEP 3: Creating embeddings...")
os.system("python 3_create_embeddings.py")

print("STEP 4: Starting question answering...")
os.system("python 4_query_answering.py")