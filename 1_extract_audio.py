import os
import subprocess

files=os.listdir("videos")
print(files)

for file in files:
    
    vid_num=file.split("_ ")[0]
    file_name=file.split("_")[1]
    
    
    input_video=f"videos/{file}"
    output_audio=f"audios/{vid_num} {file_name}.mp3"

    subprocess.run(["ffmpeg","-i",input_video,output_audio])