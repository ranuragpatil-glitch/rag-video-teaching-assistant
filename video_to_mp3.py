
import os
import subprocess


files = os.listdir("video")
for file in files:
    if file.startswith(".") or not os.path.isfile(os.path.join("video", file)):
        continue

    input_path = os.path.join("video", file)
    output_path = os.path.join("audios", f"{os.path.splitext(file)[0]}.mp3")
    subprocess.run(["ffmpeg", "-i", input_path, output_path], check=True)
