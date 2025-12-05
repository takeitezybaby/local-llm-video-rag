import os
import subprocess

files = os.listdir('C:\\Users\\yash3\\Desktop\\videoData')
for file in files :
      print(file)
      tutorialNo = file.split('_')[0]
      tutorialName = file.split("_")[1].capitalize().split(".")[0]
      subprocess.run(["ffmpeg","-i",f"C:\\Users\\yash3\\Desktop\\videoData/{file}",f"mp3files/{tutorialNo}-{tutorialName}.mp3"])
      print(f"\n\n Successfully converted {file} to mp3\n\n")