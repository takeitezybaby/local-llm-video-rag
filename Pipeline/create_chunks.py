import whisper
import json
import os
model = whisper.load_model("large-v2") 
files = os.listdir("C:\\Users\\yash3\\Desktop\\RAG\\mp3files")
for file in files : 
      result = model.transcribe(audio=f"C:\\Users\\yash3\\Desktop\\RAG\\mp3files\\{file}",
                          language = "hi",
                          task = "translate",
                          word_timestamps=False)
      vid_number = file.split("-")[0]
      vid_title = file.split("-")[1].split(".")[0]
      chunks=[]
      for segment in result["segments"] :
            chunks.append({"Video number" : vid_number,"title":vid_title,"start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})
      with open(f"C:\\Users\\yash3\\Desktop\\RAG\\jsonChunks\\{vid_number}-{vid_title}(Chunks).json", "w") as f:
            json.dump(chunks,f)
