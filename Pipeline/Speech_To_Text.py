import whisper
import json
model = whisper.load_model("large-v2") 

result = model.transcribe(audio="C:\\Users\\yash3\\Desktop\\RAG\\mp3files\\sample.mp3",
                          language = "hi",
                          task = "translate",
                          word_timestamps=False)
chunks = []
for segment in result["segments"] :
      chunks.append({"start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})

with open("output.json", "w") as f :
      json.dump(chunks,f)
