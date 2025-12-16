import json
import os
import requests
import pandas as pd

#Using ollama model to read chunks
def createEmbeddings(text) :
      r=requests.post("http://localhost:11434/api/embed",json= #http://localhost:11434/api/embed
                      {
                            "model":"bge-m3",
                            "input":text
                      })
      embedding = r.json()["embeddings"] #extracting only embeddings
      return embedding

files = os.listdir("C:\\Users\\yash3\\Desktop\\RAG\\jsonChunks") #listing all json files
data = []
chunk_id = 0 

for json_file in files :
   #   print(json_file)
      with open(f"C:\\Users\\yash3\\Desktop\\RAG\\jsonChunks\\{json_file}") as f:
            content = json.load(f) #loading each json file
      print(f"Creating embeddings for {json_file}")
      embeddings = createEmbeddings([c['text'] for c in content]) #makes embedding for a file (all text of it)

      for i,chunk in enumerate(content) :
            chunk["Chunk id"] = chunk_id
            chunk_id += 1
            chunk["Embedding"] = embeddings[i]
            data.append(chunk)

df = pd.DataFrame.from_records(data)
print(df)