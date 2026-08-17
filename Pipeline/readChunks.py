import json
import os
import requests
import chromadb



#connecting to local chromadb server
client = chromadb.PersistentClient(path="Data/chromaDB") #path to store the database

##Creating a collection to store the embeddings
collection = client.get_or_create_collection(
      name="video_transcripts",
      metadata= {"hnsw:space" : "cosine"}
)


#Using ollama model to read chunks
def createEmbeddings(text) :
      r=requests.post("http://localhost:11434/api/embed",json= #http://localhost:11434/api/embed
                      {
                            "model":"nomic-embed-text",
                            "input":text
                      })
      embedding = r.json()["embeddings"] #extracting only embeddings
      return embedding

if __name__ == "__main__" :
      files = os.listdir("Data/Chunks/newjsonChunks") #listing all json files
      data = []
      chunk_id = 0 

      for json_file in files :
      #   print(json_file)
            with open(os.path.join("Data/Chunks/newjsonChunks", json_file), "r", encoding="utf-8") as f:
                  content = json.load(f) #loading each json file
            print(f"Creating embeddings for {json_file}")
            embeddings = createEmbeddings([c['text'] for c in content]) #makes embedding for a file (all text of it)

            #preparing 4 data lists to store in the database
            ids = []
            metadatas = []
            documents = []

            for chunk in content :
                  ids.append(f"chunk_{chunk_id}") #creating unique id for each chunk
                  documents.append(chunk['text']) #storing the text of each chunk
                  metadatas.append({
                        "Title" : str(chunk.get('Title', '')),
                        "video_number" : int(chunk.get('Video number', 0)),
                        "start_time" : float(chunk.get('start', 0.0)),
                        "end_time" : float(chunk.get('end', 0.0)),
                  })
                  chunk_id += 1


            #upserting the data into the collection
            collection.upsert(
                  ids=ids,
                  metadatas=metadatas,
                  documents=documents,
                  embeddings=embeddings
            )

            print(f"Embeddings for {json_file} created and stored in the database.")