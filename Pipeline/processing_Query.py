import chromadb
from readChunks import createEmbeddings
import json 
import requests


#connecting to local chromadb server
client = chromadb.PersistentClient(path="Data/chromaDB") #path to store the database


#getting the collection from the database
collection = client.get_collection(name="video_transcripts")

def LLMresponse(prompt) :
      r = requests.post("http://localhost:11434/api/generate", json=
                        {
                              "model":"mistral:7b",
                              "prompt":prompt,
                              "stream":False
                        })
      response = r.json()["response"]
      return response

      


incomingQuery = input("Ask a Question : ")
questionEmbedding = createEmbeddings([incomingQuery])


#Finding most similar embeddings
results = collection.query(
      query_embeddings=questionEmbedding,
      n_results=10
)

context = []

#iterating through the results to get the context

#structuring list for easy reading by the llm
for metadata, document in zip( results['metadatas'][0], results['documents'][0]):
      context.append({
            "Video number" : metadata["video_number"],
            "Title" : metadata["Title"],
            "start_time" : metadata["start_time"],
            "end_time" : metadata["end_time"],
            "text" : document
      })
json_context = json.dumps(context, indent=4) #converting to json for easy reading by the llm


prompt = f'''
You are a friendly, knowledgeable Python tutor helping a student navigate their course videos. Talk directly to the student in a warm, encouraging, and natural conversational tone (use "you" and "your", never refer to them as "the student").

=== VIDEO TRANSCRIPT CONTEXT ===
{json_context}

=== USER QUESTION ===
{incomingQuery}

=== GUIDELINES ===
1. SPEAK DIRECTLY & NATURALLY:
   - Start by directly answering their question in 1–2 simple, friendly sentences.
   - Naturally guide them to the exact spot to watch it: mention the Video Title, Video Number, and the timestamp range in MM:SS format (e.g., "Check out Video 1 ('Lists in Python') from 00:45 to 01:00...").
   - Avoid robotic headers like "Primary Recommendation" or "Related Videos". Instead, weave your suggestions into natural paragraphs.

2. HELPFUL NEXT STEPS:
   - If other videos in the context build on this topic (like methods or related data types), naturally recommend them as helpful next steps (e.g., "Once you're comfortable with that, Video 2 dives into list methods...").

3. STRICT COURSE BOUNDARIES:
   - Base your help ONLY on the provided video transcripts.
   - If they ask something outside this course, kindly let them know: "That topic isn't covered in our Python course videos, but feel free to ask anything about what's in the course!"
   - Do NOT ask follow-up questions at the end. Keep the response concise, warm, and helpful.
'''

with open("prompt.txt", "w") as f:
      f.write(prompt)

response = LLMresponse(prompt)
print(response)
with open("response.txt", "w") as f :
     f.write(response)

