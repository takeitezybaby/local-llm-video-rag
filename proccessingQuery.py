import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from readChunks import createEmbeddings
import numpy as np
import json 
import requests

def LLMresponse(prompt) :
      r = requests.post("http://localhost:11434/api/generate", json=
                        {
                              "model":"mistral:7b",
                              "prompt":prompt,
                              "stream":False
                        })
      response = r.json()["response"]
      return response

      

df = joblib.load("embeddingData.joblib")

incomingQuery = input("Ask a Question : ")
questionEmbedding = createEmbeddings([incomingQuery])


#Finding most similar embeddings
Similarities = cosine_similarity(np.vstack(df["Embedding"]),questionEmbedding).flatten()
top_picks = 10
max_ind =  Similarities.argsort()[::-1][0:top_picks]
#making new dataframe out of the top results
temp_df = df.iloc[max_ind]


prompt = f'''
So I'm making a RAG based interface for my uploaded videos. the videos are about basics of python. Below are the subvideo chunks containing title, video number, start time in seconds, end time in seconds and the text at that time
{temp_df[["title", "Video number", "start", "end", "text"]].to_json(orient = "records")}
------------------------------------------------------------
{incomingQuery}
Here's the incoming query of the user. Guide the user to appropriate timeline and video, where he can find the answer to his question(mention all specifics like video title, number, timestamp, what's there in that  video etc.).Also, tell the other videos that the user can refer to related to his topic(only refer the user to the additional video and only state the content in it). If the user asks unrelated question, tell him you can only answer related to the course sepecified above

Also greet the user at the end (don't mintion best regards!)
----Don't ask a follow question, only response----
'''

with open("prompt.txt", "w") as f:
      f.write(prompt)

response = LLMresponse(prompt)
print(response)
with open("response.txt", "w") as f :
     f.write(response)
