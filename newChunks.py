#CLubbing chunks to improve accuracy

import json
import os
import math
n = 5

files =  os.listdir("jsonChunks")
for file in files :
      full_path = os.path.join("jsonChunks", file)
      with open(full_path,"r", encoding="utf-8") as f:
            data =  json.load(f)
            newChunks = []
            numChuncks = len(data)
            numGroups = math.ceil(numChuncks/n)

            for i in range(numGroups) :
                  start_idx = i*n
                  end_idx = min((i+1)*n, numChuncks)

                  Chunk_group = data[start_idx:end_idx]

                  newChunks.append({
                        "Video number" : data[0]["Video number"],
                        "Title": data[0]["title"],
                        "start" : Chunk_group[0]["start"],
                        "end" : Chunk_group[-1]["end"],
                        "text" : "".join(c["text"] for c in Chunk_group)
                  })
                  os.makedirs("newjsonChunks",exist_ok=True)
                  with open(os.path.join("newjsonChunks",file), "w", encoding="utf-8") as f:
                        json.dump(newChunks, f, indent=4)


