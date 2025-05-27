from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

app = FastAPI()

# Load the model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')

# Google Sheet setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1Xm3ZfojYLGVTcTFefMue9f1GctoPpJ76Yl7U0Ow-9tk/edit?usp=sharing").sheet1

# Preload data and create vector index
descriptions = sheet.col_values(2)[1:]  # Assuming column 2 is descriptions (excluding header)
links = sheet.col_values(1)[1:]         # Assuming column 1 is links

embeddings = model.encode(descriptions, convert_to_tensor=False)
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

@app.get("/search")
def search_ppts(q: str = Query(..., description="Enter case description")):
    query_embedding = model.encode([q])[0]
    D, I = index.search(np.array([query_embedding]), k=5)

    results = []
    for i in I[0]:
        results.append({
            "description": descriptions[i],
            "link": links[i]
        })

    return JSONResponse(content={"results": results})
