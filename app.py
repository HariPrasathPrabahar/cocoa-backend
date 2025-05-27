from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI()

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Google Sheet setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1Xm3ZfojYLGVTcTFefMue9f1GctoPpJ76Yl7U0Ow-9tk/edit?usp=sharing").sheet1

# Load data once
descriptions = sheet.col_values(2)[1:]  # Column 2: Descriptions
links = sheet.col_values(1)[1:]         # Column 1: PPT links
embeddings = model.encode(descriptions, convert_to_tensor=True)

@app.get("/search")
def search_ppts(q: str = Query(...)):
    query_embedding = model.encode([q], convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = np.argsort(-cos_scores.numpy())[:5]

    results = []
    for idx in top_results:
        results.append({
            "description": descriptions[idx],
            "link": links[idx]
        })

    return JSONResponse(content={"results": results})
