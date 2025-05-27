from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import gspread
import numpy as np
import json
import os

from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Authenticate Google Sheets
def load_gsheet():
    service_account_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)
    gc = gspread.authorize(creds)

    # 🛠️ Put your actual Google Sheet URL here in .env or Render secret as a key like: SHEET_URL
    sheet_url = os.getenv("SHEET_URL")
    sheet = gc.open_by_url(sheet_url).sheet1

    descriptions = sheet.col_values(2)[1:]  # Column B
    links = sheet.col_values(1)[1:]         # Column A
    return descriptions, links

descriptions, links = load_gsheet()

# Generate embeddings once at startup
def get_embedding(text: str) -> np.ndarray:
    return model.encode(text)

desc_embeddings = [get_embedding(desc) for desc in descriptions]

@app.get("/search")
def search(q: str = Query(...)):
    query_emb = get_embedding(q)
    similarities = [np.dot(query_emb, emb) for emb in desc_embeddings]
    top_indices = np.argsort(similarities)[-5:][::-1]

    results = [{
        "description": descriptions[i],
        "link": links[i]
    } for i in top_indices]

    return JSONResponse(content={"results": results})
