from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from openai import OpenAI
import gspread
import numpy as np
import json
import os

from oauth2client.service_account import ServiceAccountCredentials

app = FastAPI()

# Set up OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Authenticate Google Sheets
def load_gsheet():
    service_account_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)
    gc = gspread.authorize(creds)
    sheet = gc.open_by_url(os.getenv("https://docs.google.com/spreadsheets/d/1Xm3ZfojYLGVTcTFefMue9f1GctoPpJ76Yl7U0Ow-9tk/edit?usp=sharing")).sheet1
    descriptions = sheet.col_values(2)[1:]  # Column B
    links = sheet.col_values(1)[1:]         # Column A
    return descriptions, links

descriptions, links = load_gsheet()

# Generate embeddings once at startup
def get_embedding(text: str) -> np.ndarray:
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

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
