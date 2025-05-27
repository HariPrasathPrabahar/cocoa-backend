from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import openai
import numpy as np
import os

app = FastAPI()

# Load OpenAI API key from env var
openai.api_key = os.getenv("OPENAI_API_KEY")

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1Xm3ZfojYLGVTcTFefMue9f1GctoPpJ76Yl7U0Ow-9tk/edit?usp=sharing").sheet1

# Load data from sheet
descriptions = sheet.col_values(2)[1:]  # Column B: Descriptions
links = sheet.col_values(1)[1:]         # Column A: Links

# Embed all descriptions ONCE using OpenAI
def get_embedding(text):
    res = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(res['data'][0]['embedding'])

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
