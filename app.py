from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from openai import OpenAI
import gspread
import numpy as np
import json
import os
from oauth2client.service_account import ServiceAccountCredentials

app = FastAPI()

# Set up OpenAI client with API key from environment variable
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Authenticate Google Sheets and load data
def load_gsheet():
    # Load service account credentials JSON from env var
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_JSON environment variable")
    service_account_info = json.loads(service_account_json)
    
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)
    gc = gspread.authorize(creds)
    
    # Load sheet URL from environment variable to avoid hardcoding
    sheet_url = os.getenv("GOOGLE_SHEET_URL")
    if not sheet_url:
        raise ValueError("Missing GOOGLE_SHEET_URL environment variable")
    
    sheet = gc.open_by_url(sheet_url).sheet1
    
    # Get descriptions from column B (skip header)
    descriptions = sheet.col_values(2)[1:]
    # Get links from column A (skip header)
    links = sheet.col_values(1)[1:]
    return descriptions, links

descriptions, links = load_gsheet()

# Generate embeddings once at startup
def get_embedding(text: str) -> np.ndarray:
    response = openai_client.embeddings.create(
        input=[text],  # input must be a list
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
