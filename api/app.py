from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import setup_watsonx_model, get_watsonx_response
from utils import setup_chromadb, search_similar_documents, construct_prompt

import os
from dotenv import load_dotenv

load_dotenv('.env', override=True)

app = FastAPI()

# Setup the IBM Watsonx AI model
model = setup_watsonx_model()

# Setup ChromaDB collection with persistence
collection_name = os.getenv("CHROMADB_COLLECTION_NAME", "default_collection")
persist_directory = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_db")
collection = setup_chromadb(collection_name, persist_directory)

class QueryRequest(BaseModel):
    query: str

@app.post("/generate")
async def generate_response(request: QueryRequest):
    query = request.query
    try:
        similar_docs = search_similar_documents(collection, query)
        prompt = construct_prompt(query, similar_docs)
        response = get_watsonx_response(model, prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
