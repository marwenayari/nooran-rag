from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import setup_watsonx_model, get_watsonx_response
from utils import setup_chromadb, search_similar_documents, construct_prompt

import os

app = FastAPI()

# Setup the IBM Watsonx AI model
model = setup_watsonx_model()

# Setup ChromaDB collection
collection_name = os.getenv("CHROMADB_COLLECTION_NAME", "nooran_x_allam")
persist_directory = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_db")
collection = setup_chromadb(collection_name, persist_directory)

class QueryRequest(BaseModel):
    query: str

@app.post("/generate")
def generate_response(request: QueryRequest):
    query = request.query
    try:
        similar_docs = search_similar_documents(collection, query)
        print(f"Similar documents: {similar_docs}")
        prompt = construct_prompt(query, similar_docs)
        print(f"Constructed prompt: {prompt}")
        response = get_watsonx_response(model, prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
