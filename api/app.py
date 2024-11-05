import sys
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import setup_watsonx_model, get_watsonx_response
from utils import setup_chromadb, search_similar_documents, construct_prompt

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'), override=True)

app = FastAPI(
    title="Nooran X Allam API",
    description="An API for generating stories using Watsonx AI and ChromaDB",
    version="1.0.0"
)

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
