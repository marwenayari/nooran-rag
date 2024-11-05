from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import generate_story
from utils import extract_text_from_pdf, clean_and_normalize_arabic_text, index_document
import os
import glob
from dotenv import load_dotenv
import json  # Import JSON for parsing

# Load environment variables
load_dotenv()

app = FastAPI()

# Request model for the API
class StoryRequest(BaseModel):
    words: list[str]
    sentences: list[str]

# Response model
class StoryResponse(BaseModel):
    title: str
    title_en: str
    brief: str
    brief_en: str
    content: list[str]
    content_en: list[str]
    min_age: int
    max_age: int

@app.on_event("startup")
async def startup_event():
    """Ingest PDFs and index content on application startup"""
    pdf_files = glob.glob("pdf_files/*.pdf")  # Adjust path to where your PDFs are stored
    for i, pdf_file in enumerate(pdf_files):
        print(f"Processing PDF: {pdf_file}")
        pdf_text = extract_text_from_pdf(pdf_file)
        clean_text = clean_and_normalize_arabic_text(pdf_text)
        index_document(clean_text, f"pdf-{i}")
    print("All documents indexed successfully.")

@app.post("/api/story", response_model=StoryResponse)
async def generate_story_api(request: StoryRequest):
    try:
        # Call the model function and parse the JSON response
        story_data = generate_story(request.words, request.sentences)

        # If the response is a string, try to parse it as JSON
        if isinstance(story_data, str):
            story_data = json.loads(story_data)

        return StoryResponse(**story_data)  # Ensure it's a valid StoryResponse
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
