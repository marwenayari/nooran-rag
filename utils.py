import os
import glob
import re
import chromadb
from chromadb.utils import embedding_functions
import pypdfium2 as pdfium
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pypdfium2."""
    text = ""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for page in pdf:
            textpage = page.get_textpage()
            text += textpage.get_text_bounded()
    finally:
        pdf.close()
    return text

def clean_and_normalize_arabic_text(text):
    """Clean and normalize the extracted text, focusing on Arabic characters and specific issues."""
    arabic_range = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDCF\uFDF0-\uFEFF]'
    replacements = {
        "û": "ل",
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    text = text.replace("؟", "?")
    text = text.replace("٬", ",")
    cleaned_text = re.sub(f'[^{arabic_range}\w\s\.,:!؟]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def chunk_text(text, chunk_size=300):
    """Split text into chunks of specified size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def construct_prompt(query, retrieved_contexts):
    context = "\n".join(retrieved_contexts)
    prompt = (
        f"اجب من السياق التالي : {query}\n\n"
        f"السياق:\n{context}\n\n"
    )
    return prompt

def setup_chromadb(collection_name="default_collection", persist_directory="./chroma_db"):
    print(f"Setting up ChromaDB... Collection: {collection_name}, Persist Directory: {persist_directory}")
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")
    try:
        # Include the embedding_function when retrieving the collection
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        print(f"Using existing collection: {collection_name}")
    except Exception as e:
        # Create the collection if it doesn't exist
        collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)
        print(f"Created new collection: {collection_name}")
    return collection

def index_document(collection, text, id):
    """Index a document in ChromaDB with specified chunk size."""
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        chunk_id = f"{id}-chunk-{i}"
        collection.add(
            documents=[chunk],
            ids=[chunk_id]
        )
    print(f"Indexed document {id} in {len(chunks)} chunks.")

def search_similar_documents(collection, query, top_k=2):
    """Search for similar documents in ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    print("Raw query results:", results)  # Debug print
    return results['documents'][0] if results['documents'] else []

def index_pdfs(collection, pdf_directory="pdf_files"):
    pdf_files = glob.glob(f"{pdf_directory}/*.pdf")
    for i, pdf_file in enumerate(pdf_files):
        print(f"Processing PDF: {pdf_file}")
        pdf_text = extract_text_from_pdf(pdf_file)
        clean_text = clean_and_normalize_arabic_text(pdf_text)
        index_document(collection, clean_text, f"pdf-{i}")
    print("Documents indexed. Ready for queries!")
