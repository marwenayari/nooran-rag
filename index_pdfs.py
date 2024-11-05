import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)

import chromadb

from utils import setup_chromadb, index_pdfs

collection_name = os.getenv("CHROMADB_COLLECTION_NAME", "nooran_x_allam")
persist_directory = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_db")

try:
  chromadb.Client().delete_collection(collection_name)
except Exception as e:
  collection = setup_chromadb(collection_name, persist_directory)

index_pdfs(collection, pdf_directory="pdf_files")