import os
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from dotenv import load_dotenv
load_dotenv('.env', override=True)
import requests

api_key = os.getenv('IBM_API_KEY')

url = 'https://iam.cloud.ibm.com/identity/token'
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
data = {
    'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
    'apikey': api_key
}

response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    # print(response.json())
    print("token retrieved")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

res_format = response.json()
ACCESS_TOKEN = res_format['access_token']
# Set up the credentials
credentials = Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    token=ACCESS_TOKEN
)

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 1000,
    "repetition_penalty": 1.0
}

model = Model(
    model_id=os.getenv("MODEL_ID"),
    params=parameters,
    credentials=credentials,
    project_id=os.getenv("PROJECT_ID")
)
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
import chromadb
from chromadb.utils import embedding_functions
import glob

# Chroma setup
chroma_client = chromadb.Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")

collection_name = "nooran_x_allam"
try:
    # collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Using existing collection: {collection_name}")
except ValueError:
    collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)
    print(f"Created new collection: {collection_name}")

import pypdfium2 as pdfium
import glob
import re

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pypdfium2."""
    text = ""
    pdf = pdfium.PdfDocument(pdf_path)  # Create PdfDocument object
    try:
        for page in pdf:
            textpage = page.get_textpage()
            text += textpage.get_text_bounded()
    finally:
        pdf.close()
    return text

def clean_and_normalize_arabic_text(text):
    """Clean and normalize the extracted text, focusing on Arabic characters and specific issues."""
    # Define the Arabic character range
    arabic_range = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDCF\uFDF0-\uFEFF]'

    # Common replacements for misencoded characters
    replacements = {
        "û": "ل",
    }

    # Apply replacements
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    # Handle specific cases
    text = text.replace("؟", "?")  # Replace combination with single character
    text = text.replace("٬", ",")  # Remove specific combination of characters

    # Keep only Arabic characters, numbers, and basic punctuation
    cleaned_text = re.sub(f'[^{arabic_range}\w\s\.,:!؟]', '', text)

    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def chunk_text(text, chunk_size=700):
    """Split text into chunks of specified size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def index_document(text, id):
    """Index a document in ChromaDB with specified chunk size."""
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        chunk_id = f"{id}-chunk-{i}"
        collection.add(
            documents=[chunk],
            ids=[chunk_id]
        )
    print(f"Indexed document {id} in {len(chunks)} chunks.")

def search_similar_documents(query, top_k=5):
    """Search for similar documents in ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    print("Raw query results:", results)  # Debug print
    return results['documents'][0] if results['documents'] else []
 
def construct_prompt(keywords, retrieved_contexts, age=6):
    # Join retrieved contexts
    context = "\n".join(retrieved_contexts)
    
    # Construct the system prompt with rules and guidelines
    system_prompt = (
        "قم بإنشاء قصة باستخدام مجموعة كلمات تُعطى لك، مع اتباع الأسلوب المُستخدم في الأمثلة التالية. اجعل القصة لطيفة وسهلة وموجهة لحديثي تعلم اللغة العربية. \n"
        "اريد عنواناً للقصة مع نبذة قصيرة عنها وترجمتها إلى الإنجليزية، كذلك قم بتقييم العمر المناسب للقصة بإعطاء حد أدنى وحد أقصى للعمر.\n"
        f"{context}\n\n"
        "**القواعد:**\n"
        "- لا تقدم أي كلام خارج عن القصة.\n"
        "- استخدم كلمات لطيفة موجهة للأطفال.\n"
        "**المدخل:**\n"
        "كلمات متفرقة تعلمها الطفل حديثا.\n"
        "عمر الطفل.\n"
        "**المخرج المتوقع:**\n"
        "{{\n"
        '  "title": "title in Arabic",\n'
        '  "title_en": "Title in English",\n'
        '  "brief": "Brief in Arabic",\n'
        '  "brief_en": "Brief in English",\n'
        '  "content": ["first sentence in Arabic", "second sentence in Arabic", ...],\n'
        '  "content_en": ["First sentence in English", "Second sentence in English", ...],\n'
        '  "min_age": min_age,\n'
        '  "max_age": max_age,\n'
        "}}\n\n"
        "مثال للمدخل:\n\n"
        "باستخدام الكلمات التالية، قم باكمال القصة التالية\n"
        "كان يامكان، كان هناك\n"
        "الكلمات: مزرعة، دجاج، أبقار، خرفان، حيوانات\n"
        f"العمر: {age}\n\n"
        "لا تكتب أي شيء واي ملاحظة بخلاف المخرج المتوقع، على شكل object.\n"
        "مثل هذا المخرج يجب أن يكون الناتج النهائي للقصة.\n\n"
        "المخرج:\n"
        "{{\n"
        '  "title": "الدجاجة الذهبية",\n'
        '  "title_en": "The Golden Chicken Story",\n'
        '  "brief": "قصة عن مزارع وزوجته يملكان دجاجة ذهبية تضع بيضات ذهبية",\n'
        '  "brief_en": "A story about a farmer and his wife who own a golden chicken that lays golden eggs",\n'
        '  "content": ["يُحكى أنّ مزارعاً وزوجته..."],\n'
        '  "content_en": ["It is said that a farmer and his wife..."],\n'
        f'  "min_age": 6,\n'
        f'  "max_age": 10,\n'
        "}}\n\n"
        "IMPORTANT: Keep the output result in the same format as the expected output below\n\n"
        "{{\n"
        '  "title": "title in Arabic",\n'
        '  "title_en": "Title in English",\n'
        '  "brief": "Brief in Arabic",\n'
        '  "brief_en": "Brief in English",\n'
        '  "content": ["first sentence in Arabic", "second sentence in Arabic", ...],\n'
        '  "content_en": ["First sentence in English", "Second sentence in English", ...],\n'
        '  "min_age": min_age,\n'
        '  "max_age": max_age,\n'
        "}}"
    )

    query = ("باستخدام الكلمات التالية، قم باكمال القصة التالية\n"
        "كان يامكان، كان هناك\n"
        "الكلمات:" f"{keywords}\n")

    # Construct the final prompt

    prompt = f"<<SYS>>{system_prompt}<<SYS>>[INST]{query}[/INST]"

    return prompt

def get_watsonx_response(prompt):
    response = model.generate_text(prompt=prompt)
    return response
pdf_files = glob.glob("pdf_files/*.pdf")
for i, pdf_file in enumerate(pdf_files):
    print(f"Processing PDF: {pdf_file}")
    pdf_text = extract_text_from_pdf(pdf_file)
    clean_text = clean_and_normalize_arabic_text(pdf_text)
    index_document(clean_text, f"pdf-{i}")

import re
import json

def parse_llm_response(response):
    # Step 1: Use regex to extract only the JSON part within the first set of braces
    match = re.search(r"\{\s*\"title\".*?\}\s*", response, re.DOTALL)
    
    if match:
        cleaned_response = match.group(0)  # Extract the JSON portion
        
        # Step 2: Remove any trailing commas before a closing brace
        cleaned_response = re.sub(r",\s*}", "}", cleaned_response)
        
        # Attempt to parse the extracted JSON content
        try:
            parsed_response = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            parsed_response = {}
    else:
        print("Error: JSON structure not found.")
        parsed_response = {}
    
    return parsed_response

print("Documents indexed. Ready for queries!")
query = " الحب، السلام، الاهل، الصداقة، السعادة، الفرح"
similar_docs = search_similar_documents(query)
prompt = construct_prompt(query, similar_docs)
response = get_watsonx_response(prompt)
parsed_response = parse_llm_response(response)

print("\nPrompt:\n")
print(prompt)

print("\nRetrieved Documents:\n")
for i, doc in enumerate(similar_docs, 1):
    print(f"{i}. {doc}...")

print("\nWatsonx Response:\n")
print(response)
print("\nParsed Response:\n")
print(parsed_response)
collection_name = os.getenv('nooran_x_allam')
print (collection_name)