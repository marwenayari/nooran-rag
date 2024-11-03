import re
import chromadb
from chromadb.utils import embedding_functions
import pypdfium2 as pdfium

# Initialize ChromaDB client and embedding function
chroma_client = chromadb.Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")

collection_name = "allam_x_nooran"
try:
    collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
except ValueError:
    collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pypdfium2."""
    pdf = pdfium.PdfDocument(pdf_path)
    text = ""
    for page in pdf:
        textpage = page.get_textpage()
        text += textpage.get_text_range()
    return text

def clean_and_normalize_arabic_text(text):
    """Clean and normalize the extracted text, focusing on Arabic characters and specific issues."""
    arabic_range = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDCF\uFDF0-\uFEFF]'
    replacements = {"û": "ل"}
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    cleaned_text = re.sub(f'[^{arabic_range}\w\s\.,:!؟]', '', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

def chunk_text(text, chunk_size=1000):
    """Split text into chunks of a specified size."""
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

def search_similar_documents(query, top_k=2):
    """Search for similar documents in ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results['documents'][0] if results['documents'] else []

def construct_prompt(query, retrieved_contexts):
    """Construct a prompt for story generation using the retrieved contexts."""
    context = "\n".join(retrieved_contexts)
    prompt = (
        f"الكلمات المفتاحية: {query}\n\n"
        f"السياق:\n{context}\n\n"
        "باستخدام الكلمات المفتاحية والسياق المذكورين، اكتب قصة شيقة للأطفال باللغة العربية "
        "تبدأ بعبارة 'كان يا مكان كان هناك...' وتشمل الكلمات المفتاحية. احرص على أن تكون القصة متسلسلة وجذابة، "
        "وقم بتقديم عنوان مناسب يعبر عن القصة."
    )
    return prompt
