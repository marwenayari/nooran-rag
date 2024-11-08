# Nooran X Allam | Arabic Story Generation with IBM Watsonx and ChromaDB

This Documentation file provides a detailed explanation of the Jupyter Notebook titled **"Arabic Story Generation with IBM Watsonx and ChromaDB"**. The notebook demonstrates how to generate Arabic stories using IBM Watsonx language model and ChromaDB for vector storage and retrieval.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Setup and Imports](#setup-and-imports)
3. [Authentication with IBM Cloud](#authentication-with-ibm-cloud)
4. [Setting up the IBM Watsonx Model](#setting-up-the-ibm-watsonx-model)
5. [Setting up ChromaDB](#setting-up-chromadb)
6. [Helper Functions](#helper-functions)
   - [Text Extraction from PDFs](#text-extraction-from-pdfs)
   - [Text Cleaning and Normalization](#text-cleaning-and-normalization)
   - [Text Chunking](#text-chunking)
   - [Document Indexing](#document-indexing)
   - [Similarity Search](#similarity-search)
   - [Prompt Construction](#prompt-construction)
   - [Model Response Retrieval](#model-response-retrieval)
   - [Response Parsing](#response-parsing)
7. [Indexing PDF Documents](#indexing-pdf-documents)
8. [Constructing and Executing a Query](#constructing-and-executing-a-query)
9. [Parsing the Model Response](#parsing-the-model-response)
10. [Displaying the Results](#displaying-the-results)
11. [Verifying Collection Name (Optional)](#verifying-collection-name-optional)
12. [Environment Setup](#environment-setup)
13. [Dependencies](#dependencies)

---

## Introduction

This notebook showcases a workflow for generating Arabic stories tailored for children who are new to learning the Arabic language. It leverages:

- **IBM Watsonx AI Foundation Model**: A powerful language model for text generation.
- **ChromaDB**: An open-source embedding database for vector similarity search.

The process involves:

- Extracting text from Arabic PDF documents.
- Cleaning and normalizing the extracted text.
- Indexing the text into ChromaDB.
- Performing similarity search based on given keywords.
- Constructing a prompt for the language model using the retrieved contexts.
- Generating a story using the IBM Watsonx model.
- Parsing and displaying the generated story.

---

## Setup and Imports

### Description

We begin by importing necessary libraries and loading environment variables from the `.env` file. The environment variables are used for authentication and configuration purposes.

### Code

```python
import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)
import requests
```

- **os**: Provides functions for interacting with the operating system.
- **dotenv**: Loads environment variables from a `.env` file into `os.environ`.
- **requests**: Used for making HTTP requests.

---

## Authentication with IBM Cloud

### Description

To interact with IBM Watsonx services, we need to authenticate with IBM Cloud using an API key. This section handles the authentication process and retrieves an access token.

### Code

```python
api_key = os.getenv('IBM_API_KEY')

url = 'https://iam.cloud.ibm.com/identity/token'
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
data = {
    'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
    'apikey': api_key
}

response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    print("Token retrieved")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

res_format = response.json()
ACCESS_TOKEN = res_format['access_token']
```

- **api_key**: Retrieved from environment variables; used for authentication.
- **Access Token Retrieval**:
  - **URL**: IBM Cloud IAM token endpoint.
  - **Headers**: Specifies the content type.
  - **Data**: Contains the grant type and API key.
  - **Response Handling**: Checks if the token was successfully retrieved.

---

## Setting up the IBM Watsonx Model

### Description

With the access token, we configure the IBM Watsonx AI Foundation Model for text generation. We set up the credentials and define parameters for the model.

### Code

```python
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials

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
```

- **Imports**: Import necessary modules from `ibm_watsonx_ai`.
- **Credentials**: Contains the service URL and access token.
- **Parameters**:
  - **decoding_method**: Specifies the decoding strategy for text generation.
  - **max_new_tokens**: Maximum number of tokens to generate.
  - **repetition_penalty**: Penalty for repeated tokens to ensure diversity.
- **Model Initialization**:
  - **model_id**: Retrieved from environment variables.
  - **project_id**: Retrieved from environment variables.

---

## Setting up ChromaDB

### Description

ChromaDB is used for storing and retrieving document embeddings. This section initializes the ChromaDB client, sets up the embedding function, and creates or retrieves a collection for storing the embeddings.

### Code

```python
import chromadb
from chromadb.utils import embedding_functions

# Chroma setup
chroma_client = chromadb.Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-base")

collection_name = "nooran_x_allam"
try:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Using existing collection: {collection_name}")
except ValueError:
    collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)
    print(f"Created new collection: {collection_name}")
```

- **chromadb**: The main ChromaDB library.
- **embedding_functions**: Contains functions for embedding text.
- **Embedding Function**:
  - Uses the multilingual model `intfloat/multilingual-e5-base` from Sentence Transformers.
- **Collection Handling**:
  - **Try** to retrieve an existing collection.
  - **Except** create a new collection if it doesn't exist.

---

## Helper Functions

### Overview

This section defines several helper functions that perform key tasks in the workflow:

- **Text Extraction**: From PDFs.
- **Text Cleaning**: Specifically for Arabic text.
- **Text Chunking**: Splitting text into manageable pieces.
- **Document Indexing**: Adding documents to ChromaDB.
- **Similarity Search**: Finding similar documents based on a query.
- **Prompt Construction**: For the language model.
- **Model Response Retrieval**: Generating text using the model.
- **Response Parsing**: Extracting structured data from the model's output.

---

### Text Extraction from PDFs

#### Description

Extracts text from a PDF file using `pypdfium2`.

#### Code

```python
import pypdfium2 as pdfium

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
```

- **pdfium**: A PDF rendering library.
- **Functionality**:
  - Opens the PDF document.
  - Iterates through each page.
  - Extracts text content.
  - Ensures the PDF file is closed after processing.

---

### Text Cleaning and Normalization

#### Description

Cleans and normalizes the extracted Arabic text by:

- Correcting misencoded characters.
- Removing unwanted characters.
- Normalizing whitespace.

#### Code

```python
import re

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
    text = text.replace("٬", ",")  # Replace specific combination of characters

    # Keep only Arabic characters, numbers, and basic punctuation
    cleaned_text = re.sub(f'[^{arabic_range}\w\s\.,:!؟]', '', text)

    # Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text
```

- **Regular Expressions**: Used for pattern matching and replacements.
- **Replacements Dictionary**: Corrects common misencoded characters.
- **Character Filtering**: Keeps only relevant characters.
- **Whitespace Normalization**: Ensures consistent spacing.

---

### Text Chunking

#### Description

Splits large text into smaller chunks of a specified size. This is useful for processing and indexing large documents.

#### Code

```python
def chunk_text(text, chunk_size=700):
    """Split text into chunks of specified size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

- **Parameters**:
  - **text**: The text to be chunked.
  - **chunk_size**: The size of each chunk (default is 700 characters).
- **Functionality**: Uses list comprehension to create a list of text chunks.

---

### Document Indexing

#### Description

Indexes a document into ChromaDB by splitting it into chunks and adding each chunk as a separate document.

#### Code

```python
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
```

- **Parameters**:
  - **text**: The text to index.
  - **id**: An identifier for the document.
- **Functionality**:
  - Splits the text into chunks.
  - Iterates over each chunk.
  - Adds the chunk to the ChromaDB collection with a unique ID.

---

### Similarity Search

#### Description

Searches for documents in ChromaDB that are similar to the given query.

#### Code

```python
def search_similar_documents(query, top_k=5):
    """Search for similar documents in ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    print("Raw query results:", results)  # Debug print
    return results['documents'][0] if results['documents'] else []
```

- **Parameters**:
  - **query**: The query text.
  - **top_k**: Number of top results to return.
- **Functionality**:
  - Performs a query on the ChromaDB collection.
  - Retrieves the most similar documents.
  - Returns the documents if found.

---

### Prompt Construction

#### Description

Constructs a detailed prompt for the language model, incorporating the retrieved contexts and specific guidelines for story generation.

#### Code

```python
def construct_prompt(keywords, retrieved_contexts, age=6):
    """Construct the prompt for the language model using the keywords and retrieved contexts."""
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
```

- **Parameters**:
  - **keywords**: The keywords to be included in the story.
  - **retrieved_contexts**: Contexts retrieved from similar documents.
  - **age**: Target age for the story audience.
- **Functionality**:
  - **System Prompt**: Contains instructions, guidelines, and examples for the model.
  - **Query**: Specific input for the model, including keywords.
  - **Prompt Structure**: Combines the system prompt and query in a specific format expected by the model.

---

### Model Response Retrieval

#### Description

Generates text using the IBM Watsonx model based on the constructed prompt.

#### Code

```python
def get_watsonx_response(prompt):
    """Get the response from the IBM Watsonx model."""
    response = model.generate_text(prompt=prompt)
    return response
```

- **Parameters**:
  - **prompt**: The prompt constructed in the previous step.
- **Functionality**:
  - Calls the `generate_text` method of the model.
  - Returns the generated text.

---

### Response Parsing

#### Description

Parses the JSON response from the model to extract structured information such as title, brief, content, and age range.

#### Code

```python
def parse_llm_response(response):
    """Parse the JSON response from the language model."""
    # Use regex to extract the JSON part within the first set of braces
    match = re.search(r"\{\s*\"title\".*?\}\s*", response, re.DOTALL)

    if match:
        cleaned_response = match.group(0)  # Extract the JSON portion

        # Remove any trailing commas before a closing brace
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
```

- **Imports**: `re` for regular expressions, `json` for parsing JSON.
- **Functionality**:
  - Uses regex to locate the JSON structure in the response.
  - Cleans the JSON string to fix common formatting issues.
  - Parses the JSON string into a Python dictionary.
  - Handles errors if the JSON is malformed.

---

## Indexing PDF Documents

### Description

Processes all PDF files in the `pdf_files` directory:

1. Extracts text from each PDF.
2. Cleans and normalizes the text.
3. Indexes the text into ChromaDB.

### Code

```python
import glob

pdf_files = glob.glob("pdf_files/*.pdf")
for i, pdf_file in enumerate(pdf_files):
    print(f"Processing PDF: {pdf_file}")
    pdf_text = extract_text_from_pdf(pdf_file)
    clean_text = clean_and_normalize_arabic_text(pdf_text)
    index_document(clean_text, f"pdf-{i}")
```

- **glob**: Used for file pattern matching.
- **Process**:
  - **Enumerate** over all PDF files.
  - **Extract** text from each PDF.
  - **Clean** the extracted text.
  - **Index** the cleaned text into ChromaDB.

---

## Constructing and Executing a Query

### Description

Defines a query using specific Arabic keywords, performs a similarity search, constructs a prompt, and generates a response from the model.

### Code

```python
query = " الحب، السلام، الاهل، الصداقة، السعادة، الفرح"
similar_docs = search_similar_documents(query)
prompt = construct_prompt(query, similar_docs)
response = get_watsonx_response(prompt)
```

- **query**: Contains the keywords for the story.
- **similar_docs**: Retrieved documents similar to the query.
- **prompt**: Constructed using the query and similar documents.
- **response**: Generated text from the model.

---

## Parsing the Model Response

### Description

Parses the model's response to extract structured data.

### Code

```python
parsed_response = parse_llm_response(response)
```

- **parsed_response**: A dictionary containing the structured story information.

---

## Displaying the Results

### Description

Prints the constructed prompt, retrieved documents, the raw response from the model, and the parsed structured response.

### Code

```python
print("\nPrompt:\n")
print(prompt)

print("\nRetrieved Documents:\n")
for i, doc in enumerate(similar_docs, 1):
    print(f"{i}. {doc}...")

print("\nWatsonx Response:\n")
print(response)
print("\nParsed Response:\n")
print(parsed_response)
```

- **Outputs**:
  - **Prompt**: The final prompt sent to the model.
  - **Retrieved Documents**: The similar documents found in ChromaDB.
  - **Watsonx Response**: The raw text generated by the model.
  - **Parsed Response**: The structured data extracted from the response.

---

## Verifying Collection Name (Optional)

### Description

Retrieves and prints the collection name from environment variables to verify the collection used in ChromaDB.

### Code

```python
collection_name = os.getenv('nooran_x_allam')
print(collection_name)
```

- **Purpose**: To confirm the collection being used.
- **Note**: Ensure the environment variable `nooran_x_allam` is set if using this section.

---

## Environment Setup

### .env File Configuration

Ensure you have a `.env` file in your project directory with the following variables:

```env
IBM_API_KEY=your_ibm_api_key
MODEL_ID=your_model_id
PROJECT_ID=your_project_id
```

Replace `your_ibm_api_key`, `your_model_id`, and `your_project_id` with your actual credentials.

### Directory Structure

Ensure your project has the following structure:

```
project/
├── .env
├── arabic_story_generation.ipynb
├── pdf_files/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
```

- Place all your PDF files inside the `pdf_files/` directory.

---

## Dependencies

Install the required Python packages before running the notebook:

```bash
pip install requests ibm-watsonx-ai chromadb pypdfium2 sentence-transformers
```

- **requests**: For making HTTP requests.
- **ibm-watsonx-ai**: IBM Watsonx AI Foundation Model SDK.
- **chromadb**: For vector storage and retrieval.
- **pypdfium2**: For PDF text extraction.
- **sentence-transformers**: For embedding functions.

---

## Execution Steps

1. **Set Up Environment**:

   - Configure your `.env` file with the necessary credentials.
   - Ensure all PDF files are in the `pdf_files/` directory.

2. **Run the Notebook Cells Sequentially**:

   - Start from the top and execute each cell in order.
   - Monitor the output for any errors.

3. **Review the Outputs**:
   - After running all cells, review the generated story and ensure it meets your expectations.

---

## Conclusion

This notebook provides a comprehensive workflow for generating Arabic stories tailored to young learners. By integrating IBM Watsonx AI and ChromaDB, it demonstrates the power of combining advanced language models with efficient vector search techniques.
