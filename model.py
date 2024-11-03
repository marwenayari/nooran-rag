from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials
from utils import clean_and_normalize_arabic_text, index_document, search_similar_documents, construct_prompt
import requests
import os

def get_access_token(api_key):
    url = 'https://iam.cloud.ibm.com/identity/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
        'apikey': api_key
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"Error retrieving access token: {response.status_code}")

# Initialize Watsonx credentials and model
def initialize_model():
    api_key = os.getenv('IBM_API_KEY')
    access_token = get_access_token(api_key)
    
    credentials = Credentials(
        url="https://eu-de.ml.cloud.ibm.com",
        token=access_token
    )

    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 100,
        "repetition_penalty": 1.0
    }

    model = Model(
        model_id="sdaia/allam-1-13b-instruct",
        params=parameters,
        credentials=credentials,
        project_id=os.getenv("PROJECT_ID")
    )
    return model

# Main function to generate a story
def generate_story(words, sentences):
    # Construct the query and prompt
    query = " ".join(words + sentences)
    retrieved_contexts = search_similar_documents(query, top_k=2)  # Adjust top_k as needed
    
    # Construct the refined system prompt using the retrieved context
    prompt = (
        f"Using the keywords provided, generate a children's story in Arabic based on the retrieved context:\n"
        f"{' '.join(retrieved_contexts)}\n\n"
        f"Keywords: {query}\n\n"
        "The output must include:\n"
        "1. A title for the story in Arabic and English.\n"
        "2. A brief summary of the story in Arabic and English.\n"
        "3. The content of the story in Arabic, broken down into sentences.\n"
        "4. An English translation of the content, broken down into sentences.\n"
        "5. The minimum and maximum ages suitable for the story.\n\n"
        "Ensure the output is in this format:\n"
        "{\n"
        '"title": "title in Arabic",\n'
        '"title_en": "Title in English",\n'
        '"brief": "Brief in Arabic",\n'
        '"brief_en": "Brief in English",\n'
        '"content": ["first sentence in Arabic", "second sentence in Arabic", ...],\n'
        '"content_en": ["First sentence in English", "Second sentence in English", ...],\n'
        '"min_age": minimum age,\n'
        '"max_age": maximum age\n'
        "}\n"
        "Start the story with 'كان يا مكان كان هناك...' and make sure to use the provided keywords."
    )

    # Initialize model and generate the story
    model = initialize_model()
    
    try:
        response = model.generate_text(prompt=prompt)
        if isinstance(response, str):
            story_text = response
        elif isinstance(response, dict) and 'text' in response:
            story_text = response['text']
        else:
            raise ValueError("Unexpected response structure from the model.")
    except Exception as e:
        raise Exception(f"Error generating text: {str(e)}")

    # The response is expected to be formatted as per the system prompt
    return story_text  # Directly return the model's output since it's already structured
