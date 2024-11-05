import os
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv('.env', override=True)

def get_ibm_token(api_key):
    url = 'https://iam.cloud.ibm.com/identity/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
        'apikey': api_key
    }
    
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        print("IBM token retrieved")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()
    res_format = response.json()
    return res_format['access_token']

def setup_watsonx_model():
    api_key = os.getenv('IBM_API_KEY')
    YOUR_ACCESS_TOKEN = get_ibm_token(api_key)
    # Set up the credentials
    credentials = Credentials(
        url="https://eu-de.ml.cloud.ibm.com",
        token=YOUR_ACCESS_TOKEN
    )
    
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 100,
        "repetition_penalty": 1.0
    }
    
    model = Model(
        model_id=os.getenv("MODEL_ID"),
        params=parameters,
        credentials=credentials,
        project_id=os.getenv("PROJECT_ID")
    )
    return model

def get_watsonx_response(model, prompt):
    response = model.generate_text(prompt=prompt)
    return response
