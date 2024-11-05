import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)

collection_name = os.getenv('nooran_x_allam')
print (collection_name)