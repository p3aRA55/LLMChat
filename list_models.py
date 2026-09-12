import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv('OPENROUTER_API_KEY'), base_url='https://openrouter.ai/api/v1')

print("🔍  List of Available model: ")
models = client.models.list()
free_models = [m.id for m in models.data if ':free' in m.id]
for i, m in enumerate(free_models, 1):
    print(f"{i}. {m}")