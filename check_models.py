import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = client.models.list()

print("\nAvailable Groq Models:\n")

for model in models.data:
    if model.active:
        print(model.id)