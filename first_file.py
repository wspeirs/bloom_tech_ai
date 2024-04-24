import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("SHARED_OPENAI_KEY")

client = OpenAI(api_key=key)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke."}]
)

print(completion.choices[0].message)
