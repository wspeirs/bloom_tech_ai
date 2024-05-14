"""Upload the vectors to the DB"""
import os
import pinecone
import time

from dotenv import load_dotenv
from json import load, dump

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

start_time = time.time()

with open('embeddings.json', 'r') as f:
    embeddings = load(f)
    vectors = [(str(i), v['vector'], {'text': v['text']}) for i, v in enumerate(embeddings['vectors'])]

print(f"Read {len(vectors)} vectors in {time.time() - start_time}s")

client = pinecone.Pinecone(api_key=pinecone_api_key)
index = client.Index(pinecone_index)

start_time = time.time()

for i, idx in enumerate(range(0, len(vectors), 10)):
    index.upsert(vectors=vectors[idx:idx+10])
    print(f'Uploaded vector {i} of {len(vectors) / 10}: {time.time() - start_time}s')

print(f"Uploaded {len(vectors)} vectors in {time.time() - start_time}s")
