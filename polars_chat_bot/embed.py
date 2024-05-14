"""Embed the documents to get vectors"""
import os
import time

from dotenv import load_dotenv
from json import load, dump
from langchain.schema.document import Document
from langchain_community.embeddings import BedrockEmbeddings

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

MODEL='arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-g1-text-02'

# read in all the docs from the JSON file
start_time = time.time()
with open('data.json', 'r') as f:
    docs = load(f)
    # docs = [Document(page_content=doc['kwargs']['page_content']) for doc in docs]
    docs = [doc['kwargs']['page_content'] for doc in docs]
print(f"Loaded {len(docs)} documents from JSON in {time.time() - start_time}s")

res = {
    'model': MODEL,
    'vectors': []
}

embeddings = BedrockEmbeddings(model_id=MODEL)
for i, doc in enumerate(docs):
    start_time = time.time()

    vals = embeddings.embed_query(doc)
    res['vectors'].append({'vector': vals, 'text': doc})

    dur = time.time() - start_time

    if i % 100 == 0:
        print(f"{i}: {dur}s to embed; ~{(dur*len(docs))/60}minutes for all")

with open('embeddings.json', 'w') as f:
    dump(res, f)

