"""Embed the documents to get vectors"""
import os
import ollama
import time

from dotenv import load_dotenv
from json import load, dump
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain.schema.document import Document
from langchain.schema.document import Document

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

# read in all the docs from the JSON file
start_time = time.time()
with open('data.json', 'r') as f:
    docs = load(f)
    docs = [Document(page_content=doc['kwargs']['page_content']) for doc in docs]
print(f"Loaded {len(docs)} documents from JSON in {time.time() - start_time}s")

# Initialize the embeddings class
# start_time = time.time()
# embeddings = OllamaEmbeddings(model="llama3:text", temperature=0.25, num_gpu=1, show_progress=True)
# res = embeddings.embed_documents(docs)  # 4096
# print(f"Took {time.time() - start_time}s to embed documents")

embeddings = dict()
embeddings['model'] = 'qwen:0.5b-text'
embeddings['vectors'] = list()

for i, doc in enumerate(docs):
    start_time = time.time()
    res = ollama.embeddings(model="qwen:0.5b-text", prompt=doc.page_content)
    embeddings['vectors'].append({'vector': res['embedding'], 'text': doc.page_content})
    print(f"{i:03} of {len(docs)}: {time.time() - start_time}s to embed")

with open('embeddings.json', 'w') as f:
    dump(embeddings, f)

exit(1)

# connect to Pinecone
pc_vector_store = PineconeVectorStore(embedding=embeddings, pinecone_api_key=pinecone_api_key)


# Embed a single query
query = "Hello, world!"
vector = embeddings.embed_query(query)
print(vector[:5])

# Embed multiple documents at once
documents = ["Alice works in finance", "Bob is a database administrator", "Carl manages Bob and Alice"]
vectors = embeddings.embed_documents(documents)
print(len(vectors), len(vectors[0]))

db = FAISS.from_documents(documents, embeddings)
query = "Tell me about Alice"
docs = db.similarity_search(query)
print(docs[0].page_content)

# Perform a similarity search with scores
docs_and_scores = db.similarity_search_with_score(query)
print(docs_and_scores)

