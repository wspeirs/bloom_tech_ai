import os

from dotenv import load_dotenv
from bs4 import BeautifulSoup as Soup
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pinecode_key = os.getenv("PINECONE_KEY")

# go through and grab all the docs from: https://docs.pola.rs/py-polars/html/reference/index.html
loader = ReadTheDocsLoader('data/docs.pola.rs') #, features="html.parser")
docs = loader.load()
# docs = loader.lazy_load()

# filter out those without content
docs = [d for d in docs if len(d.page_content) > 0]

print(len(docs))

exit(1)


pc = Pinecone(api_key=pinecode_key)

# Initialize the embeddings class
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

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

