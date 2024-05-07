"""Extract the text from the HTML files for embedding"""
import os
import time

from dotenv import load_dotenv
from json import dump
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.html_bs import BSHTMLLoader

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

# go through and grab all the docs from: https://docs.pola.rs/py-polars/html/reference/index.html
# loader = ReadTheDocsLoader('data/docs.pola.rs/py-polars/html/') # this is super-slow
loader = DirectoryLoader(
        path='data/docs.pola.rs/py-polars/',
        glob="**/*.html",
        show_progress=True,
        # use_multithreading=True,
        # max_concurrency=4,
        recursive=True,
        # sample_size=2,
        loader_cls=BSHTMLLoader,
        loader_kwargs={'bs_kwargs': {'features': 'lxml'}}
    )

# get all the docs, filter out those without content
start_time = time.time()
raw_docs = [d for d in loader.load() if len(d.page_content) > 0]
print(f"Found {len(raw_docs)} documents in {time.time() - start_time}s")

# split up our documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
split_docs = text_splitter.split_documents(raw_docs)

# save our docs as JSON so we can read them back more efficiently
with open('data.json', 'w') as f:
    dump([d.to_json() for d in split_docs], f)
