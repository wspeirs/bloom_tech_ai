"""Extract the text from the HTML files for embedding"""
import time

from json import dump
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import BSHTMLLoader

# go through and grab all the docs from: https://docs.pola.rs/py-polars/html/reference/index.html
# loader = ReadTheDocsLoader('data/docs.pola.rs/py-polars/html/') # this is super-slow
loader = DirectoryLoader(
        path='data/',
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
