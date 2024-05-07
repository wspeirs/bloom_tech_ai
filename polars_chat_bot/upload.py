"""Upload the vectors to the DB"""
import os
import pinecone

from dotenv import load_dotenv
from json import load, dump

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

client = pinecone.Pinecone(api_key=pinecone_api_key)
index = client.Index(pinecone_index)

with open('embeddings.json', 'r') as f:
    embeddings = load(f)
    vectors = [(str(i), v['vector'], {'text': v['text']}) for i, v in enumerate(embeddings['vectors'])]

for i, vec in enumerate(vectors):
    # we should have chunked the text so everything fits nicely
    if len(vec[2]['text']) > 35000:
        continue

    index.upsert(vectors=[vec])
    print(f'Uploaded vector {i}')

