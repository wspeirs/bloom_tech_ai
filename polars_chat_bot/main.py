"""Main loop of the RAG chat bot"""
from langchain_core.prompts import PromptTemplate

import os
import time

from dotenv import load_dotenv
from json import load, dump
from langchain_pinecone import PineconeVectorStore
from langchain.schema.document import Document
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_community.llms import Bedrock
from langchain_core.messages import HumanMessage


load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

embeddings = BedrockEmbeddings(model_id='arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-g1-text-02')
llm = ChatBedrock(model_id='amazon.titan-tg1-large', model_kwargs={"temperature": 0.1, "maxTokenCount": 4096})
# llm = Bedrock(model_id='amazon.titan-tg1-large', model_kwargs={"temperature": 0.1, "maxTokenCount": 4096})

vector_store = PineconeVectorStore(pinecone_api_key=pinecone_api_key, index_name=pinecone_index, embedding=embeddings)

retriever = vector_store.as_retriever()

query = "Please provide an example in Python of the correlation function"

context = retriever.get_relevant_documents(query)
print(f"Found {len(context)} documents")

context_str = ""
for doc in context:
    context_str += doc.page_content
    # if 'corr' in doc.page_content:
    #     print("Found correlation")
    # print(f"Content: {doc.page_content[:100]}")

template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
prompt_with_context = template.invoke({"query": query, "context": context})
result = llm.invoke(prompt_with_context)

# print(f"{query} Context: {context_str}")
# msg = HumanMessage(content=f"{query} Context: {context}")
# result = llm(f"{query} Context: {context_str}")

print("*** RESPONSE ***")
print(result.content)
