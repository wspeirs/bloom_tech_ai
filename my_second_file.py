import os

from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

load_dotenv()
key = os.getenv("SHARED_OPENAI_KEY")

llms = [OpenAI(model="gpt-3.5-turbo-instruct", api_key=key), Ollama(model="llama3")]

model_names = ["gpt-3.5-turbo-0125", "llama3"]

# First prompt
first_prompt = """
    I want you to tell me if the following game review is positive or negative.

    Review: It’s a great thing that Open Roads is an incredibly short experience, because when it comes to actually playing it, there just isn’t much there. Thankfully, its story and characters do more than enough to make experiencing Open Roads worthwhile, at least once.
    """
# Final prompt
final_prompt = """
    I want you to tell me if the following game review is positive or negative.

    Review: It’s a great thing that Open Roads is an incredibly short experience, because when it comes to actually playing it, there just isn’t much there. Thankfully, its story and characters do more than enough to make experiencing Open Roads worthwhile, at least once.

    The final response should be in the following format:
    sentiment: "positive" or "negative"
    reasoning: reason for giving the sentiment
"""

for i, llm in enumerate(llms):
    chain = llm | StrOutputParser()
    response = chain.invoke(first_prompt)
    result = f"{model_names[i].upper()}\n{response}\n_________________________\n"
    print(result)
