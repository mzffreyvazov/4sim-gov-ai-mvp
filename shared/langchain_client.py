import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI  # correct import

def get_chat_llm(temperature=0.2):
    return ChatOpenAI(model="gpt-4o", temperature=temperature, openai_api_key=os.getenv("OPENAI_API_KEY"))
