import os
from langchain_groq import ChatGroq

def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    return ChatGroq(
        model="mixtral-8x7b-32768",
        api_key=groq_api_key
    )
