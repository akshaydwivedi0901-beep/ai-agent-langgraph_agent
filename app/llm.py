from langchain_groq import ChatGroq
import os

def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
