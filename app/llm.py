import os
from langchain_groq import ChatGroq

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192"
    )
