import os
from langchain_groq import ChatGroq


def get_llm(streaming: bool = False):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        streaming=streaming  # ✅ key change
    )
