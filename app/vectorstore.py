import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

VECTOR_DIR = "data/index"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def build_vectorstore(pdf_path: str) -> int:
    os.makedirs(VECTOR_DIR, exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)

    return len(chunks)



def load_vectorstore():
    if not os.path.exists(VECTOR_DIR):
        raise RuntimeError("Vectorstore not found. Upload a PDF first.")

    try:
        return FAISS.load_local(
            VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        # Corrupted or incompatible index
        print("⚠️ FAISS load failed, deleting index:", e)
        import shutil
        shutil.rmtree(VECTOR_DIR)
        raise RuntimeError("Vectorstore was incompatible. Please upload PDF again.")

