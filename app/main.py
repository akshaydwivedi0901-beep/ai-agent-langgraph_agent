@app.post("/chat")
def chat(req: ChatRequest):
    try:
        db = load_vectorstore()
    except RuntimeError as e:
        return {"error": str(e)}

    docs = db.similarity_search(req.message, k=3)
    context = "\n".join(d.page_content for d in docs)

    llm = get_llm()
    response = llm.invoke(context + "\n\nQuestion: " + req.message)

    return {"answer": response.content}
