from fastapi import FastAPI
from app.rag_pipeline import load_vector_db, query_rag

app = FastAPI(title="LLM Knowledge Assistant (Lightweight)")
vector_db = load_vector_db()

@app.get("/")
def home():
    return {"message": "LLM Knowledge Assistant is running!"}

@app.get("/query/")
def ask_question(q: str):
    answer = query_rag(q, vector_db)
    return {"question": q, "answer": answer}
