# 🧠 LLM Knowledge Assistant

A lightweight Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI** and **LangChain** that answers questions from your own documents using **Hugging Face open-source models** — no API keys required.

---

## 🚀 Features

- 📄 **Document Question-Answering**: Ask questions about any text file.
- 🔍 **Vector Search with FAISS** for fast, semantic retrieval.
- 🧩 **LangChain-based RAG Pipeline** for contextual answers.
- ⚡ **FastAPI Backend** with automatic Swagger UI (`/docs`).
- 🐳 **Docker Support** for easy deployment.
- 💻 **Completely Free** — uses Hugging Face models instead of paid APIs.

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.10 + |
| **Framework** | FastAPI |
| **AI Stack** | LangChain 1.x, Hugging Face Transformers, FAISS |
| **Embeddings** | sentence-transformers / all-MiniLM-L6-v2 |
| **Model** | google/flan-t5-base (CPU-friendly) |
| **Containerization** | Docker |
| **Environment** | .env file for model and port configuration |

---

## 📂 Project Structure

