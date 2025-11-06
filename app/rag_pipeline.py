from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chains import RetrievalQA
from transformers import pipeline


def load_vector_db(file_path="data/sample_docs/example.txt"):
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split document into small chunks
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Use a small embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)
    return vector_db

def query_rag(query, vector_db):
    # Tiny T5 model that runs on CPU
    generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=200)
    llm = HuggingFacePipeline(pipeline=generator)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )
    return qa_chain.run(query)
