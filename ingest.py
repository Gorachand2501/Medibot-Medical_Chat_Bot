import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(find_dotenv())

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load documents
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create chunks
def create_chunks(extracted_data):
    # Use RecursiveCharacterTextSplitter for general-purpose text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Get embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

if __name__ == "__main__":
    documents = load_pdf_files(data_path=DATA_PATH)
    text_chunks = create_chunks(extracted_data=documents)
    embedding_model = get_embedding_model()

    # Create and save FAISS database
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("Database created and saved successfully.")