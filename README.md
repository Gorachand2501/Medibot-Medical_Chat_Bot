# 🤖 MediBot: Your AI Medical Assistant

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** using **LangChain** and **Streamlit** to answer medical questions based on a knowledge base of PDF documents.  
The model leverages **Hugging Face** for the LLM and embeddings, and **FAISS** for vector storage.

---

## 🎯 Objective
To create a chatbot that provides **accurate, context-aware answers** to medical queries by retrieving information from a private knowledge base.

---

## ✅ Features
- 📦 Ingests **PDF documents** to build a custom knowledge base.  
- 🧠 Creates a **FAISS vector database** for efficient semantic search.  
- 📝 Splits documents into chunks using **RecursiveCharacterTextSplitter**.  
- 🗣️ Utilizes a **Hugging Face LLM (Mistral-7B-Instruct-v0.3)** for generating answers.  
- 🎨 Provides a **user-friendly chat interface** with Streamlit.  
- 🔒 Manages **API keys securely** using Streamlit's secrets management.  

---

## 🛠️ Tech Stack
- **Python 3.x**  
- **Streamlit**  
- **LangChain**  
- **HuggingFace LLM**  
- **FAISS-CPU**  
- **python-dotenv**  
- **PyPDF**  
- **Sentence-transformers**  

---

## 🗂️ Knowledge Base
- **Data Directory**  
  - Location: `data/`  
  - Content: PDF files (`.pdf`) containing medical information.  

---

## 🧠 Model Architecture (RAG)

