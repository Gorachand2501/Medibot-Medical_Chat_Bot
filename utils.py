import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Step 1: Setup LLM
def load_llm():
    # Retrieve Hugging Face API token from environment variables
    hf_token = os.environ.get("HF_TOKEN")
    huggingface_repo_id = os.environ.get("HUGGINGFACE_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3")

    hf_endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
        task="conversational"
    )
    llm = ChatHuggingFace(llm=hf_endpoint)
    return llm

# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say that you don’t know, don’t try to make up an answer. 
Don’t provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return prompt

# Step 3: Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # WARNING: Using allow_dangerous_deserialization=True. Only use for trusted sources.
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Step 4: Create RetrievalQA Chain
def get_qa_chain():
    llm = load_llm()
    db = get_vectorstore()
    prompt = set_custom_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain