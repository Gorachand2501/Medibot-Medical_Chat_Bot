import streamlit as st
import os
from utils import get_qa_chain

# Use Streamlit's secrets management for API keys
os.environ["HF_TOKEN"] = st.secrets.HF_TOKEN
os.environ["HUGGINGFACE_REPO_ID"] = st.secrets.HUGGINGFACE_REPO_ID

# Initialize the QA chain
qa_chain = get_qa_chain()

st.set_page_config(
    page_title="MediBot: Your Medical Assistant 🩺",
    page_icon="🤖"
)

st.title("MediBot 💊")
st.write("Hello! I am your AI medical assistant. Ask me anything about medicine.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about health..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            # Invoke the RetrievalQA chain with the user's query
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Format the output to show result and source documents
            result_to_show = f"{result}\n\n**Source Documents:**\n"
            for doc in source_documents:
                result_to_show += f"- **Source:** {doc.metadata['source']} (Page: {doc.metadata.get('page', 'N/A')})\n"
                result_to_show += f"  - **Content:** {doc.page_content[:200]}...\n\n"

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(result_to_show)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {e}. Please try again.")
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}. Please try again."})