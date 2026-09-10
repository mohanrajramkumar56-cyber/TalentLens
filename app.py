# main.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

# --- Constants ---
# Directory to storgit add .git add .git commit -m "Initial commit"git config --global user.email "youremail@example.com"e uploaded resume files
UPLOAD_DIR = "uploaded_resumes"

# --- Helper Functions ---

def setup_directories():
    """
    Creates the necessary directories for the application to function
    if they don't already exist.
    """
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

def process_uploaded_files(uploaded_files):
    """
    Saves uploaded PDF files to the specified directory, extracts text content,
    and returns a list of LangChain Document objects. Each Document object
    represents a page from the PDFs.
    """
    if not uploaded_files:
        return None

    all_docs = []
    for uploaded_file in uploaded_files:
        # Save the file to the local directory
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the PDF and extract its content
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

    return all_docs

def get_text_chunks(documents):
    """
    Takes a list of LangChain Document objects and splits their text content
    into smaller, manageable chunks for processing.
    """
    if not documents:
        return None
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The size of each chunk in characters
        chunk_overlap=200 # The number of characters to overlap between chunks
    )
    return text_splitter.split_documents(documents)

def create_vector_store(text_chunks):
    """
    Creates a FAISS vector store from the provided text chunks. This involves
    generating embeddings for each chunk and storing them in a searchable index.
    """
    if not text_chunks:
        return None
    try:
        # Using an open-source model from Ollama for embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
        vector_store = FAISS.from_documents(text_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.info("Please ensure you have Ollama installed and running with the 'nomic-embed-text' model.")
        return None


def create_conversational_chain(vector_store):
    """
    Creates a conversational retrieval chain that combines a language model
    with the vector store to answer questions based on the resume content.
    """
    if not vector_store:
        return None
    try:
        # Using an open-source LLM from Ollama
        llm = Ollama(model="llama3")

        # Setting up memory to remember the conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Creating the main conversational chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        st.info("Please ensure you have Ollama installed and running with the 'llama3' model.")
        return None


# --- Streamlit App ---

def main():
    """The main function that runs the Streamlit web application."""
    st.set_page_config(page_title="Resume Chatbot", layout="wide")
    st.title("📄 RAG Resume Chatbot")
    st.markdown("---")

    # Ensure the necessary directories are created on startup
    setup_directories()

    # --- Sidebar for File Upload and Processing ---
    with st.sidebar:
        st.header("Upload Resumes")
        st.write("Upload one or more PDF resumes to get started.")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple resume PDFs at once."
        )

        if st.button("Process Resumes", key="process_button"):
            with st.spinner("Processing resumes... This may take a moment."):
                # Step 1: Process uploaded files and get document objects
                raw_text_docs = process_uploaded_files(uploaded_files)
                if not raw_text_docs:
                    st.warning("No files were uploaded or processed.")
                    st.stop()

                # Step 2: Split documents into text chunks
                text_chunks = get_text_chunks(raw_text_docs)
                if not text_chunks:
                    st.error("Failed to split documents into chunks.")
                    st.stop()

                # Step 3: Create the vector store from the text chunks
                vector_store = create_vector_store(text_chunks)
                if not vector_store:
                    st.error("Failed to create the vector store.")
                    st.stop()

                # Step 4: Create the conversational chain and store it in the session
                st.session_state.conversation = create_conversational_chain(vector_store)
                st.session_state.resumes_processed = True
                st.success("Resumes processed successfully! You can now ask questions.")


    # --- Main Chat Interface ---
    # Initialize session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "resumes_processed" not in st.session_state:
        st.session_state.resumes_processed = False


    if not st.session_state.resumes_processed:
        st.info("Please upload and process resumes using the sidebar to begin the chat.")
    else:
        st.info("Ask me anything about the candidates in the uploaded resumes!")

        # Display previous chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle new user input
        if prompt := st.chat_input("e.g., 'Who has experience with Python and Django?'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.conversation:
                with st.spinner("Thinking..."):
                    try:
                        # Get the response from the conversational chain
                        response = st.session_state.conversation({'question': prompt})
                        answer = response['answer']
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                    except Exception as e:
                        st.error(f"An error occurred during conversation: {e}")
            else:
                st.error("The conversation chain is not initialized. Please process resumes first.")


if __name__ == "__main__":
    main()