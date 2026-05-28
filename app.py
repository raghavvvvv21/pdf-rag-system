import streamlit as st
import os
import tempfile
import shutil
import uuid  # <-- Added to generate unique database names
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (API keys)
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Mistral RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Dynamic Mistral RAG Chatbot")

# --- Session State Initialization ---
# Keep track of the current database directory
if "db_dir" not in st.session_state:
    st.session_state.db_dir = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        # Check if this is a NEW file
        if st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processing new PDF and building database..."):
                try:
                    # 1. Clear the old chat history
                    st.session_state.messages = []
                    
                    # 2. Cleanup old database folder if it exists
                    if st.session_state.db_dir and os.path.exists(st.session_state.db_dir):
                        shutil.rmtree(st.session_state.db_dir, ignore_errors=True)

                    # 3. Create a NEW unique directory name for this session to avoid SQLite locks
                    new_db_dir = f"Chroma_DB_{uuid.uuid4().hex[:8]}"
                    st.session_state.db_dir = new_db_dir

                    # 4. Save the uploaded file to a temporary file path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    # 5. Load and parse the PDF
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()

                    # 6. Split the document into chunks
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)

                    # 7. Initialize the Mistral Embeddings model
                    embeddings_model = MistralAIEmbeddings(model="mistral-embed")

                    # 8. Create and persist the Chroma Vectorstore in the UNIQUE directory
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings_model,
                        persist_directory=st.session_state.db_dir
                    )
                    
                    # Clean up the temporary file
                    os.remove(tmp_file_path)
                    
                    # Mark as processed in session state
                    st.session_state.processed_file = uploaded_file.name
                    st.success(f"Successfully loaded '{uploaded_file.name}'!")
                    
                    # Clear caching so the retriever reloads the newly created DB
                    st.cache_resource.clear()
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

# --- Initialize RAG Components ---
@st.cache_resource
def initialize_rag(db_directory):
    # If the database directory doesn't exist yet, we can't initialize
    if not db_directory or not os.path.exists(db_directory):
        return None, None, None

    embeddings_model = MistralAIEmbeddings(model="mistral-embed")
    
    vectorstore = Chroma(
        persist_directory=db_directory,
        embedding_function=embeddings_model
    )
    
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )
    
    llm = ChatMistralAI(model='mistral-small-2506')
    
    prompt = ChatPromptTemplate.from_messages([
        (
            'system', '''you are a helpful AI assistant. 
             Use only the provided context to answer the question. 
             Also, you are friendly. '''
        ),
        ("human", """ 
        Context:{context}
         question:{question}
        """)
    ])
    
    return retriever, llm, prompt

# Load RAG components using the dynamic directory from session state
retriever, llm, prompt = initialize_rag(st.session_state.db_dir)

# --- Chat Interface Logic ---
if retriever is None:
    st.info("👈 Please upload a PDF file in the sidebar to get started!")
else:
    # Display previous chat messages on the screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Wait for user input
    query = st.chat_input("Ask something about your document...")

    if query:
        # 1. Display user query and save it to history
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # 2. Process AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking..."):
                # Retrieve relevant chunks from Chroma
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Format the prompt with retrieved context
                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })
                
                # Get response from Mistral
                response = llm.invoke(final_prompt)
                
                # Display response
                st.markdown(response.content)
                
        # 3. Save AI response to history
        st.session_state.messages.append({"role": "assistant", "content": response.content})