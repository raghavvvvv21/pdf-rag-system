# RAG Chatbot using Mistral AI

A Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **Mistral AI**, and **ChromaDB**.
The system allows users to upload PDF documents, store embeddings in a vector database, and ask questions grounded in the uploaded documents.

---

# Features

* PDF document ingestion
* Text chunking using RecursiveCharacterTextSplitter
* Embeddings using Mistral AI Embeddings
* Persistent vector storage using ChromaDB
* Retrieval-based question answering
* Friendly conversational AI responses
* Terminal-based chatbot interface
* Streamlit UI support

---

# Tech Stack

* Python
* LangChain
* Mistral AI
* ChromaDB
* Streamlit
* PyPDFLoader

---

# Project Structure

```bash
RAG-project/
│
├── app.py
├── main.py
├── create_db.py
├── requirements.txt
├── .env
│
├── document_loader/
│   └── deeplearning.pdf
│
├── retrivers/
│   └── arixv.py
│
└── vector_store/
```

---

# Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd RAG-project
```

---

## 2. Create virtual environment

```bash
uv venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
uv add -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

# Creating the Vector Database

Run:

```bash
python create_db.py
```

This will:

* Load the PDF
* Split text into chunks
* Generate embeddings
* Store embeddings inside ChromaDB

---

# Running the Chatbot

Run:

```bash
python main.py
```

You can now ask questions related to your uploaded document.

Example:

```text
You: Give me summary of chapter 1
AI: ...
```

---

# Streamlit UI

Run:

```bash
streamlit run app.py
```

---

# RAG Pipeline

```text
PDF Document
     ↓
Document Loader
     ↓
Text Splitter
     ↓
Embeddings
     ↓
Chroma Vector Database
     ↓
Retriever
     ↓
Mistral LLM
     ↓
AI Response
```

---

# Future Improvements

* Multi-document support
* Website URL ingestion
* Chat memory
* Source citations
* Hybrid search
* Better UI/UX
* Conversational memory
* Deployment on cloud

---

# Author

Raghav Sahu

B.Tech Mathematics & Computing
IIIT Raichur
