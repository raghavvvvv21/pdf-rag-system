#load pdfs
#split into chunks 
#create the embeddings
#store in teh chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

data=PyPDFLoader("document_loader/deeplearning.pdf")
docs=data.load()


splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks=splitter.split_documents(docs)

embeddings_model = MistralAIEmbeddings(
    model="mistral-embed",
)

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="Chroms_DB"
   
)
