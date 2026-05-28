from langchain_community.document_loaders import PyPDFLoader
#from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

data=PyPDFLoader("document_loader/transformer_research.pdf")
#splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
docs=data.load()
#print(len(docs))
#print(docs[10])

chunks=splitter.split_documents(docs)

print(chunks[0].page_content)