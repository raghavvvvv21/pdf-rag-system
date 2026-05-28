from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()


docs = [
    Document(
        page_content="""
        Inception is a science fiction thriller directed by Christopher Nolan.
        The movie follows Dom Cobb, a thief who enters dreams to steal secrets.
        """
    ),

    Document(
        page_content="""
        Interstellar is a space exploration film directed by Christopher Nolan.
        It explores black holes, time dilation, and survival of humanity.
        """
    ),

    Document(
        page_content="""
        Breaking Bad is a crime drama television series created by Vince Gilligan.
        It follows Walter White's transformation into a drug kingpin.
        """
    )
]


embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="Chroma-DB"
    
)

result=vectorstore.similarity_search("series directed by vince",k=2)

for r in result:
    print(r.page_content)
