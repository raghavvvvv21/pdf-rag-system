from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

embedings_model=MistralAIEmbeddings(
    model="mistral-embed"
)
vectorstore=Chroma(
    
    persist_directory="Chroms_DB",
  embedding_function=embedings_model
   
)

retriver=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5

    }
)
llm=ChatMistralAI(model='mistral-small-2506')





#prompt_template

prompt=ChatPromptTemplate.from_messages([
    (
        'system','''you are helpful Ai assistant 
         use only the provied context to answer the question
          the question, also you are friendly  '''
    
    ),
    ("human",""" 
    Context:{context}
     question:{question}
""")
])
 

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriver.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    
