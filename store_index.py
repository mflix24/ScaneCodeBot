# importing local packages
from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os


# Step-01 : Setting the API KEY for embedding
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Step-02 : Calling the all functions that created through src.helper

# loading the documents through load_repo() function
documents = load_repo("repo/")
# splitting the docs into chunkings through text_splitter() function
text_chunks = text_splitter(documents)
# loading the embeddings through load_embedding() function
embeddings = load_embedding()


# Step-03 : creating the vector store database-chromadb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()