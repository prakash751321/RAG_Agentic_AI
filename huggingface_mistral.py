from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings deprecated
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
import numpy 


# NOT COMPLETED


# QA Chatbot App with HuggingFace & Mistral model

# Data Ingestion Layer
loader = PyPDFDirectoryLoader("./Us_Census")  # loading the entire folder(directory)
docs=loader.load()

# converting to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = text_splitter.split_documents(docs)
# print(chunked_docs)

# created Embeddings using HuggingFaceEmbeddings
# BAAI/bge-small-en-v1.5  embeddings model
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-base-en",
    model_kwargs = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)

num_arr=numpy.array(
    huggingface_embeddings.embed_documents(
        [chunked_docs[0].page_content]
    )
)
# print(num_arr.shape)

# stored in vector db
vector_store = FAISS.from_documents(chunked_docs[:100], huggingface_embeddings)

# creating retriver
retriver_tool = vector_store.as_retriever()

# query using similarity search
query = "WHAT IS HEALTH INSURANCE COVERAGE ?"
relivant_docs = vector_store.similarity_search(query)
print(relivant_docs[0].page_content)