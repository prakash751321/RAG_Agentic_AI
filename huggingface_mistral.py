from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings deprecated
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain 
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace # huggingface hub replaced with huggingface hub
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import numpy

load_dotenv()  # loaded HF_TOKEN from .env (No need to declare)


# QA Chatbot App with HuggingFace & Mistral model

# Data Ingestion Layer
loader = PyPDFDirectoryLoader("./Us_Census")  # loading the entire folder(directory)
docs=loader.load()

# converting to chunks / splitting docs
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

# creating retriver MMR type
retriver_tool = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})

# HuggingFace LLM
llm = ChatGroq(
    model="llama3.1",
    temperature=0.1,
)

# chat wrapper required for enabling QA chatbot
# llm = ChatHuggingFace(llm=base_llm)

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the context.

    <context>
    {context}
    </context>

    Question: {input}
"""
)

# document_chain
document_chain = create_stuff_documents_chain(llm, prompt)

# retrieval chain
retrieval_chain = create_retrieval_chain(retriver_tool, document_chain)

# Query
query = "What is health insurance coverage ?"

response = retrieval_chain.invoke({
    'input': query
})

print("\n Final Answer : \n")
print(response['answer'])