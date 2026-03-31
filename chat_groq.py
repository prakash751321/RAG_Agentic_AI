import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS

load_dotenv()

# loading groq api key from .env
groq_api_key=os.getenv("GROQ_API_KEY")

# to prevent rerun we are storing the datas in the session
if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings(model="llama3.1")  # creating the embeddings & stored in session
    st.session_state.loader=WebBaseLoader("https://docs.langchain.com/")  #  loading the given webpage & stored in session
    st.session_state.docs=st.session_state.loader.load()  #  loaded
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)  #  splitting docs & stored in session
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  #  creating vector db & stored in session

st.title("Chat With Groq")  #   definded the title
llm=ChatGroq(groq_api_key=groq_api_key, model="openai/gpt-oss-120b")    # definded llm with model
prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on there provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
"""
)

document_chain=create_stuff_documents_chain(llm, prompt)  #  it will create doc processing chain
retriever=st.session_state.vector.as_retriever()
retrieval_chain=create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input Your Prompt Here ")

if prompt:
    with st.spinner("Generating response...."):
        response = retrieval_chain.invoke({"input": prompt})
        st.write(response["answer"])