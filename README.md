🤖 Agentic AI System with Multi-LLM & RAG

An advanced Agentic AI system that combines LLM reasoning, tool usage, and Retrieval-Augmented Generation (RAG) to solve complex queries using multiple knowledge sources like Wikipedia, arXiv, and web data.

This project demonstrates how to build intelligent AI agents capable of decision-making, tool selection, and context-aware responses using modern AI frameworks.

📌 Overview

This repository showcases:

ReAct-based AI agents
Multi-tool integration (Wikipedia, arXiv, Web)
RAG pipelines with vector databases
Multi-LLM usage (Ollama, Groq, HuggingFace)
Production-ready agent with memory

Unlike traditional chatbots, this system enables reasoning + action + retrieval, making it closer to real-world AI assistants.

🧠 Key Features
🤖 ReAct-based intelligent agents
🔎 Tool integration (Wikipedia + arXiv + Web search)
📄 RAG-based document retrieval using FAISS
🧠 Conversation memory for context retention
⚡ Multi-LLM support (Ollama, Groq, HuggingFace)
🌐 Streamlit UI for interactive chat
📊 Reduced hallucination via grounded responses
🏗️ Architecture

User Query
↓
Agent (ReAct Framework)
↓
Thought → Action → Tool Selection
↓
Tools (Wikipedia / arXiv / Retriever)
↓
Context Retrieval (FAISS / Web)
↓
LLM (Ollama / Groq / HuggingFace)
↓
Final Answer

⚙️ Tech Stack
Python
LangChain
Ollama (LLM + Embeddings)
Groq API
HuggingFace Embeddings
FAISS (Vector Store)
Streamlit
Wikipedia API
arXiv API
📂 Project Structure

agents.py → Basic ReAct agent with tools
prod_agent.py → Production-grade agent with memory & improved prompting
chat_groq.py → Streamlit-based chat UI with Groq LLM
huggingface_mistral.py → RAG pipeline using HuggingFace embeddings

🔍 How It Works
1. Agent-Based Reasoning
Uses ReAct framework (Reason + Act)
Dynamically selects tools based on query
Executes step-by-step reasoning
2. Tool Integration
Wikipedia → General knowledge
arXiv → Research papers
Web Loader → External content
Retriever → Internal knowledge (FAISS)
3. RAG Pipeline
Documents are chunked
Converted into embeddings
Stored in FAISS
Retrieved based on semantic similarity
4. Multi-LLM Support
Ollama → Local LLM inference
Groq → High-speed cloud inference
HuggingFace → Custom embeddings
5. Production Agent Enhancements
Conversation memory
Structured prompting
Tool orchestration
Controlled reasoning steps

💡 Key Learnings
Agentic AI enables reasoning + tool usage
RAG reduces hallucination in LLMs
Multi-LLM architecture improves flexibility
Memory enhances conversational intelligence
