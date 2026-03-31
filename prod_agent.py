from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory 

from dotenv import load_dotenv
import os


# PRODUCTION GRADE VERSION OF agents.py (MEMORY, PLANNER, STRUCTURED TOOLS, REFLECTION/RETRY INCLUDED)



load_dotenv()

user_agent=os.getenv("USER_AGENT")

# checking if LangChain API Key is found from env or not, if found set tracing to TRUE
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatOllama(model = "llama3.1", temperature=0)    # setting the llama3.1 model -- chatOllama is better for agents

# Prompt Template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpfull assistant. Please response to the user queries : "),
#         ("user", "Question: {question}")
#     ]
# )

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Wikipedia Tool
api_wrapper=WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200
)

wiki_tool=WikipediaQueryRun(
    api_wrapper=api_wrapper
    )

# Tool 2 Arxiv
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=400
)
arxiv_tool = ArxivQueryRun(api_wrapper= arxiv_wrapper)

# RAG

loader=WebBaseLoader("https://docs.langchain.com/langsmith/home")   # Tool used = WikipediaAPI
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vector_db=FAISS.from_documents(documents, OllamaEmbeddings(model="llama3.1"))
retriver=vector_db.as_retriever()
# print(retriver)

# in the below line we are converting the above retriever to a callable tool by AGENT
retriver_tool = create_retriever_tool(retriever=retriver, name="langchain_search", description="Search for information about LangChain. For any question in LangChain you must use this tool")


# Tools
tools = [retriver_tool, arxiv_tool, wiki_tool]     # Agent will go through these tool for retrival process

# Prompts
# prompt = PromptTemplate.from_template("""
# You are a helpful AI assistant.

# You MUST follow this format strictly:

# Question: {input}
# Thought: think step-by-step
# {tools}
# Action: one of [{tool_names}] OR "None"
# Action Input: input for the tool
# Observation: result of the tool
# ... (repeat if needed)
# Thought: I now know the final answer
# Final Answer: provide final answer here

# Rules:
# - Do NOT repeat after Final Answer
# - If no tool is needed, use Action: None
# - Keep answers concise

# {agent_scratchpad}
# """)

# PROMPTS WITH MEMORY
prompt = PromptTemplate.from_template("""
You are an intelligent AI assistant.

You can use tools to answer questions.

Chat History:
{chat_history}

Follow STRICT format:

Question: {input}
Thought: think step by step
Action: one of [{tool_names}] OR "None"
Action Input: input for tool
Observation: result
... (repeat if needed)
Thought: I now know the answer
Final Answer: final response

Rules:
- Use tools only when needed
- Keep answers concise
- Do NOT hallucinate
- If answer is known, skip tools

{agent_scratchpad}
""")


# creating agent
agent = create_react_agent(llm, tools, prompt)

# agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory,
    verbose=True, 
    max_iterations=3)
agent_executor.invoke({
    "input": "What is Python & why it's used ?"
})




