# import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
# import openai

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
print("HF_TOKEN", os.environ['HF_TOKEN'])

groq_api_key=os.getenv("GROQ_API_KEY")
# print(groq_api_key)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# print("Embeddings",embeddings)
# from langchain_community.embeddings import HuggingFaceEmbeddings
# model_name = "sentence-transformers/all-mpnet-base-v2"                              ##"sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
# embeddings = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")
# print(llm)


loader=PyPDFDirectoryLoader('D:/Virtual Assistent/Groq/knowledge_base')
docs=loader.load()
# print(f"first documents loaded{docs[0]}")

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
splits=text_splitter.split_documents(docs)
vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
retriever=vectorstore.as_retriever()
# print('--->',retriever)

## Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)


response=rag_chain.invoke({"input":"For any kind of promise what should we say?"})
print(response)