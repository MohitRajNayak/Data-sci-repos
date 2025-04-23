# Load all the keys

import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# OpenAI key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Langsmith Key for tracking the project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistance. Please respond to the question asked"),
        ("user","Question{question}")
    ]
)

## Stream lit framwork

st.title("Langchain demo with google gemma 2")
input_text = st.text_input("What qus you have in mind?")


## call ollama model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))