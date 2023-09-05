import os

from langchain import chains
from langchain import memory
from apikey import OPENAI_KEY

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# UI for App
st.set_page_config(page_title="JUG")

with st.sidebar:
    st.title("JUG")
    st.subheader("The entity you need the most")

prompt = st.chat_input(disabled=False)

therapy_template = PromptTemplate(
        input_variables=["user_experience"],
        template = "You are an experienced therapy professional.Your client is saying this {user_experience}, respond appropriately"
        )

therapy_memory = ConversationBufferMemory(input_key="user_experience", memory_key="chat_history")

# llms
llm = OpenAI(temperature=0.9)
therapy_chain = LLMChain(llm = llm, prompt = therapy_template, verbose = True, output_key="therapy", memory=therapy_memory)

if prompt:
    therapy = therapy_chain.run(user_experience = prompt)
    st.write(therapy)

    with st.expander("Conversation History"):
        st.info(prompt)
        st.info(therapy_memory.buffer)
