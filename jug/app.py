import os

from apikey import OPENAI_KEY

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# Testing
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# More Testing
# chat = ChatOpenAI(
#         temperature=0,
#         model = "gpt-3.5-turbo"
#         )

# UI for App
st.set_page_config(page_title="JUG")

with st.sidebar:
    st.title("JUG")
    st.subheader("The entity you need the most")

prompt = st.chat_input(disabled=False)

_THERAPY_ = """         
The following is a conversation between a therapist and client at therapist's office. The therapist has immense experience in the field of therapy. The therapist never reveals the answer directly but gently pushes the client towards a better narrative.
Current conversation:
{history}
Human: {input}
Therapist:
"""



therapy_template = PromptTemplate(
            input_variables=["history", "input"],
            template = _THERAPY_        
            )


# llms
llm = OpenAI(temperature=0)
therapy_chain = ConversationChain(llm = llm, verbose= True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun

for message in st.session_state.messages:
    print("message", message)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    with st.chat_message("client"):
        st.markdown(prompt)
    st.session_state.messages.append({"role" : "client", "content" : prompt})

    therapy = therapy_chain.predict(input = prompt, history = st.session_state.messages)
    with st.chat_message("Therapist"):
        st.markdown(therapy)
    print("Here")
    print(therapy)
    
    st.session_state.messages.append({"role" : "therapist", "content" : therapy})

    # with st.expander("Conversation History"):
    #     st.info(prompt)
    #     st.info(therapy_chain.memory.buffer)
