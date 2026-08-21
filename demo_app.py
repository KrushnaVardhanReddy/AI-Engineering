import streamlit as st
import logging
from dotenv import load_dotenv
load_dotenv(".env.local")
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Simple LLM Chatbot")

# Initialize LLM 
try:
    llm = ChatGroq(model="groq/compound")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    st.error("Error initializing LLM. Make sure GROQ_API_KEY is in your .env.local file.")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    logger.info(f"User input received: {prompt}")

    # Generate response
    with st.chat_message("assistant"):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            logger.info("LLM response generated successfully.")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            st.error("Failed to generate response.")
