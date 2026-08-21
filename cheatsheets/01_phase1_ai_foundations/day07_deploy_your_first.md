# Day 7: Deploy Your First LLM-Powered App on Streamlit Cloud & Basic Logging

## Concise Summary
Streamlit Cloud provides a seamless way to deploy Python web apps directly from a GitHub repository. Integrating logging ensures you can monitor application state, user interactions, and LLM responses in production.

## Code Snippets

### Basic Logging Setup
```python
import logging

# Configure production-ready logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logger.info("Application started successfully.")
```

### Streamlit App with LLM Call & Logging
```python
import streamlit as st
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Simple LLM Chatbot")

# Initialize LLM (requires GROQ_API_KEY in Streamlit Secrets)
try:
    llm = ChatGroq(model="groq/compound")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    st.error("Error initializing LLM.")
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
```

## Key Concepts
- **Streamlit Secrets:** A secure way to store environment variables (like API keys) in Streamlit Cloud, accessible via `st.secrets`.
- **`st.session_state`:** Streamlit's mechanism to store data across reruns (e.g., maintaining chat history).
- **`logging` Module:** Python's built-in module for tracking events. Log levels include `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
- **Requirements File (`requirements.txt`):** Essential for deployment, telling Streamlit Cloud which dependencies to install.

## Common Gotchas
- **Missing `requirements.txt`:** Streamlit Cloud will fail to build if dependencies (like `langchain-groq`, `streamlit`) are missing from `requirements.txt`.
- **Exposing API Keys in Code:** Never hardcode API keys. Always use Streamlit Secrets in production (`st.secrets["GROQ_API_KEY"]`) or `.env` files locally.

## Reference Links
- [Streamlit App Deployment Documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
