# Streamlit & Groq API Cheatsheet

## Day 6: Rapid Prototyping with Streamlit and Groq

### 1. Core Concept Summary
Streamlit is a Python library that transforms data scripts into shareable web applications in minutes. Groq API provides ultra-low latency LLM inference using custom LPU (Language Processing Unit) hardware. Combining them allows you to build highly responsive, production-ready AI chat interfaces with sub-second response times.

### 2. Code Snippets: Streamlit Chat Interface with Groq

```python
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration & Initialization ---
st.set_page_config(page_title="Groq Chatbot", page_icon="⚡")
st.title("⚡ Ultra-Fast Groq Chatbot")

# Initialize ChatGroq LLM
# Note: Ensure GROQ_API_KEY is set in your environment variables
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7,
    max_tokens=1024,
    request_timeout=5.0
)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Chat History ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# --- Handle User Input ---
if prompt := st.chat_input("What is on your mind?"):
    # 1. Display User Message
    st.chat_message("user").write(prompt)

    # 2. Append User Message to State
    st.session_state.messages.append(HumanMessage(content=prompt))

    # 3. Get LLM Response
    try:
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.messages)

            # 4. Display & Append Assistant Message
            st.chat_message("assistant").write(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))

    except Exception as e:
        st.error(f"Error communicating with Groq API: {str(e)}")
```

### 3. Key Concepts
*   **`st.set_page_config()`:** Configures the main properties of the Streamlit page (must be the first Streamlit command).
*   **`st.session_state`:** A dictionary-like object that persists state across reruns of the Streamlit app. Essential for storing chat history.
*   **`st.chat_message()`:** UI component for displaying chat messages, differentiated by role (e.g., "user", "assistant").
*   **`st.chat_input()`:** UI component that renders a chat input widget at the bottom of the screen.
*   **`st.spinner()`:** A context manager that displays a loading spinner during long-running tasks.
*   **`ChatGroq`:** The LangChain integration for Groq, optimized for ultra-fast inference.

### 4. Common Gotchas
*   **State Reset on Rerun:** Streamlit reruns the entire script from top to bottom upon every user interaction. Always store variables that need to persist (like `messages`) in `st.session_state`; otherwise, your chat history will vanish after every message.
*   **Missing API Key:** Ensure your `GROQ_API_KEY` is loaded into the environment (e.g., via `python-dotenv` or direct `export`) before initializing `ChatGroq`.

### 5. Reference Links
*   [Streamlit Chat Elements Documentation](https://docs.streamlit.io/library/api-reference/chat)
*   [LangChain Groq Integration](https://python.langchain.com/v0.2/docs/integrations/chat/groq/)
