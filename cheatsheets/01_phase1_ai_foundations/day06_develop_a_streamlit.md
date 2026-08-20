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

### 6. Advanced OOP Implementation with AI Security

For production environments and interviews, demonstrating clean Object-Oriented Programming (OOP) principles and robust AI security (e.g., PII protection, secure key handling, fallback mechanisms) is critical. Here is a production-ready template.

```python
import streamlit as st
import os
import re
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class SecurityFilter:
    """Handles basic PII detection and redaction."""

    @staticmethod
    def mask_email(text: str) -> str:
        """Masks email addresses in user input."""
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        return re.sub(email_pattern, "[REDACTED EMAIL]", text)

    @staticmethod
    def sanitize_input(text: str) -> str:
        """Applies all security filters to the input."""
        text = SecurityFilter.mask_email(text)
        # Add other PII masking here (e.g., phone numbers, SSNs)
        return text

class StreamlitGroqChatbot:
    """Manages the Streamlit UI and Groq API interactions."""

    def __init__(self, model_name: str = "llama3-8b-8192", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self._initialize_session_state()

    def _initialize_llm(self) -> Optional[ChatGroq]:
        """Initializes the ChatGroq client securely."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("Authentication Error: GROQ_API_KEY environment variable is missing.")
            st.stop() # Halt execution if critical configuration is missing

        try:
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
                request_timeout=5.0
            )
        except Exception as e:
            st.error(f"Failed to initialize AI client: {str(e)}")
            return None

    def _initialize_session_state(self):
        """Initializes the chat history in Streamlit session state."""
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render_chat_history(self):
        """Renders the previous messages in the chat interface."""
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("assistant").write(msg.content)

    def generate_response(self, user_input: str) -> str:
        """Generates a response from the LLM with fallback mechanisms."""
        if not self.llm:
            return "System Error: AI service is currently unavailable."

        try:
            sanitized_input = SecurityFilter.sanitize_input(user_input)

            # Temporary copy of messages for context, appending the new sanitized input
            context = st.session_state.messages + [HumanMessage(content=sanitized_input)]

            response = self.llm.invoke(context)
            return response.content
        except Exception as e:
             # Graceful fallback on API failure (e.g., rate limits, timeout)
            st.error(f"Error during API call: {str(e)}")
            return "I apologize, but I am having trouble connecting to my knowledge base right now. Please try again later."

    def run(self):
        """Main execution loop for the Streamlit application."""
        st.set_page_config(page_title="Secure Groq Chat", page_icon="🛡️")
        st.title("🛡️ Secure & Fast Groq Chatbot")

        self.render_chat_history()

        if prompt := st.chat_input("Enter your message..."):
            # Display user message immediately
            st.chat_message("user").write(prompt)
            st.session_state.messages.append(HumanMessage(content=prompt))

            with st.spinner("Processing securely..."):
                response_text = self.generate_response(prompt)

                # Display and store assistant response
                st.chat_message("assistant").write(response_text)
                st.session_state.messages.append(AIMessage(content=response_text))

if __name__ == "__main__":
    # Ensure this script runs cleanly when executed directly
    app = StreamlitGroqChatbot()
    app.run()
```
