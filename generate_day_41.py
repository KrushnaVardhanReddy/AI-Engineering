import json

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_markdown(source: str):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True)
        })

    def add_code(source: str):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True)
        })

    theory_md = """# Day 41: Build a Professional UI for your Agent using Chainlit

Welcome to Day 41! Today, we transition from terminal outputs and basic scripts to a professional, chat-based User Interface (UI) using **Chainlit**. We'll focus on integrating your LangChain/LangGraph agents with real-time token streaming to create an interactive experience.

## Core Theory (Just-in-Time)

### The "Why"
Users expect instantaneous feedback when interacting with AI. Waiting 10-20 seconds for a full generation to complete feels broken. Real-time token streaming (printing words as they are generated) drastically improves perceived performance. Chainlit is designed specifically for Chat UIs with LLMs, managing the complexities of session state, message history, and asynchronous streaming streams out of the box.

### The "How"
Chainlit operates via event hooks. Instead of writing a standard top-to-bottom script, you define functions that Chainlit calls when specific events occur, such as:
- `@cl.on_chat_start`: Triggered when a user opens the chat. Good for initializing agents or memory.
- `@cl.on_message`: Triggered when a user sends a message. This is where your core agent logic and streaming happen.

To stream tokens from LangChain to Chainlit, you typically use `cl.AsyncLangchainCallbackHandler`. This handler bridges LangChain's internal events (like `on_llm_new_token`) directly to the Chainlit UI.
"""
    add_markdown(theory_md)

    setup_code = """# Setup - uncomment and run if needed
# !uv pip install chainlit langchain langchain-openai"""
    add_code(setup_code)

    implementation_md = """## Code Implementation

Here is a production-grade example of a simple LangChain agent integrated with Chainlit.

*Note: Since Chainlit runs as a separate server (via `chainlit run app.py`), the following cell writes the application code to a file `app.py`. To run the UI, you would execute `chainlit run app.py -w` in your terminal.*"""
    add_markdown(implementation_md)

    implementation_code = """import os

# Write the Chainlit app to a file for execution
app_code = '''import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Use your API key, or a dummy for local validation
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-dummy-key")

@cl.on_chat_start
async def on_chat_start() -> None:
    \"\"\"Initialize the chat session and agent.\"\"\"

    # 1. Initialize the model with streaming enabled
    # Note: Depending on the provider, streaming=True may be required.
    model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

    # 2. Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and concise AI assistant."),
        ("human", "{question}")
    ])

    # 3. Build the LCEL chain
    chain = prompt | model | StrOutputParser()

    # 4. Store the chain in the user session
    cl.user_session.set("chain", chain)

    await cl.Message(content="Hello! I am your assistant. How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message) -> None:
    \"\"\"Handle incoming user messages and stream the response.\"\"\"

    # 1. Retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    # 2. Create a Chainlit message to hold the streaming response
    msg = cl.Message(content="")
    await msg.send()

    try:
        # 3. Stream the response using astream
        # We use astream to yield chunks as they arrive from the LLM
        async for chunk in chain.astream(
            {"question": message.content}
        ):
            # Stream the chunk to the UI
            await msg.stream_token(chunk)

        # 4. Finalize the message update
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()
'''

# Write to app.py
with open("app.py", "w") as f:
    f.write(app_code)

print("Created app.py. To run the Chainlit UI, execute: chainlit run app.py -w in your terminal.")
"""
    add_code(implementation_code)

    lab_md = """## Practical Lab / Homework

**Task:**
Enhance the `app.py` script above to include **Conversation Memory**.
Currently, the chain only responds to the immediate user prompt and forgets past context.

1. Modify `app.py` to use `MessagesPlaceholder` in your prompt.
2. Initialize a memory component (e.g., a list of messages) in `cl.user_session` during `@cl.on_chat_start`.
3. Append user messages and AI responses to this memory list inside `@cl.on_message`.
4. Pass the memory list into the chain during `.astream()`.

*Hint: Chainlit's `cl.user_session` acts as an in-memory dictionary scoped to the current user's session.*"""
    add_markdown(lab_md)

    lab_code = """# Provide your completed memory-enabled Chainlit app code here.
# For example, rewriting the app_code string and saving it as `app_memory.py`.

app_memory_code = '''import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-dummy-key")

@cl.on_chat_start
async def on_chat_start() -> None:
    \"\"\"Initialize the chat session and agent.\"\"\"
    model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

    # Add MessagesPlaceholder for history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and concise AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()

    cl.user_session.set("chain", chain)
    # Initialize chat history
    cl.user_session.set("history", [])

    await cl.Message(content="Hello! I remember our conversations. How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message) -> None:
    \"\"\"Handle incoming user messages, stream the response, and manage memory.\"\"\"
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    try:
        response_content = ""
        # Pass history to the chain
        async for chunk in chain.astream(
            {"question": message.content, "history": history}
        ):
            await msg.stream_token(chunk)
            response_content += chunk

        await msg.update()

        # Append to history
        history.append(HumanMessage(content=message.content))
        history.append(AIMessage(content=response_content))
        cl.user_session.set("history", history)

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()
'''

with open("app_memory.py", "w") as f:
    f.write(app_memory_code)

print("Created app_memory.py.")
"""
    add_code(lab_code)

    test_md = """## Verification

We will just run a basic smoke test invoking a mock LLM setup to verify our environment works, handling any API key missing exceptions."""
    add_markdown(test_md)

    test_code = """from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-dummy-key")

llm = ChatOpenAI(model="gpt-3.5-turbo")
try:
    # Await invoke inside an async environment or just try synchronous invoke
    response = llm.invoke([HumanMessage(content="Hi!")])
    print("Success:", response.content)
except Exception as e:
    print(f"Expected API error (since we use a dummy key): {e}")
"""
    add_code(test_code)

    pitfalls_md = """## Common Pitfalls in Production

1. **Blocking Event Loop:** Using synchronous calls (like `.invoke()` instead of `.ainvoke()` or `.astream()`) inside Chainlit hooks blocks the main thread. This prevents other users' requests from being processed and breaks concurrent connections. Always use `async`/`await`.
2. **Redundant Streaming Hooks:** Using both LCEL `.astream()` for manual UI updates and `AsyncLangchainCallbackHandler` simultaneously can cause duplicate text rendering or conflicting UI messages.
3. **Session State Leaks:** Storing agent memory globally in variables outside of `@cl.on_chat_start` or `cl.user_session` will cause users to see each other's data. Always scope variables to `cl.user_session`.
4. **Unhandled Exceptions in Streams:** If the LLM provider times out or throws an error during the stream loop, you need a `try/except` block to catch it and notify the user (e.g., `await cl.Message(content="Error...").send()`), otherwise the chat window simply freezes.
"""
    add_markdown(pitfalls_md)

    with open("modules/day_41.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
    print("Notebook modules/day_41.ipynb created successfully.")
