import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Use your API key, or a dummy for local validation
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-dummy-key")

@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the chat session and agent."""

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
    """Handle incoming user messages and stream the response."""

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
