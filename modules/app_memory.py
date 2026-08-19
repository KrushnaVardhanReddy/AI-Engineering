import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-dummy-key")

@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the chat session and agent."""
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
    """Handle incoming user messages, stream the response, and manage memory."""
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
