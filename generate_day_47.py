import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_markdown(source: str):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source: str):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Theory Section
    theory_md = """# Day 47: Agent Correction Loops with LangGraph

Welcome to Day 47! Today we dive into **Correction Loops**, an essential pattern for building robust multi-agent systems and preventing catastrophic failures.

## Core Theory (Just-in-Time)

### The "Why"
LLMs hallucinate, and tools fail. If your agent calls a `calculator` tool with `{"expression": "5 / 0"}`, the tool will throw an error. A naive agent pipeline might simply crash or, worse, confidently pass that error back to the user as the final answer.
To build production-ready systems, agents need a mechanism to gracefully handle failures, interpret error messages, and try again. This is known as a **Correction Loop**.

### The "How"
In LangGraph, we achieve this by creating a state machine loop:
1. **Agent Node:** The LLM decides which tool to call or what the final answer is.
2. **Tool Node:** The requested tool is executed.
3. **Validation / Routing:** If the tool fails (e.g., raises an exception), the Tool Node intercepts the error, wraps it in a message, and passes it *back* to the Agent Node.
4. **Correction:** The LLM sees its previous tool call *and* the resulting error message. This prompts it to reason about the mistake and generate a corrected tool call.

We'll use LangGraph's `StateGraph`, `ToolNode`, and `tools_condition` to orchestrate this cycle.

## Common Pitfalls in Production

1. **Infinite Loops:** An agent might repeatedly make the same mistake (e.g., calling an API with the wrong parameter format). Always implement a maximum iteration limit (`recursion_limit`) in LangGraph to break out of infinite loops.
2. **Obscure Error Messages:** If a tool fails with an unhelpful error like "Internal Server Error" or a deep stack trace, the LLM won't know how to fix it. Catch exceptions in your tools and return clear, actionable feedback (e.g., "Error: 'salary' must be an integer, not a string").
3. **Assuming Tools Always Succeed:** Never assume a 3rd party API or database query will succeed. Always wrap tool execution logic in `try/except` blocks.
4. **State Bloat:** Repeatedly failing tool calls will bloat the conversation history (`messages` state). Over a long loop, you might exceed the LLM's context window. Implement pruning or summarizing if loops get too long."""

    add_markdown(theory_md)

    # Code Implementation Section
    code_intro_md = """## Code Implementation: Building a Correction Loop

Let's build an agent that uses a calculator tool. The tool will intentionally enforce strict rules to simulate common errors. When the agent messes up, the correction loop will guide it to the right answer."""

    add_markdown(code_intro_md)

    setup_code = """# Run this in your terminal if needed:
# uv pip install langchain langchain-openai langgraph pydantic

import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
import operator

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Set a dummy key if running locally for validation without hitting the API
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
"""
    add_code(setup_code)

    tool_code = """# Define a strict calculator tool
@tool
def strict_calculator(expression: str) -> str:
    \"\"\"
    Evaluates a mathematical expression.
    You must ONLY use numbers and basic operators (+, -, *, /).
    Do NOT use letters, variables, or functions like sqrt or pow.
    \"\"\"
    # PRODUCTION WARNING: Using eval() is dangerous.
    # We restrict inputs strictly for educational purposes.
    allowed_chars = set("0123456789+-*/.() ")
    if not set(expression).issubset(allowed_chars):
        # We intentionally return a clear error string rather than raising a fatal exception.
        # This string becomes the ToolMessage content, giving the LLM immediate feedback.
        return f"Error: Invalid characters in expression '{expression}'. Only numbers and (+, -, *, /) are allowed."

    try:
        # Evaluate the expression
        result = eval(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed. Please modify the denominator."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# We wrap the tool in a list for LangGraph
tools = [strict_calculator]
"""
    add_code(tool_code)

    graph_code = """# Define the Graph State
# LangGraph's MessagesState provides a standard `messages` key
# with built-in append/reduce logic.

# Initialize the LLM and bind the tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Define the Agent Node
def reasoner(state: MessagesState) -> dict:
    \"\"\"
    The agent node that decides what to do next.
    It looks at the conversation history and generates either a final answer
    or a tool call.
    \"\"\"
    messages = state["messages"]

    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        # Graceful degradation for local validation without real API keys
        return {"messages": [HumanMessage(content=f"LLM API Error (Mocked): {e}")]}

# Define the Tool Node
# LangGraph's prebuilt ToolNode automatically executes the tool requested by the LLM
# and returns the output as a ToolMessage. If the tool returns an error string (like ours),
# it's simply passed back to the LLM.
tool_node = ToolNode(tools)

# Build the Graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent", reasoner)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.add_edge(START, "agent")

# Use the prebuilt tools_condition to route.
# If the agent returned a tool_calls list, route to "tools".
# Otherwise, route to END.
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": END}
)

# Route from tools back to the agent to form the correction loop
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()
print("Graph compiled successfully. Correction loop ready.")
"""
    add_code(graph_code)

    execution_code = """def run_agent(query: str):
    \"\"\"Utility function to run the agent and print the interaction.\"\"\"
    print(f"\\n{'='*50}\\nUser Query: {query}\\n{'='*50}")

    inputs = {"messages": [HumanMessage(content=query)]}

    # We set recursion_limit to prevent infinite loops in case the LLM stubbornly fails
    try:
        for event in app.stream(inputs, stream_mode="values", config={"recursion_limit": 5}):
            message = event["messages"][-1]
            if isinstance(message, HumanMessage):
                pass # Already printed in query
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"🧠 Agent reasoning... Calling tool: {message.tool_calls[0]['name']} with args {message.tool_calls[0]['args']}")
            elif isinstance(message, ToolMessage):
                print(f"🛠️ Tool Result: {message.content}")
                if "Error" in message.content:
                    print(f"   --> ⚠️ Error detected! Routing back to Agent for correction...")
            else:
                print(f"✅ Final Answer: {message.content}")
    except Exception as e:
        print(f"Execution Error / Limit Reached: {e}")

# Run a simple, successful query
run_agent("What is 15 multiplied by 4?")

# Run a query designed to trigger our custom tool validation error
# The LLM might try to use `sqrt(144)` or similar, triggering our character check.
run_agent("What is the square root of 144? Use the calculator tool.")

# Run a query designed to trigger the division by zero error
run_agent("Divide 10 by 0.")
"""
    add_code(execution_code)

    # Lab Section
    lab_md = """## Practical Lab / Homework

**Your Task: The Strict JSON Formatter**

1. Create a new tool called `json_formatter` that takes a single string parameter: `data`.
2. The tool's job is to load the string as JSON using `json.loads()`.
3. **If it succeeds:** return a success message.
4. **If it fails (JSONDecodeError):** Catch the error and return a specific string: `"Error: Invalid JSON format. Did you use double quotes for keys?"`
5. Build a new LangGraph StateGraph (similar to the one above) that uses this tool.
6. Test it by asking the agent to format the following string: `"{'name': 'Alice', 'age': 30}"` (Note the single quotes, which are invalid JSON). Observe how the agent corrects itself based on your tool's error message."""
    add_markdown(lab_md)

    lab_code = """import json

@tool
def json_formatter(data: str) -> str:
    \"\"\"
    Validates if a string is properly formatted JSON.
    \"\"\"
    try:
        parsed = json.loads(data)
        return "Success: The string is valid JSON."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format. Did you use double quotes for keys? Details: {str(e)}"

# Setup tools and LLM
lab_tools = [json_formatter]
lab_llm_with_tools = llm.bind_tools(lab_tools)

# Define agent node
def lab_reasoner(state: MessagesState) -> dict:
    try:
        response = lab_llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [HumanMessage(content=f"LLM API Error (Mocked): {e}")]}

# Define Tool Node
lab_tool_node = ToolNode(lab_tools)

# Build Graph
lab_workflow = StateGraph(MessagesState)
lab_workflow.add_node("agent", lab_reasoner)
lab_workflow.add_node("tools", lab_tool_node)

lab_workflow.add_edge(START, "agent")
lab_workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
lab_workflow.add_edge("tools", "agent")

lab_app = lab_workflow.compile()

# Execute Lab
print("\\n--- Lab Execution ---")
query = "Format this data into JSON: {'name': 'Alice', 'age': 30}"
print(f"User Query: {query}")

try:
    for event in lab_app.stream({"messages": [HumanMessage(content=query)]}, stream_mode="values", config={"recursion_limit": 5}):
        message = event["messages"][-1]
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"🧠 Agent calling tool: {message.tool_calls[0]['name']} with args {message.tool_calls[0]['args']}")
        elif isinstance(message, ToolMessage):
            print(f"🛠️ Tool Result: {message.content}")
        elif not isinstance(message, HumanMessage):
             print(f"✅ Final Answer: {message.content}")
except Exception as e:
    print(f"Lab Execution Error: {e}")
"""
    add_code(lab_code)

    os.makedirs('modules', exist_ok=True)
    with open('modules/day_47.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
    print("Notebook modules/day_47.ipynb generated successfully.")
