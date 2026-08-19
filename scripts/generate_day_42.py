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
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
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

    theory_md = """# Day 42: Human-in-the-Loop (HITL) with Chainlit and LangGraph

Welcome to Day 42 of the AI Engineering Mastery curriculum. Today we are focusing on a critical aspect of production AI systems: **Human-in-the-Loop (HITL)**.

When deploying autonomous agents that can take actions (like deleting records, sending emails, or triggering financial transactions), you cannot blindly trust the LLM. You need a mechanism to pause execution, request human approval, and then resume.

## Core Theory (Just-in-Time)

### The "Why"
Autonomous AI agents can hallucinate or misinterpret user intent. If an agent is hooked up to a critical API, a mistake can be disastrous. Implementing HITL ensures:
1.  **Safety & Security:** Sensitive actions require explicit human sign-off.
2.  **Quality Control:** Humans can correct the agent's course if it goes off track.
3.  **Compliance:** Many enterprise use cases legally require human oversight for automated decisions.

### The "How"
We will implement HITL using **LangGraph** (for the agent state machine) and **Chainlit** (for the user interface).
1.  **LangGraph State & Breakpoints:** LangGraph allows you to define "breakpoints" before or after specific nodes in your graph. When the graph execution hits a breakpoint, it pauses and waits for external input.
2.  **Chainlit Actions:** Chainlit provides UI elements (like buttons) that allow users to interact with the chat. We can use a Chainlit Action to capture the human's approval or rejection.
3.  **Resuming State:** Once the human provides input, we update the LangGraph state and resume execution.

## Common Pitfalls in Production
1.  **State Mismatch:** If the external UI (Chainlit) and the agent state (LangGraph) get out of sync, the system can hang indefinitely waiting for input that was already provided.
2.  **Timeout Handling:** In production, humans take time to respond. Your underlying infrastructure must handle long-running state pauses gracefully (e.g., using persistent state storage rather than in-memory).
3.  **Vague Approval Prompts:** If the user is asked to approve an action but isn't given the full context (what exactly is the agent about to do?), the human approval becomes a rubber stamp, defeating the purpose."""

    add_markdown(theory_md)

    setup_md = """## Setup and Dependencies

Let's install the necessary packages: `langgraph`, `chainlit`, and `langchain`."""

    add_markdown(setup_md)

    setup_code = """# Run this in your terminal if you haven't already:
# uv pip install langgraph chainlit langchain"""

    add_code(setup_code)

    impl_md = """## 1. LangGraph HITL Architecture

First, we will build a simplified LangGraph agent that has two nodes: `reasoning_node` and `action_node`. We want to pause *before* the `action_node` to get human approval.

*Note: Since Chainlit is designed to run as a standalone server, we will simulate the graph execution and state management here in standard Python to understand the mechanics, and then show how it integrates.*"""

    add_markdown(impl_md)

    impl_code = """from typing import TypedDict, Annotated, Sequence, Any
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Define the State
class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    approved: bool
    action_to_take: str

# 2. Define the Nodes
def reasoning_node(state: AgentState) -> dict:
    \"\"\"Simulates the LLM reasoning about what action to take.\"\"\"
    print("Agent is reasoning...")
    # In a real app, the LLM would decide this based on the user prompt
    action = "DELETE_DATABASE"
    return {"messages": ["Reasoning complete."], "action_to_take": action}

def action_node(state: AgentState) -> dict:
    \"\"\"Executes the action, but ONLY if approved.\"\"\"
    if not state.get("approved"):
         print("Action was NOT approved. Aborting.")
         return {"messages": ["Action aborted by user."]}

    print(f"Executing sensitive action: {state['action_to_take']}")
    return {"messages": [f"Action {state['action_to_take']} executed successfully."]}

# 3. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("reasoner", reasoning_node)
workflow.add_node("actor", action_node)

workflow.set_entry_point("reasoner")
workflow.add_edge("reasoner", "actor")
workflow.add_edge("actor", END)

# 4. Set the Breakpoint and Compile
# We need a checkpointer to save state when the graph pauses
memory = MemorySaver()

# We set an interrupt BEFORE the 'actor' node
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["actor"]
)

# 5. Simulate the Execution (The "Loop")
def run_simulation():
    thread_config = {"configurable": {"thread_id": "1"}}

    # Start execution
    print("--- Initial Run ---")
    for event in app.stream({"messages": ["Please optimize the DB"], "approved": False}, thread_config):
        print(event)

    # Check if we are paused
    state = app.get_state(thread_config)
    print(f"\\nGraph is currently paused: {len(state.next) > 0}. Next node to run: {state.next}")

    if "actor" in state.next:
        # Simulate Human Input (This is what Chainlit would capture)
        print("\\n--- Human Input Required ---")
        print(f"Agent wants to perform: {state.values.get('action_to_take')}")
        user_input = 'y' # Simulate user input

        is_approved = user_input.lower() == 'y'

        # Update the state with the human's decision
        app.update_state(thread_config, {"approved": is_approved})

        # Resume execution
        print("\\n--- Resuming Run ---")
        for event in app.stream(None, thread_config):
             print(event)

run_simulation()"""

    add_code(impl_code)

    chainlit_md = """## 2. Integrating with Chainlit

To move this into a real application, you must wrap this logic in a Chainlit application (`app.py`). Here is a full, runnable script of how Chainlit maps to the LangGraph breakpoint."""

    add_markdown(chainlit_md)

    chainlit_code = """%%writefile app.py
import chainlit as cl
from typing import TypedDict, Annotated, Sequence, Any
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    approved: bool
    action_to_take: str

def reasoning_node(state: AgentState) -> dict:
    action = "DELETE_DATABASE"
    return {"messages": ["Reasoning complete."], "action_to_take": action}

def action_node(state: AgentState) -> dict:
    if not state.get("approved"):
         return {"messages": ["Action aborted by user."]}
    return {"messages": [f"Action {state['action_to_take']} executed successfully."]}

workflow = StateGraph(AgentState)
workflow.add_node("reasoner", reasoning_node)
workflow.add_node("actor", action_node)

workflow.set_entry_point("reasoner")
workflow.add_edge("reasoner", "actor")
workflow.add_edge("actor", END)

memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["actor"]
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", "chainlit_thread_1")

@cl.on_message
async def main(message: cl.Message):
    # 1. Start or resume the graph
    thread_id = cl.user_session.get("thread_id")
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Stream the graph
    for event in app.stream({"messages": [message.content]}, thread_config):
         # Send intermediate steps to the user
         await cl.Message(content=str(event)).send()

    # 2. Check if we hit a breakpoint
    state = app.get_state(thread_config)
    if "actor" in state.next:
        # 3. Create a Chainlit Action (Button) for approval
        res = await cl.AskActionMessage(
            content=f"The agent wants to perform: {state.values.get('action_to_take')}. Do you approve?",
            actions=[
                cl.Action(name="approve", value="yes", label="Approve ✅"),
                cl.Action(name="reject", value="no", label="Reject ❌")
            ]
        ).send()

        if res and res.get("value") == "yes":
             # 4. Update state and RESUME
             app.update_state(thread_config, {"approved": True})
             # Resume graph by passing None
             for event in app.stream(None, thread_config):
                 await cl.Message(content=str(event)).send()
        else:
             app.update_state(thread_config, {"approved": False})
             for event in app.stream(None, thread_config):
                 await cl.Message(content=str(event)).send()
"""

    add_code(chainlit_code)

    chainlit_run_md = """To run this application, use the following command in your terminal:
`chainlit run app.py -w`"""

    add_markdown(chainlit_run_md)

    lab_md = """## 3. Practical Lab / Homework

**Task:**
You need to extend the provided standard Python LangGraph simulation to handle a more complex HITL scenario: **Correction**.

Instead of just `Approve` or `Reject`, the user should be able to provide feedback that forces the agent to re-reason.

1.  Modify the `AgentState` to include a `feedback` string field.
2.  Add a `human_review_node` that the graph routes to instead of interrupting before the `actor` node directly. This node is where the breakpoint will actually be. The conditional edge will be placed *after* this `human_review_node`. Note that your `human_review_node` should return an empty dict `{}` instead of using `pass` to satisfy strict type hinting.
3.  Add a conditional edge after `human_review_node`:
    *   If `approved` is true, go to `actor`.
    *   If `approved` is false AND `feedback` is provided, route back to the `reasoner` to try again.
4.  Update the `reasoning_node` to look at the `feedback` (if any) and change its intended action (e.g., from "DELETE_DATABASE" to "BACKUP_DATABASE").
5.  Run the simulation, reject the first action, provide feedback, and verify it loops back correctly."""

    add_markdown(lab_md)

    lab_code = """from typing import Literal

# Your implementation here

# 1. Update State


# 2. Update Nodes


# 3. Define Conditional Routing


# 4. Build Graph


# 5. Simulation Loop
"""

    add_code(lab_code)


    with open("modules/day_42.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
    print("Notebook modules/day_42.ipynb created successfully.")
