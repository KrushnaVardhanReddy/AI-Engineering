import json

def create_notebook():
    notebook_dict = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_markdown(content):
        notebook_dict["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" if i < len(content.split("\n")) - 1 else line for i, line in enumerate(content.split("\n"))]
        })

    def add_code(content):
        notebook_dict["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" if i < len(content.split("\n")) - 1 else line for i, line in enumerate(content.split("\n"))]
        })

    # ---------------------------------------------------------
    # Core Theory
    # ---------------------------------------------------------
    theory_md = """# Day 44: Implement "Statelessness" in LangGraph

Welcome to Day 44! Today we explore a crucial architectural pattern for reliable Multi-Agent Systems (MAS): **Statelessness** within LangGraph nodes.

When transitioning from scripts to production-grade agents, state management is often the biggest source of bugs. LangGraph provides a robust state object that is passed from node to node. The principle of statelessness means that *each node's execution must depend only on the current state object it receives*.

## The "Why" and "How"

**Why Stateless Nodes?**
1.  **Idempotency and Retries:** If a node fails (e.g., due to a temporary API timeout) and needs to be retried, its outcome shouldn't depend on hidden local variables or instance variables modified during the first attempt.
2.  **Concurrency and Scalability:** In a production server, multiple agents might run concurrently. If nodes use global variables or shared mutable instance states, race conditions and unpredictable behavior will occur.
3.  **Testability:** Stateless functions are purely deterministic based on their inputs. You can easily unit test a node by passing a mock state object without needing to set up complex object states or environment variables.

**How to Implement Statelessness:**
*   **Never modify global variables or class instance variables within a node.**
*   **Read only from the `state` dictionary provided to the node.**
*   **Return a dictionary containing *only* the state updates.** LangGraph's reducer functions (like `operator.add` for lists, or standard dict updating) handle merging this returned dictionary into the global state.

Let's look at the implementation."""
    add_markdown(theory_md)

    # ---------------------------------------------------------
    # Code Implementation
    # ---------------------------------------------------------
    impl_md = """## 1. Stateless Node Implementation

Let's build a simple multi-step pipeline and ensure the nodes strictly follow the statelessness principle. We will use `typing.TypedDict` for strict state definition."""
    add_markdown(impl_md)

    impl_code = """from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, START, END

# 1. Define the State strictly using TypedDict
class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    current_status: str

# 2. Define Stateless Nodes
def initialize_node(state: AgentState) -> dict:
    \"\"\"
    Initializes the process.
    Reads from state (though empty initially) and returns state updates.
    \"\"\"
    print(f"[initialize_node] Received state: {state}")

    # Pure function behavior: Return the updates dictionary
    return {
        "messages": ["Process initialized."],
        "current_status": "initialized"
    }

def process_node(state: AgentState) -> dict:
    \"\"\"
    Processes the data based ONLY on the input state.
    \"\"\"
    print(f"[process_node] Received state: {state}")

    # We read from the state
    current_status = state.get("current_status", "unknown")

    if current_status == "initialized":
        new_status = "processing_complete"
        new_message = "Data processed successfully."
    else:
        new_status = "error"
        new_message = "Invalid state for processing."

    # We return ONLY the updates
    return {
        "messages": [new_message],
        "current_status": new_status
    }

# 3. Build the Graph
builder = StateGraph(AgentState)
builder.add_node("init", initialize_node)
builder.add_node("process", process_node)

builder.add_edge(START, "init")
builder.add_edge("init", "process")
builder.add_edge("process", END)

graph = builder.compile()

# 4. Execute the Graph
if __name__ == "__main__":
    initial_state = {"messages": [], "current_status": "started"}
    print("\\n--- Starting Execution ---")
    final_state = graph.invoke(initial_state)
    print(f"\\n--- Final State ---\\n{final_state}")
"""
    add_code(impl_code)

    # ---------------------------------------------------------
    # Common Pitfalls
    # ---------------------------------------------------------
    pitfalls_md = """## Common Pitfalls in Production

1.  **State Mutation in Place:** Instead of returning a dictionary of updates, developers sometimes modify the `state` dictionary directly within the node (e.g., `state["messages"].append("new msg")`). While Python allows this, it breaks LangGraph's internal state tracking and reducers, leading to hard-to-debug state inconsistencies, especially when features like time-travel or branching are introduced.
2.  **Relying on Object Instance State:** If you define a node as a method of a class (e.g., `def my_node(self, state):`), do not read or write to `self.some_variable`. The node must only depend on the `state` parameter.
3.  **Non-Deterministic Side Effects:** Performing operations that affect external systems without reflecting them in the state (e.g., writing to a file but not updating the state that the file was written) breaks idempotency if the node is retried."""
    add_markdown(pitfalls_md)

    # ---------------------------------------------------------
    # Practical Lab / Homework
    # ---------------------------------------------------------
    lab_md = """## Practical Lab: Refactoring Stateful Nodes

**Your Task:**
Below is an example of a "bad" implementation where a node relies on a class instance variable (`self.run_count`) and modifies the state in place.

Refactor this code to be strictly stateless.
1. Remove the dependency on class instance variables. (Hint: Move the count into the `AgentState` if you need to track it).
2. Ensure the node returns a state update dictionary rather than modifying the state directly."""
    add_markdown(lab_md)

    lab_code_bad = """# Bad Implementation (Do not run this, it is for reference)
# class BadAgent:
#     def __init__(self):
#         self.run_count = 0
#
#     def bad_node(self, state: AgentState):
#         self.run_count += 1
#         state["messages"].append(f"Run {self.run_count}")
#         state["current_status"] = "running"
#         return state # Returning the whole modified state instead of updates
"""
    add_code(lab_code_bad)

    lab_code = """# Lab Implementation
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, START, END

# 1. Update the state definition to include the tracking variable
class RefactoredState(TypedDict):
    messages: Annotated[List[str], operator.add]
    current_status: str
    run_count: int

# 2. Implement the stateless node
def stateless_node(state: RefactoredState) -> dict:
    \"\"\"
    A refactored, stateless version of the bad_node.
    It reads current run_count from state, and returns the updates.
    \"\"\"
    # Read from state
    current_count = state.get("run_count", 0)
    new_count = current_count + 1

    new_message = f"Run {new_count}"

    # Return ONLY the updates
    return {
        "messages": [new_message],
        "current_status": "running",
        "run_count": new_count
    }

# 3. Test the refactored node
builder = StateGraph(RefactoredState)
builder.add_node("run", stateless_node)
builder.add_edge(START, "run")
builder.add_edge("run", END)

graph = builder.compile()

if __name__ == "__main__":
    initial_state = {"messages": [], "current_status": "idle", "run_count": 0}
    print("\\n--- Starting Refactored Execution ---")
    final_state = graph.invoke(initial_state)
    print(f"\\n--- Final State ---\\n{final_state}")
"""
    add_code(lab_code)

    # Write notebook to file
    with open('modules/day_44.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    create_notebook()
