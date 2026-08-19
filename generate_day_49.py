import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Day 49: Visualizing LangGraph Architecture\n",
                "\n",
                "## Core Theory (Just-in-Time)\n",
                "LangGraph is powerful for defining complex, stateful workflows for AI agents. However, as the number of nodes (agents or tools) and conditional edges (decision points) grows, understanding the execution flow simply by looking at the code becomes challenging. \n",
                "\n",
                "**Why visualize?**\n",
                "- **Debugging:** Quickly spot missing edges, dead ends, or disconnected nodes.\n",
                "- **Documentation:** Share the graph's architecture with non-technical stakeholders or team members.\n",
                "- **Mental Model:** Reinforce your understanding of how state transitions occur in your multi-agent system.\n",
                "\n",
                "**How?**\n",
                "LangGraph provides built-in methods to generate visual representations of the compiled graph. Specifically, `app.get_graph().draw_mermaid_png()` (and other variants like ASCII or Mermaid text) allows you to render the graph. The underlying mechanism relies on Mermaid, a JavaScript-based diagramming and charting tool that renders Markdown-inspired text definitions to create diagrams dynamically."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Code Implementation\n",
                "This snippet demonstrates how to define a simple state graph, compile it, and visualize it."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from typing import Annotated, TypedDict\n",
                "from langgraph.graph import StateGraph, START, END\n",
                "import operator\n",
                "\n",
                "# 1. Define the State schema\n",
                "class AgentState(TypedDict):\n",
                "    messages: Annotated[list[str], operator.add]\n",
                "\n",
                "# 2. Define node functions\n",
                "def node_a(state: AgentState) -> dict:\n",
                "    \"\"\"Simulates the first processing step.\"\"\"\n",
                "    return {\"messages\": [\"Node A processed.\"]}\n",
                "\n",
                "def node_b(state: AgentState) -> dict:\n",
                "    \"\"\"Simulates a secondary processing step.\"\"\"\n",
                "    return {\"messages\": [\"Node B processed.\"]}\n",
                "\n",
                "def node_c(state: AgentState) -> dict:\n",
                "    \"\"\"Simulates a fallback or alternative step.\"\"\"\n",
                "    return {\"messages\": [\"Node C processed.\"]}\n",
                "\n",
                "# 3. Define a conditional edge routing function\n",
                "def route_messages(state: AgentState) -> str:\n",
                "    \"\"\"\n",
                "    A dummy router that checks the last message to decide the next node.\n",
                "    In a real app, this might use an LLM output to route.\n",
                "    \"\"\"\n",
                "    if len(state[\"messages\"]) > 0 and \"Node A\" in state[\"messages\"][-1]:\n",
                "        return \"to_b\"  # Key mapping to next node\n",
                "    return \"to_c\"\n",
                "\n",
                "# 4. Build the Graph\n",
                "workflow = StateGraph(AgentState)\n",
                "\n",
                "# Add nodes\n",
                "workflow.add_node(\"agent_a\", node_a)\n",
                "workflow.add_node(\"agent_b\", node_b)\n",
                "workflow.add_node(\"agent_c\", node_c)\n",
                "\n",
                "# Add edges\n",
                "workflow.add_edge(START, \"agent_a\")\n",
                "\n",
                "# Add conditional edges\n",
                "workflow.add_conditional_edges(\n",
                "    \"agent_a\",\n",
                "    route_messages,\n",
                "    {\n",
                "        \"to_b\": \"agent_b\",\n",
                "        \"to_c\": \"agent_c\"\n",
                "    }\n",
                ")\n",
                "\n",
                "# Add edges to END\n",
                "workflow.add_edge(\"agent_b\", END)\n",
                "workflow.add_edge(\"agent_c\", END)\n",
                "\n",
                "# 5. Compile the graph\n",
                "app = workflow.compile()\n",
                "\n",
                "# 6. Visualize the graph (ASCII representation for notebook compatibility)\n",
                "print(\"ASCII Visualization:\")\n",
                "print(app.get_graph().draw_ascii())\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Practical Lab / Homework\n",
                "\n",
                "**Your Task:**\n",
                "1. Build a new graph with a different structure (e.g., a cycle where a 'reviewer' node sends work back to a 'writer' node).\n",
                "2. Compile the graph.\n",
                "3. Try visualizing the graph. Since rendering PNGs inline in Jupyter might require extra dependencies (`IPython.display.Image`), use `.draw_ascii()` or print the `.draw_mermaid()` string output."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Implement your lab assignment here\n",
                "from typing import Annotated, TypedDict\n",
                "from langgraph.graph import StateGraph, START, END\n",
                "import operator\n",
                "\n",
                "class LabState(TypedDict):\n",
                "    content: str\n",
                "    review_count: int\n",
                "\n",
                "def writer_node(state: LabState) -> dict:\n",
                "    \"\"\"Generates content.\"\"\"\n",
                "    return {\"content\": \"Draft content.\", \"review_count\": state.get(\"review_count\", 0)}\n",
                "\n",
                "def reviewer_node(state: LabState) -> dict:\n",
                "    \"\"\"Reviews content.\"\"\"\n",
                "    return {\"review_count\": state[\"review_count\"] + 1}\n",
                "\n",
                "def check_approval(state: LabState) -> str:\n",
                "    \"\"\"Routes back to writer if review count < 2, else finishes.\"\"\"\n",
                "    if state[\"review_count\"] < 2:\n",
                "        return \"needs_revision\"\n",
                "    return \"approved\"\n",
                "\n",
                "lab_workflow = StateGraph(LabState)\n",
                "lab_workflow.add_node(\"writer\", writer_node)\n",
                "lab_workflow.add_node(\"reviewer\", reviewer_node)\n",
                "lab_workflow.add_edge(START, \"writer\")\n",
                "lab_workflow.add_edge(\"writer\", \"reviewer\")\n",
                "lab_workflow.add_conditional_edges(\n",
                "    \"reviewer\",\n",
                "    check_approval,\n",
                "    {\n",
                "        \"needs_revision\": \"writer\",\n",
                "        \"approved\": END\n",
                "    }\n",
                ")\n",
                "\n",
                "lab_app = lab_workflow.compile()\n",
                "print(\"\\nLab ASCII Visualization:\")\n",
                "print(lab_app.get_graph().draw_ascii())\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Common Pitfalls\n",
                "- **Missing END node routing:** Forgetting to link terminal nodes to `END` causes the graph execution to hang or fail validation.\n",
                "- **Incorrect Conditional Edge Maps:** The dictionary mapping strings returned by the router function to node names must be exact. If the router returns a string not in the map, LangGraph will throw an error.\n",
                "- **PNG Rendering Dependencies:** `app.get_graph().draw_mermaid_png()` requires network access to a Mermaid rendering API by default, or specific local dependencies. Using `.draw_ascii()` or `.draw_mermaid()` (and pasting into an online Mermaid live editor) is often more reliable during rapid prototyping."
            ]
        }
    ],
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
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('modules/day_49.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)
