# Day 49: Create a Visualization of Your Graph Architecture

## Concise Summaries
LangGraph provides built-in tools to visualize your graph architecture. The most straightforward method for terminal or log-based visualization is using `.draw_ascii()`. Visualizing your graph is essential for debugging routing logic, understanding node transitions, and communicating system design.

## Code Snippets

```python
from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    """Represents the state of the graph."""
    messages: Annotated[list[str], operator.add]


def node_a(state: GraphState) -> GraphState:
    """A dummy node."""
    return {"messages": ["Node A executed"]}


def node_b(state: GraphState) -> GraphState:
    """Another dummy node."""
    return {"messages": ["Node B executed"]}


def build_graph() -> StateGraph:
    """Constructs and returns the StateGraph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("node_a", node_a)
    workflow.add_node("node_b", node_b)

    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()

    # Visualize the graph in ASCII format
    print("Graph Visualization:")
    print(app.get_graph().draw_ascii())
```

## Key Concepts
*   **`.draw_ascii()`**: A built-in method on compiled LangGraph applications (`app.get_graph().draw_ascii()`) that returns an ASCII string representation of the graph.
*   **`grandalf`**: An external dependency required for `.draw_ascii()` to function.
*   **`StateGraph`**: The core component in LangGraph used to define the workflow, nodes, and edges.
*   **`START` / `END`**: Special nodes defining the entry and exit points of the graph execution.
*   **Security Context (PII)**: While visualizations themselves do not typically contain PII, avoid hardcoding sensitive data into node logic or state schemas that might inadvertently be logged alongside the graph representation.

## Common Gotchas
*   **Missing Dependency:** Failing to install the `grandalf` package will result in an error when calling `.draw_ascii()`. Always include it in your environment dependencies (`pip install grandalf`).
*   **Visualizing Uncompiled Graphs:** Attempting to visualize the graph before calling `.compile()` will fail, as the graph structure must be finalized first. Use `app = workflow.compile()` and then `app.get_graph()`.

## Reference Links
*   [LangGraph Documentation: Graph Visualization](https://langchain-ai.github.io/langgraph/how-tos/visualization/)
*   [grandalf PyPI](https://pypi.org/project/grandalf/)
