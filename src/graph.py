"""LangGraph pipeline definition.

Topology: START → main → END

The MainAgent calls sub-agents as tools within its ReAct loop.
No fan-out/fan-in — the graph is a single node.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.base import load_agent
from src.schemas import PipelineState


def build_graph(debug: bool = False):
    """Build and compile the pipeline graph.

    Args:
        debug: If True, enables debug logging on the MainAgent.

    Returns a compiled LangGraph ready to invoke with:
        graph.invoke({"query": "Should I invest in Apple?", "errors": []})
    """
    g = StateGraph(PipelineState)

    agent = load_agent("main")
    agent.debug = debug
    g.add_node("main", agent.as_node())

    g.add_edge(START, "main")
    g.add_edge("main", END)

    return g.compile()
