"""Simple LangGraph State Graph Example.

This module demonstrates LangGraph state management and routing.

Workshop: L3 LangGraph Workflows - Exercise 1
Objective: Create a graph with custom state and conditional routing.

Learning Goals:
- Define state with TypedDict and Annotated
- Use add_messages reducer for message accumulation
- Implement conditional edges and routing
- Handle END condition
"""
from __future__ import annotations

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()


# State Definition
class AgentState(TypedDict):
    """State schema for the agent graph.
    
    Attributes:
        messages: List of conversation messages (uses add_messages reducer).
        next_action: The next action to take (search, calculate, respond).
        iteration: Number of iterations through the graph.
    """
    messages: Annotated[list, add_messages]
    next_action: str
    iteration: int


# Node Functions
def router(state: AgentState) -> dict:
    """Analyze the last message and decide next action.
    
    This node examines the user's message and routes to the
    appropriate handler based on content.
    
    Args:
        state: Current agent state.
    
    Returns:
        Dict with next_action set.
    """
    if not state["messages"]:
        return {"next_action": "respond"}
    
    last_message = state["messages"][-1].content.lower()
    
    if any(word in last_message for word in ["search", "find", "look up"]):
        next_action = "search"
    elif any(word in last_message for word in ["calculate", "compute", "math", "+"]):
        next_action = "calculate"
    else:
        next_action = "respond"
    
    print(f"  [Router] Detected action: {next_action}")
    return {"next_action": next_action}


def search_node(state: AgentState) -> dict:
    """Handle search requests.
    
    Args:
        state: Current agent state.
    
    Returns:
        Dict with new message added.
    """
    query = state["messages"][-1].content
    result = f"Search results for: '{query}'\n- Result 1: LangGraph documentation\n- Result 2: Tutorial examples"
    
    print(f"  [Search] Executed search")
    return {
        "messages": [AIMessage(content=result)],
        "iteration": state["iteration"] + 1
    }


def calculate_node(state: AgentState) -> dict:
    """Handle calculation requests.
    
    Args:
        state: Current agent state.
    
    Returns:
        Dict with calculation result.
    """
    # Simulated calculation
    result = "Calculation result: 42 (simulated)"
    
    print(f"  [Calculate] Executed calculation")
    return {
        "messages": [AIMessage(content=result)],
        "iteration": state["iteration"] + 1
    }


def respond_node(state: AgentState) -> dict:
    """Generate a general response.
    
    Args:
        state: Current agent state.
    
    Returns:
        Dict with response message.
    """
    result = "I'm here to help! You can ask me to search for information or perform calculations."
    
    print(f"  [Respond] Generated response")
    return {
        "messages": [AIMessage(content=result)],
        "iteration": state["iteration"] + 1
    }


# Routing Function
def route_next(state: AgentState) -> Literal["search", "calculate", "respond"]:
    """Determine which node to execute next.
    
    Args:
        state: Current agent state.
    
    Returns:
        Name of the next node to execute.
    """
    return state["next_action"]


def create_graph():
    """Build the state graph with conditional routing.
    
    Returns:
        Compiled LangGraph application.
    """
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("router", router)
    graph.add_node("search", search_node)
    graph.add_node("calculate", calculate_node)
    graph.add_node("respond", respond_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_next,
        {
            "search": "search",
            "calculate": "calculate",
            "respond": "respond"
        }
    )
    
    # All action nodes go to END
    graph.add_edge("search", END)
    graph.add_edge("calculate", END)
    graph.add_edge("respond", END)
    
    # Compile the graph
    app = graph.compile()
    
    return app


# TODO: Exercise - Add a loop pattern
def create_graph_with_loop():
    """YOUR EXERCISE: Create a graph with a loop.
    
    The graph should:
    1. Start with a router
    2. Execute an action
    3. Check if more iterations needed
    4. Loop back or end
    
    Hint: Add a "check_done" node that decides whether to loop.
    """
    # TODO: Implement your looping graph
    graph = StateGraph(AgentState)
    
    # Add your nodes and edges here
    
    app = graph.compile()
    return app


def main():
    """Run graph examples."""
    print("=" * 60)
    print("LangGraph State Graph Example")
    print("=" * 60)
    
    app = create_graph()
    
    # Test 1: Search query
    print("\n1. Search Query:")
    print("-" * 40)
    result = app.invoke({
        "messages": [HumanMessage(content="Search for Python tutorials")],
        "next_action": "",
        "iteration": 0
    })
    print(f"Final message: {result['messages'][-1].content}")
    print(f"Iterations: {result['iteration']}")
    
    # Test 2: Calculation query
    print("\n2. Calculation Query:")
    print("-" * 40)
    result = app.invoke({
        "messages": [HumanMessage(content="Calculate 5 + 10")],
        "next_action": "",
        "iteration": 0
    })
    print(f"Final message: {result['messages'][-1].content}")
    
    # Test 3: General query
    print("\n3. General Query:")
    print("-" * 40)
    result = app.invoke({
        "messages": [HumanMessage(content="Hello, how are you?")],
        "next_action": "",
        "iteration": 0
    })
    print(f"Final message: {result['messages'][-1].content}")
    
    # Visualize the graph (requires graphviz)
    print("\n" + "=" * 60)
    print("Graph Visualization (Mermaid):")
    print("-" * 40)
    try:
        print(app.get_graph().draw_mermaid())
    except Exception as e:
        print(f"Visualization not available: {e}")
    
    print("\n" + "=" * 60)
    print("Exercise: Implement create_graph_with_loop() with iteration!")
    print("=" * 60)


if __name__ == "__main__":
    main()
