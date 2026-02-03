"""Simple Agent with Custom Tools.

This module demonstrates creating agents with custom tools.

Workshop: L7 LangChain Fundamentals - Exercise 2
Objective: Create an agent with custom tools using @tool decorator.

Learning Goals:
- Define tools with @tool decorator
- Use Pydantic schemas for tool inputs
- Create agents with create_tool_calling_agent
- Execute agents with AgentExecutor
"""
from __future__ import annotations

import os
import ast
import operator
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


# Tool Input Schemas
class CalculatorInput(BaseModel):
    """Input for the calculator tool."""
    expression: str = Field(description="Mathematical expression to evaluate (e.g., '2 + 3 * 4')")


class SearchInput(BaseModel):
    """Input for the search tool."""
    query: str = Field(description="Search query to look up")


# Tool Definitions
@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression.
    
    Use this tool when you need to perform calculations.
    Supports: +, -, *, /, ** (power)
    
    Examples:
    - "2 + 3" returns "5"
    - "10 * 5 + 2" returns "52"
    - "2 ** 8" returns "256"
    """
    try:
        # Safe evaluation using AST
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return allowed_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                return allowed_operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")
        
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool(args_schema=SearchInput)
def search_web(query: str) -> str:
    """Search the web for information.
    
    Use this tool when you need to find current information online.
    Returns search results for the given query.
    """
    # Simulated search results (replace with real API in production)
    simulated_results = {
        "langchain": "LangChain is a framework for developing LLM-powered applications.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "langsmith": "LangSmith is a platform for debugging, testing, and monitoring LLM applications.",
    }
    
    query_lower = query.lower()
    for key, value in simulated_results.items():
        if key in query_lower:
            return f"Search results for '{query}':\n{value}"
    
    return f"Search results for '{query}':\nNo specific information found. Try a more specific query."


# TODO: Exercise - Implement your own tool
@tool
def your_custom_tool(input_text: str) -> str:
    """YOUR EXERCISE: Implement a custom tool.
    
    Ideas:
    - Text analyzer (word count, sentiment)
    - Code formatter
    - Date/time utilities
    - Unit converter
    
    Args:
        input_text: The input to process.
    
    Returns:
        The processed result.
    """
    # TODO: Implement your tool logic
    return f"Processed: {input_text}"


def create_agent():
    """Create an agent with tools.
    
    Returns:
        AgentExecutor ready to run.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    tools = [calculator, search_web]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to tools.
        
Use the calculator tool for any mathematical calculations.
Use the search_web tool to find information about topics.

Always explain your reasoning and show your work."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return executor


def main():
    """Run agent examples."""
    print("=" * 60)
    print("LangChain Agent with Tools Example")
    print("=" * 60)
    
    agent = create_agent()
    
    # Test 1: Math calculation
    print("\n1. Math Calculation:")
    print("-" * 40)
    result = agent.invoke({
        "input": "What is 25 * 4 + 10?",
        "chat_history": []
    })
    print(f"\nFinal Answer: {result['output']}")
    
    # Test 2: Search
    print("\n2. Web Search:")
    print("-" * 40)
    result = agent.invoke({
        "input": "What is LangGraph?",
        "chat_history": []
    })
    print(f"\nFinal Answer: {result['output']}")
    
    # Test 3: Combined
    print("\n3. Combined Query:")
    print("-" * 40)
    result = agent.invoke({
        "input": "Search for LangChain info and calculate 2^10",
        "chat_history": []
    })
    print(f"\nFinal Answer: {result['output']}")
    
    print("\n" + "=" * 60)
    print("Exercise: Add your_custom_tool to the agent and test it!")
    print("=" * 60)


if __name__ == "__main__":
    main()
