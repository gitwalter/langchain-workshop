"""LangSmith Tracing Setup Example.

This module demonstrates how to set up LangSmith tracing.

Workshop: L16 LangSmith Observability - Exercise 1
Objective: Add tracing to an application and analyze execution traces.

Learning Goals:
- Configure environment variables for tracing
- Add metadata and tags to runs
- View and analyze traces in LangSmith UI
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-workshop")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_traced_chain():
    """Create a chain with LangSmith tracing enabled.
    
    Tracing is automatically enabled when LANGCHAIN_TRACING_V2=true.
    All LangChain calls will be traced and visible in LangSmith UI.
    
    Returns:
        A runnable chain with automatic tracing.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question concisely.
        
Question: {question}

Answer:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    return chain


def invoke_with_metadata(chain, question: str, user_id: str = "workshop-user"):
    """Invoke a chain with metadata for filtering in LangSmith.
    
    Metadata and tags help you filter and organize runs in LangSmith.
    
    Args:
        chain: The chain to invoke.
        question: The question to ask.
        user_id: User identifier for metadata.
    
    Returns:
        The chain result.
    """
    result = chain.invoke(
        {"question": question},
        config={
            "metadata": {
                "user_id": user_id,
                "session": "workshop-exercise",
                "version": "1.0"
            },
            "tags": ["workshop", "tracing", "demo"]
        }
    )
    
    return result


def run_multiple_traces():
    """Run multiple traces to demonstrate LangSmith features.
    
    After running, go to LangSmith UI to:
    1. View the run tree
    2. Filter by tags
    3. Analyze timing and costs
    4. Add feedback
    """
    chain = create_traced_chain()
    
    questions = [
        "What is LangSmith?",
        "How do I enable tracing?",
        "What is a run tree?",
    ]
    
    results = []
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        result = invoke_with_metadata(chain, question, user_id=f"user-{i}")
        print(f"Answer: {result[:100]}...")
        results.append(result)
    
    return results


def main():
    """Run tracing examples."""
    print("=" * 60)
    print("LangSmith Tracing Example")
    print("=" * 60)
    
    # Check configuration
    print("\nConfiguration:")
    print(f"  LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"  LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")
    print(f"  LANGCHAIN_API_KEY: {'*' * 8 if os.getenv('LANGCHAIN_API_KEY') else 'NOT SET'}")
    
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("\n⚠️  WARNING: LANGCHAIN_API_KEY not set!")
        print("   Tracing will not work without a valid API key.")
        print("   Get your key at: https://smith.langchain.com/")
        return
    
    print("\n" + "-" * 60)
    print("Running traced chains...")
    print("-" * 60)
    
    results = run_multiple_traces()
    
    print("\n" + "=" * 60)
    print("✅ Traces sent to LangSmith!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Go to https://smith.langchain.com/")
    print(f"2. Navigate to project: {os.getenv('LANGCHAIN_PROJECT')}")
    print("3. Click on a run to see the run tree")
    print("4. Explore: inputs, outputs, timing, token usage")
    print("5. Try filtering by tags: 'workshop', 'tracing'")
    print("6. Add feedback to a run using the UI")


if __name__ == "__main__":
    main()
