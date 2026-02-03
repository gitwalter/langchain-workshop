"""Basic LangChain Chain Example.

This module demonstrates LCEL (LangChain Expression Language) chain composition.

Workshop: L7 LangChain Fundamentals - Exercise 1
Objective: Build a multi-step processing chain that summarizes and translates text.

Learning Goals:
- Understand LCEL pipe operator composition
- Use prompt templates and output parsers
- Chain multiple LLM calls together
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()


def create_summarize_chain():
    """Create a simple summarization chain.
    
    Returns:
        A runnable chain that summarizes text.
    
    Example:
        >>> chain = create_summarize_chain()
        >>> result = chain.invoke({"text": "Long article..."})
        >>> print(result)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in 2-3 sentences:\n\n{text}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    return chain


def create_summarize_and_translate_chain():
    """Create a chain that summarizes then translates text.
    
    This demonstrates chaining multiple LLM calls together,
    passing output from one step to the next.
    
    Returns:
        A runnable chain that summarizes and translates.
    
    Example:
        >>> chain = create_summarize_and_translate_chain()
        >>> result = chain.invoke({"text": "...", "language": "Spanish"})
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in 2-3 sentences:\n\n{text}"
    )
    
    translate_prompt = ChatPromptTemplate.from_template(
        "Translate the following text to {language}:\n\n{summary}"
    )
    
    # Chain composition: summarize, then translate
    chain = (
        # First, summarize the text
        {"summary": summarize_prompt | llm | StrOutputParser(), 
         "language": lambda x: x["language"]}
        # Then translate the summary
        | translate_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# TODO: Exercise - Implement your own chain
def create_custom_chain():
    """YOUR EXERCISE: Create a custom processing chain.
    
    Ideas:
    - Sentiment analysis then response generation
    - Text extraction then formatting
    - Question expansion then answer synthesis
    
    Returns:
        A runnable chain with your custom logic.
    """
    # TODO: Implement your chain here
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Your prompt template
    prompt = ChatPromptTemplate.from_template(
        # TODO: Define your prompt
        "Your prompt here: {input}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    return chain


def main():
    """Run chain examples."""
    print("=" * 60)
    print("LangChain LCEL Chain Examples")
    print("=" * 60)
    
    # Example text
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It provides a standard interface for chains, agents, and memory, making it easy
    to build complex applications. The framework supports multiple LLM providers
    and includes tools for debugging and monitoring through LangSmith.
    """
    
    # Test summarize chain
    print("\n1. Summarize Chain:")
    print("-" * 40)
    chain = create_summarize_chain()
    result = chain.invoke({"text": sample_text})
    print(f"Summary: {result}")
    
    # Test summarize and translate chain
    print("\n2. Summarize and Translate Chain:")
    print("-" * 40)
    chain = create_summarize_and_translate_chain()
    result = chain.invoke({"text": sample_text, "language": "Spanish"})
    print(f"Translated Summary: {result}")
    
    print("\n" + "=" * 60)
    print("Exercise: Implement create_custom_chain() and test it!")
    print("=" * 60)


if __name__ == "__main__":
    main()
