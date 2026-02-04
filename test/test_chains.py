"""Tests for LangChain chains.

Run with: pytest test/ -v
"""
import pytest
from unittest.mock import patch, MagicMock


def test_imports():
    """Test that all required packages are installed."""
    import langchain
    import langgraph
    import langsmith
    import pydantic
    
    assert langchain is not None
    assert langgraph is not None
    assert langsmith is not None
    assert pydantic is not None


def test_chain_creation():
    """Test that chains can be created."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = ChatPromptTemplate.from_template("Hello {name}")
    parser = StrOutputParser()
    
    assert prompt is not None
    assert parser is not None


def test_state_definition():
    """Test LangGraph state definition."""
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages
    
    class TestState(TypedDict):
        messages: Annotated[list, add_messages]
        counter: int
    
    state: TestState = {"messages": [], "counter": 0}
    assert state["counter"] == 0


@pytest.mark.skipif(
    not pytest.importorskip("langchain_google_genai", reason="Google GenAI not installed"),
    reason="Google GenAI not available"
)
def test_mock_llm():
    """Test with mocked LLM."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_llm:
        mock_llm.return_value.invoke.return_value.content = "Mocked response"
        
        # Create chain components
        prompt = ChatPromptTemplate.from_template("Test: {input}")
        
        # Verify mock works
        result = mock_llm.return_value.invoke("test")
        assert result.content == "Mocked response"
