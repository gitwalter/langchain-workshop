# Python Rules for LangChain Development

## Type Hints and Documentation

- Always use type hints for all function parameters and return values
- Use `from __future__ import annotations` for forward references
- Use Google-style docstrings with Args, Returns, Raises, Example sections

```python
from __future__ import annotations

def process_message(message: str, temperature: float = 0.7) -> dict[str, Any]:
    """Process a message through the LLM.
    
    Args:
        message: The input message to process.
        temperature: Sampling temperature (0.0-2.0).
    
    Returns:
        A dictionary containing the response and metadata.
    
    Raises:
        ValueError: If temperature is out of range.
    
    Example:
        >>> result = process_message("Hello", temperature=0.5)
        >>> print(result["content"])
    """
```

## Async Patterns

- Prefer async functions for I/O operations (API calls, file reads)
- Use `asyncio.gather()` for parallel operations
- Always use async context managers for resources

```python
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI

async def parallel_calls(prompts: list[str]) -> list[str]:
    """Execute multiple LLM calls in parallel."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    tasks = [llm.ainvoke(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return [r.content for r in results]
```

## Pydantic for Structured Outputs

- Use Pydantic models for all structured data
- Use `with_structured_output()` for LLM responses
- Define clear field descriptions

```python
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    """Result of text analysis."""
    sentiment: str = Field(description="Positive, negative, or neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    topics: list[str] = Field(description="Main topics identified")

# Use with LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm = llm.with_structured_output(AnalysisResult)
result = structured_llm.invoke("Analyze this text...")
```

## LCEL Chain Patterns

- Use pipe operator for readable chains
- Always include output parser
- Use RunnablePassthrough for parallel inputs

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("Summarize: {text}")
chain = prompt | llm | StrOutputParser()

# With parallel inputs
chain = (
    {"text": RunnablePassthrough(), "language": lambda x: x.get("lang", "English")}
    | prompt
    | llm
    | StrOutputParser()
)
```

## Tool Creation

- Use @tool decorator with Pydantic schema
- Provide clear docstrings (becomes tool description)
- Handle errors gracefully

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results to return")

@tool(args_schema=SearchInput)
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.
    
    Use this tool when you need to find current information online.
    """
    try:
        # Implementation
        return f"Results for: {query}"
    except Exception as e:
        return f"Search failed: {str(e)}"
```

## LangGraph State Management

- Use TypedDict with Annotated for state
- Use add_messages reducer for message lists
- Return only modified fields from nodes

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_step: str
    iteration: int

def process_node(state: AgentState) -> dict:
    """Process and return only modified fields."""
    return {
        "messages": [AIMessage(content="Processed")],
        "iteration": state["iteration"] + 1
    }
```

## Error Handling

- Use try-except with specific exceptions
- Implement fallbacks for LLM calls
- Log errors appropriately

```python
from langchain_core.runnables import RunnableLambda

def safe_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse", "raw": text}

# With fallback
chain = (
    prompt 
    | llm 
    | StrOutputParser()
).with_fallbacks([fallback_chain])
```

## Environment Variables

- Use python-dotenv for local development
- Never hardcode API keys
- Validate required variables at startup

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Validate required variables
required_vars = ["GEMINI_API_KEY"]  # LANGCHAIN_API_KEY optional for tracing
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise EnvironmentError(f"Missing: {', '.join(missing)}")
```

## Testing

- Use pytest with fixtures
- Mock external API calls
- Test both success and error paths

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_llm():
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock:
        mock.return_value.invoke.return_value.content = "Mocked response"
        yield mock

def test_chain_execution(mock_llm):
    result = my_chain.invoke({"input": "test"})
    assert "response" in result
```

## LangSmith Tracing

- Enable tracing in all environments
- Add metadata for filtering
- Use tags for organization

```python
import os

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# Add metadata when invoking
result = chain.invoke(
    {"input": "test"},
    config={
        "metadata": {"user_id": "123", "session": "abc"},
        "tags": ["production", "v1"]
    }
)
```

## Import Organization

```python
# Standard library
from __future__ import annotations
import os
import asyncio
from typing import Any, TypedDict, Annotated

# Third-party
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Local
from .utils import helper_function
```
