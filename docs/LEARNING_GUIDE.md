# LangChain Ecosystem Learning Guide

> A comprehensive guide covering LangChain, LangGraph, and LangSmith

## Introduction

This guide covers the complete LangChain ecosystem for building AI applications:

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| **LangChain** | Chains, tools, agents | Building LLM-powered applications |
| **LangGraph** | Stateful workflows | Complex multi-step, multi-agent systems |
| **LangSmith** | Observability | Debugging, testing, monitoring |

## Part 1: LangChain Fundamentals

### 1.1 LCEL (LangChain Expression Language)

LCEL is the declarative way to compose chains using the pipe operator.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simple chain
chain = prompt | llm | StrOutputParser()

# Invoke
result = chain.invoke({"input": "Hello"})
```

**Key Benefits:**
- Streaming support built-in
- Async support built-in
- Batching support built-in
- Retry/fallback support

### 1.2 Chat Models vs Completion Models

```python
# Chat Model (recommended)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Messages
from langchain_core.messages import HumanMessage, SystemMessage
response = llm.invoke([
    SystemMessage(content="You are helpful."),
    HumanMessage(content="Hello!")
])
```

### 1.3 Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

# Simple template
prompt = ChatPromptTemplate.from_template("Summarize: {text}")

# With system message
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
```

### 1.4 Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel

# String output
chain = prompt | llm | StrOutputParser()

# Structured output (recommended)
class Response(BaseModel):
    answer: str
    confidence: float

structured_llm = llm.with_structured_output(Response)
```

### 1.5 Tools and Agents

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for {query}"

# Create agent
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search])
result = executor.invoke({"input": "Find Python tutorials"})
```

---

## Part 2: LangGraph Workflows

### 2.1 Why LangGraph?

Use LangGraph when you need:
- **State management** across multiple steps
- **Conditional branching** and loops
- **Human-in-the-loop** patterns
- **Multi-agent orchestration**

### 2.2 State Definition

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-accumulates
    next_action: str
    iteration: int
```

**Reducers:**
- `add_messages`: Appends new messages to list
- Default: Last write wins

### 2.3 Building a Graph

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# Add nodes (functions)
graph.add_node("router", router_function)
graph.add_node("process", process_function)

# Set entry point
graph.set_entry_point("router")

# Add edges
graph.add_edge("process", END)

# Conditional edges
graph.add_conditional_edges(
    "router",
    route_function,  # Returns node name
    {"option_a": "node_a", "option_b": "node_b"}
)

# Compile
app = graph.compile()
```

### 2.4 Node Functions

```python
def my_node(state: AgentState) -> dict:
    """Nodes receive state, return modified fields only."""
    # Do something
    new_message = AIMessage(content="Processed")
    
    # Return ONLY modified fields
    return {
        "messages": [new_message],
        "iteration": state["iteration"] + 1
    }
```

### 2.5 Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["sensitive_action"]  # Pause here
)

# First invocation - pauses before sensitive_action
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(initial_state, config)

# Human reviews...

# Resume with approval
result = app.invoke({"approved": True}, config)
```

### 2.6 Supervisor-Worker Pattern

```python
def supervisor(state):
    """Route to appropriate worker."""
    task = analyze_task(state["messages"][-1])
    return {"next_worker": task}

def researcher(state):
    """Research worker."""
    return {"messages": [AIMessage(content="Research done")]}

def writer(state):
    """Writing worker."""
    return {"messages": [AIMessage(content="Content written")]}

# Build graph with supervisor routing to workers
```

---

## Part 3: LangSmith Observability

### 3.1 Enabling Tracing

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All LangChain calls are now traced!
```

### 3.2 Adding Metadata

```python
result = chain.invoke(
    {"input": "Hello"},
    config={
        "metadata": {
            "user_id": "123",
            "session": "abc"
        },
        "tags": ["production", "v1"]
    }
)
```

### 3.3 Creating Datasets

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset("qa-examples")

# Add examples
client.create_examples(
    inputs=[{"question": "What is LangChain?"}],
    outputs=[{"answer": "A framework for LLM apps"}],
    dataset_id=dataset.id
)
```

### 3.4 Running Evaluations

```python
from langchain.evaluation import load_evaluator

# Load built-in evaluators
qa_evaluator = load_evaluator("qa")

# Run evaluation
results = client.run_on_dataset(
    dataset_name="qa-examples",
    llm_or_chain_factory=lambda: my_chain,
    evaluation=[qa_evaluator]
)
```

### 3.5 Custom Evaluators

```python
def custom_evaluator(run, example):
    """Custom evaluation function."""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    
    # Calculate score
    score = 1.0 if reference.lower() in prediction.lower() else 0.0
    
    return {
        "key": "contains_answer",
        "score": score,
        "comment": f"Prediction length: {len(prediction)}"
    }
```

---

## Best Practices

### Error Handling

```python
from langchain_core.runnables import RunnableLambda

# With fallback
chain = (
    prompt | llm | parser
).with_fallbacks([fallback_chain])

# With retry
chain = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)
```

### Streaming

```python
# Stream tokens
for chunk in chain.stream({"input": "Hello"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"input": "Hello"}):
    print(chunk, end="", flush=True)
```

### Caching

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

---

## Common Patterns

### RAG (Retrieval-Augmented Generation)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create vector store
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG chain
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Conversational Memory

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

history = ChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history"
)
```

---

## Resources

- [LangChain Python Docs](https://python.langchain.com/docs/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangSmith Docs](https://docs.smith.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [DeepLearning.AI Courses](https://www.deeplearning.ai/short-courses/)

---

*Part of the Cursor Agent Factory Learning Workshop Ecosystem*
