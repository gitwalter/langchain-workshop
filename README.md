# LangChain Ecosystem Workshop

> Learning project for the Cursor Agent Factory AI Learning Path
> Covering **L7 LangChain** + **L3 LangGraph** + **L16 LangSmith**

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Unix

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env with your API keys

# Run your first chain
python src/chains/basic_chain.py
```

## Environment Variables

Set your Gemini API key (FREE tier available at https://aistudio.google.com/app/apikey):

```bash
# System environment variable (recommended)
GEMINI_API_KEY=your-gemini-key

# Optional: LangSmith for tracing (L16)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=langchain-workshop
```

## Project Structure

```
langchain-workshop/
├── src/
│   ├── chains/             # LangChain LCEL examples
│   │   ├── basic_chain.py  # Simple prompt | llm | parser
│   │   └── rag_chain.py    # Retrieval-augmented generation
│   ├── agents/             # Tool-using agents
│   │   ├── simple_agent.py # Basic agent with tools
│   │   └── research_agent.py
│   ├── graphs/             # LangGraph workflows
│   │   ├── simple_graph.py # Basic state graph
│   │   └── supervisor.py   # Supervisor-worker pattern
│   └── evaluation/         # LangSmith evaluation
│       ├── tracing.py      # Tracing setup
│       └── evaluator.py    # Custom evaluators
├── test/                   # Test files
├── docs/                   # Learning guides
│   └── LEARNING_GUIDE.md   # Comprehensive learning guide
├── .cursor/rules/          # AI coding rules
├── requirements.txt
└── README.md
```

## Workshop Phases

### L7: LangChain Fundamentals (2.5h)

| Phase | Duration | Focus |
|-------|----------|-------|
| Concept | 30 min | LCEL architecture, models, prompts |
| Demo | 30 min | Building a research assistant |
| Exercise 1 | 20 min | Multi-step processing chain |
| Exercise 2 | 25 min | Custom tool agent |
| Challenge | 30 min | Document Q&A system (RAG) |
| Reflection | 15 min | Production considerations |

### L3: LangGraph Workflows (2.5h)

| Phase | Duration | Focus |
|-------|----------|-------|
| Concept | 30 min | StateGraph, nodes, edges, state |
| Demo | 30 min | Research agent with routing |
| Exercise 1 | 20 min | State management and routing |
| Exercise 2 | 25 min | Human-in-the-loop workflow |
| Challenge | 30 min | Supervisor-worker system |
| Reflection | 15 min | Advanced patterns |

### L16: LangSmith Observability (2.5h)

| Phase | Duration | Focus |
|-------|----------|-------|
| Concept | 30 min | Tracing, evaluation, monitoring |
| Demo | 30 min | Instrumenting apps, viewing traces |
| Exercise 1 | 20 min | Add tracing, analyze traces |
| Exercise 2 | 25 min | Datasets and evaluators |
| Challenge | 30 min | Prompt iteration workflow |
| Reflection | 15 min | Production monitoring |

## Exercises

### LangChain Exercises

1. **Multi-Step Chain** (`src/chains/basic_chain.py`)
   - Build a chain that summarizes and translates text
   - Use LCEL pipe operator composition
   - Practice with StrOutputParser

2. **Custom Tool Agent** (`src/agents/simple_agent.py`)
   - Create a calculator tool with Pydantic schema
   - Build an agent with create_tool_calling_agent
   - Test tool invocation

### LangGraph Exercises

3. **State Routing** (`src/graphs/simple_graph.py`)
   - Define custom state with TypedDict
   - Implement conditional routing
   - Handle END condition

4. **Human-in-the-Loop** (`src/graphs/human_loop.py`)
   - Add checkpointer for persistence
   - Create interrupt points
   - Resume with human input

### LangSmith Exercises

5. **Tracing Setup** (`src/evaluation/tracing.py`)
   - Configure environment variables
   - Add metadata and tags
   - View traces in LangSmith UI

6. **Custom Evaluator** (`src/evaluation/evaluator.py`)
   - Create evaluation dataset
   - Implement custom evaluators
   - Run evaluations

## Challenges

| Challenge | Description |
|-----------|-------------|
| RAG System | Build document Q&A with memory |
| Supervisor-Worker | Multi-agent orchestration |
| Prompt Iteration | A/B test prompt versions |

## Self-Assessment

After completing this workshop, you should be able to:

- [ ] Explain when to use chains vs agents
- [ ] Create LCEL chains with pipe operator
- [ ] Build custom tools with @tool decorator
- [ ] Define LangGraph state with reducers
- [ ] Implement conditional routing
- [ ] Add human-in-the-loop checkpoints
- [ ] Design supervisor-worker patterns
- [ ] Set up LangSmith tracing
- [ ] Create evaluation datasets
- [ ] Build custom evaluators

## Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/docs/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangSmith Docs](https://docs.smith.langchain.com/)

### Tutorials
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)
- [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/)
- [DeepLearning.AI LangChain Course](https://www.deeplearning.ai/short-courses/)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)

---

*Part of the Cursor Agent Factory Learning Workshop Ecosystem*
