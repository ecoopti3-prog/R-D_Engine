# Contributing to RD Engine

## How to Add a New Agent

Every agent inherits from `BaseAgent`. You need to implement 3 things:

```python
from core.base_agent import BaseAgent
from core.schemas import AgentOutput

class MyAgent(BaseAgent):
    AGENT_NAME    = "my_agent"
    CHAIN_TYPE    = "research"   # "research" or "reasoning"
    SYSTEM_PROMPT = """..."""    # LLM persona + output schema

    def build_user_prompt(self, context: dict) -> str:
        # Build the user message from context dict
        return f"Analyze: {context.get('ideas', [])}"

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        # Parse LLM JSON response into AgentOutput
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            ideas=[],
        )
```

- Use `CHAIN_TYPE = "reasoning"` for physics critics and scoring agents (Groq/SambaNova first)
- Use `CHAIN_TYPE = "research"` for harvest and extraction agents (higher throughput)
- Always end SYSTEM_PROMPT with `OUTPUT FORMAT: Valid JSON only. No markdown.`
- For batch idea processing: call `agent.run_chunked(ideas, cycle_id)` — it auto-chunks by 5

## Running Locally Without Supabase

Set `SUPABASE_URL` and `SUPABASE_KEY` to empty strings and the DB calls will fail gracefully
(logged as errors, but the cycle continues). Useful for testing individual agents:

```python
from agents.research.paper_researcher import PaperResearcher
import uuid

agent = PaperResearcher()
context = {"keywords": ["GPU thermal bottleneck"], "papers": [], "date": "2025-01-01"}
output = agent.run(context, cycle_id=str(uuid.uuid4()))
print(output.findings)
print(output.ideas)
```

## Running Physics Tests

```bash
python main.py --test-physics
```

All 29 tests must pass before any PR.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add at least one LLM key (GROQ_API_KEY is free and fast)
```

## Project Structure

```
core/           — schemas, base_agent, llm_router (don't change without tests)
agents/
  research/     — Cycle 1: harvest findings and ideas (Agents 0-4)
  extraction/   — Cycle 2: extract numerical params (Agents 5-8)
  physics_agents/ — Cycle 2: score physics feasibility (Agents 9-11)
  market/       — Cycle 2: score market pain + scalability (Agents 12-13)
  critics/      — Cycle 3: kill weak ideas (Agents 14-15)
  management/   — Cycle 4: final scoring + next cycle plan (Agent 16)
  meta/         — Weekly: cleanup + report (Agent 17)
physics/        — Deterministic validators (no LLM, no changes without proof)
novelty/        — pgvector cosine similarity (no LLM)
db/             — All Supabase operations
```
