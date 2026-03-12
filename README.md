# Autonomous Deep-Tech R&D Engine — v3.0

**18 Agents | 6 LLMs | 2 Chains | $0 | 24/7 | Python 3.12**

---

## What It Does

Runs 4 automated research cycles per day, hunting for unsolved physical bottlenecks in AI infrastructure (thermal, power, data movement, PDN, packaging). Each cycle filters ideas through physics, novelty, market, and critics — scoring survivors with a Diamond Score.

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/rd-engine.git
cd rd-engine
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Create .env file (local dev only — never commit)
cp .env.example .env
# Edit .env with your API keys

# 3. Run physics tests first
python main.py --test-physics

# 4. Run cycles manually
python main.py --cycle 1
python main.py --cycle 2
python main.py --cycle 3
python main.py --cycle 4
```

---

## Environment Variables

```
GROQ_API_KEY=gsk_...
SAMBANOVA_API_KEY=...
FIREWORKS_API_KEY=...
MISTRAL_API_KEY=...
GEMINI_API_KEY=...
COHERE_API_KEY=...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...           <- service role key (not anon key)
GITHUB_TOKEN=ghp_...          <- github.com/settings/tokens -> classic -> public_repo
```

For GitHub Actions: add all of the above as Repository Secrets.

---

## Architecture

```
Sources (arXiv / Patents / Job Postings / OSTI / GitHub / RSS)
    |
[NEW] PhysicsLimitMapper (Agent 0)   -> maps current physical bottlenecks
    |                                   injects limits as context
Agents 1-4   RESEARCH                -> findings[], ideas[] (anchored to real limits)
    |
Agents 5-8   EXTRACTION              -> typed numerical params (Pydantic)
    |
physics.py   PHYSICS GATE            -> kill if violates Carnot/JEDEC/Landauer/PDN
    |
pgvector     NOVELTY CHECK           -> cosine similarity -> auto-kill if >0.85
    |
Agents 9-11  PHYSICS SCORING         -> physics_feasibility score
    |
Agents 12-13 MARKET                  -> market_pain, ROI, scalability
    |
Agents 14-15 CRITICS                 -> kill[], open_questions[]
    |
Agent 16     DIRECTOR (ChiefScientist) -> Diamond Score + next cycle plan
    |
[NEW] Strategy Feedback Loop         -> domain success rates -> saved to Supabase
    |                                   loaded by PhysicsLimitMapper next cycle
Supabase (state) + GitHub (archive) + Dashboard
```

---

## 18 Agents

| # | Agent | Cycle | Chain |
|---|-------|-------|-------|
| 0 | PhysicsLimitMapper | 1 | reasoning |
| 1 | PaperResearcher | 1 | research |
| 2 | PatentResearcher | 1 | research |
| 3 | InfraResearcher | 1 | research |
| 4 | IntelligenceResearcher | 1 | research |
| 5 | PDNExtractor | 2 | research |
| 6 | PowerExtractor | 2 | research |
| 7 | ThermalExtractor | 2 | research |
| 8 | DataMovementExtractor | 2 | research |
| 9 | ElectricalEngineer | 2 | reasoning |
| 10 | ThermalEngineer | 2 | reasoning |
| 11 | SystemsArchitect | 2 | reasoning |
| 12 | MarketAnalyst | 2 | research |
| 13 | CostAnalyst | 2 | research |
| 14 | CompetitionAnalyst | 3 | research |
| 15 | DevilsAdvocate | 3 | reasoning |
| 16 | ChiefScientist | 4 | reasoning |
| — | WeeklyAnalyst | weekly | reasoning |

---

## LLM Chains (Dual Chain System)

**REASONING_CHAIN** — strong models, used for physics/critics/director:
```
Groq (llama-3.3-70b) -> SambaNova (DeepSeek-V3.2) -> Fireworks (DeepSeek-V3p2)
-> Mistral (large-latest) -> Cohere (command-a-reasoning) -> Gemini (2.5-flash)
```

**RESEARCH_CHAIN** — large context, used for harvest/extraction/market:
```
Gemini (2.5-flash) -> Fireworks (DeepSeek-V3p2) -> Groq (llama-3.3-70b)
-> Mistral (large-latest) -> SambaNova (DeepSeek-V3.2) -> Cohere (command-a-reasoning)
```

Fallback rules: 429/500 -> next LLM immediately | broken JSON -> retry once same LLM | all fail -> agent skipped, cycle continues.

---

## Diamond Score

```
Diamond = (Physics x 0.35) + (Market x 0.30) + (Novelty x 0.20) + (Scalability x 0.15)
```

| Score | Status |
|-------|--------|
| >= 9.0 | diamond |
| >= 7.0 | active |
| >= 5.0 | archived |
| < 5.0 | killed |

---

## Physics Gate Domains

| Domain | Validators |
|--------|-----------|
| Thermal | Carnot, JEDEC junction temp, heat flux by material, spreading resistance |
| Electrical | Landauer limit, CMOS power density, voltage scaling |
| PDN | IR drop budget, impedance, electromigration, decap sufficiency |
| Data Movement | Roofline bandwidth wall, memory latency, interconnect distance |

---

## Daily Schedule (GitHub Actions)

| Time (UTC) | Cycle | What happens |
|------------|-------|-------------|
| 06:00 | 1 — Harvest | PhysicsLimitMapper + 4 researchers -> ideas + findings |
| 12:00 | 2 — Physics + Market | Extract -> Physics Gate -> Novelty -> Physics Scoring -> Market |
| 18:00 | 3 — Kill Round | Competition + Devil's Advocate -> kills |
| 23:00 | 4 — Director | Diamond Score + Strategy Feedback Loop saved |
| 00:00 Sun | Weekly | Cleanup + weekly report + seed.json update |

---

## Supabase Setup

Run `db/schema.sql` in your Supabase SQL Editor (one time only).
Requires: pgvector extension enabled (Supabase dashboard -> Extensions -> vector).

Use **service role key** (not anon key) — needed for full read/write access.

---

## Dashboard

Open `dashboard/index.html` directly in browser (drag to Chrome).
Enter your Supabase URL + **anon key** (read-only — different from service role key).

Filter ideas by: All / Diamond / Active / Archived / Killed.

---

## File Structure

```
rd_engine/
|-- .github/workflows/rd_engine.yml    # GitHub Actions — 4 daily cycles
|-- config/
|   |-- seed.json                      # DNA of the system — edit keywords here
|   +-- settings.py                    # Chains, weights, thresholds
|-- core/
|   |-- base_agent.py                  # Base class — CHAIN_TYPE per agent
|   |-- llm_router.py                  # Dual chain fallback router
|   +-- schemas.py                     # All Pydantic models
|-- physics/
|   |-- gate.py                        # Physics Gate orchestrator
|   |-- thermal.py                     # Carnot, JEDEC, heat flux
|   |-- electrical.py                  # Landauer, CMOS, PDN
|   +-- data_movement.py               # Roofline, memory wall
|-- agents/
|   |-- research/                      # Agents 0-4 (PhysicsLimitMapper + 4 researchers)
|   |-- extraction/                    # Agents 5-8
|   |-- physics_agents/                # Agents 9-11
|   |-- market/                        # Agents 12-13
|   |-- critics/                       # Agents 14-15
|   |-- management/                    # Agent 16
|   +-- meta/                          # WeeklyAnalyst
|-- novelty/detector.py                # Embeddings + cosine similarity
|-- db/
|   |-- supabase_client.py             # All DB ops + strategy feedback loop
|   +-- schema.sql                     # Run once in Supabase SQL editor
|-- dashboard/index.html               # Live dashboard — open in browser
|-- tests/test_physics.py              # 29 physics gate unit tests
|-- main.py                            # Entry point
+-- requirements.txt
```

---

## Testing

```bash
# Physics gate unit tests (29 tests)
python main.py --test-physics

# Weekly maintenance (manual)
python main.py --weekly

# Cold start (empty DB)
python main.py --cycle 1 --cold-start
```
