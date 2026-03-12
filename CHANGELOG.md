# RD Engine — v6 Changelog

## What Changed (v5 → v6) — Production Hardening

All changes are backward compatible. No tables removed. No agent behavior changed.
v6 fixes bugs that would have silently degraded output quality from day 1.

---

### Fix 1 — IVFFlat → HNSW (CRITICAL — cold start blocker)
**File:** `db/schema.sql`

IVFFlat requires `lists × 3 = 150` rows minimum before the index builds correctly.
On an empty database, all novelty detection queries were broken or returned wrong results.

HNSW replaces it. Works from 0 rows. Faster queries. No minimum data requirement.
Parameters: `m=16` (connectivity), `ef_construction=64` (build quality) — balanced defaults.

---

### Fix 2 — RLS anon-key read policies for dashboard (CRITICAL)
**File:** `db/schema.sql`

The dashboard uses the anon key (browser-visible). The schema only had `service_role_all`
policies. The dashboard had **no read access at all** via anon key. Added:
- SELECT policies on ideas, findings, daily_summaries, hypotheses, weekly_reports
- INSERT policy on idea_feedback (allows 1-5 star ratings from dashboard UI)

---

### Fix 3 — RESEARCH_CHAIN order corrected
**File:** `config/settings.py`

The comment said "Gemini Flash first — huge context window". The code had Groq first.
Fixed: RESEARCH_CHAIN now starts with Gemini (1M token context), then Groq as fallback.
REASONING_CHAIN unchanged (Groq first — correct for physics/critic reasoning).

---

### Fix 4 — strategy seed fallback on day 1
**File:** `db/supabase_client.py`

`load_search_strategy()` returned `{}` on empty DB.
PhysicsLimitMapper received empty context on first run — "blind" first cycle.
Fixed: falls back to `seed.json` when no DB history exists.
PhysicsLimitMapper now gets `priority_domains`, `new_keywords`, `target_companies` from seed.

---

### Fix 5 — Dead code removed from supabase_client.py
**File:** `db/supabase_client.py`

Unreachable code existed after `return counts` in `cleanup_weak_ideas()`.
This was an orphaned first draft of `save_daily_summary` that was never deleted.
Removed. `save_daily_summary` is defined correctly later in the file.

---

### Fix 6 — save_idea now accepts source_url and populates source_venue
**File:** `db/supabase_client.py`

The `source_venue` column existed in the schema since v5 but was never written.
`save_idea()` now accepts an optional `source_url` parameter and denormalizes
the venue into `source_venue`. This makes `source_performance_view` usable from day 1.

---

### Fix 7 — UUID matching replaced with O(1) dict lookup
**File:** `main.py`

All extraction/scoring loops used `startswith` prefix matching across all ideas.
With 50+ ideas, two UUIDs starting with the same 8 chars could silently cross-assign scores.
Fixed: pre-built `ideas_by_id` dicts everywhere. Exact match first, prefix fallback only
when there is exactly ONE candidate (unambiguous). Applied to:
- Extraction loop (PDN/Power/Thermal/DataMovement extractors)
- Physics agent scoring (ElectricalEngineer/ThermalEngineer/SystemsArchitect)
- MarketAnalyst scoring
- CostAnalyst scoring

---

### Fix 8 — Diamond Score blending corrected
**File:** `main.py`

The previous blend `existing * 0.4 + agent_score * 0.6` had two bugs:
1. `score_map` excluded `score=0` entries (filter `if score > 0`), so intentionally
   bad scores from agents were silently ignored — those ideas kept their gate score.
2. A gate score of 8 with no agent match produced `8 * 0.4 + 0 * 0.6 = 3.2` under
   certain code paths.

Fixed:
- `score_map` now includes all scored ideas including `score=0`
- Only blend when `existing > 0` AND `new_score is not None`
- If no gate score yet, use agent score directly
- Uses named constants `PHYSICS_GATE_WEIGHT=0.40`, `PHYSICS_AGENT_WEIGHT=0.60`

---

### Fix 9 — Lineage saving activated
**File:** `main.py`, `db/lineage.py`

`save_idea_lineage()` existed in `db/lineage.py` since v5 but was never called.
The entire source-diamond yield learning loop was broken from day 1.

Fixed: Cycle 1 now calls `save_idea_lineage()` after each idea is saved,
linking ideas to their source URLs (papers, patents) from the cycle context.
After 2+ weeks this enables `get_top_yielding_sources()` to return meaningful data.

---

### Fix 10 — HypothesisGenerator guard on empty kill history
**File:** `main.py`, `config/settings.py`

HypothesisGenerator was called on day 1 with 0 kills. It wasted an expensive reasoning
LLM call and returned generic placeholder hypotheses with no analytical value.

Fixed: Skip if fewer than `MIN_KILLS_FOR_HYPOTHESIS=10` killed ideas exist.
Constant is configurable in `config/settings.py`. Activates automatically once enough
kill history accumulates.

---

### Fix 11 — Weekly maintenance handles first-week empty DB
**File:** `main.py`

`run_weekly_maintenance()` returned early with "No week stats available" on week 1,
silently skipping the seed keyword update that primes week 2.

Fixed: On empty/sparse DB, still attempts to update `seed.json` from latest strategy,
so week 2 starts with enriched keywords even when week 1 had minimal data.

---

### Fix 12 — GitHub Actions: sentence-transformers model caching
**File:** `.github/workflows/rd_engine.yml`

The `all-MiniLM-L6-v2` model (~90MB) was re-downloaded on every cycle run.
This added 30-60 seconds per cycle and risked timeout on slow Actions runners.

Fixed: Added `actions/cache@v4` step caching `~/.cache/huggingface/hub`.
Cache key is pinned to model name — change the key suffix to force a re-download.

---

### Fix 13 — GITHUB_TOKEN warning is now explicit
**File:** `utils/sources.py`

Missing `GITHUB_TOKEN` caused GitHub Issues and OCP signals to silently return `[]`.
No log message indicated why. Engineers debugging empty harvests had no clue.

Fixed: Explicit `WARNING` log when token is absent, listing exactly which sources
will be empty and how to fix it.

---

## Summary of Learning Loop Status After v6

| Learning Mechanism | Status |
|-------------------|--------|
| Strategy feedback (domain rates → priority_domains) | ✅ Working from day 1 |
| Source lineage (which venue → diamonds) | ✅ **Fixed** — now saves from day 1 |
| HypothesisGenerator (kill patterns → assumptions) | ✅ **Fixed** — activates at 10 kills |
| Human feedback (1-5 stars → score adjustment) | ✅ Working (dashboard writes now allowed) |
| Weekly keyword evolution | ✅ **Fixed** — works on week 1 |



## What Changed (v4 → v5)

### New Agents

#### `agents/synthesis/cross_domain_synthesizer.py` — Agent 5 (NEW)
Runs at the end of Cycle 1, after all 4 researchers finish.
Looks at ALL findings across domains and finds couplings: "what happens when
thermal AND PDN constraints are active simultaneously during LLM prefill?"
Generates 3-5 cross-domain ideas that no single-domain researcher would produce.
Also produces a `coupling_map` (saved to Supabase + strategy) used by ChiefScientist.
→ This is the closest the system gets to synthesis vs retrieval.

#### `agents/meta/hypothesis_generator.py` — Meta Agent (NEW)
Runs in Cycle 4, after ChiefScientist.
Reads kill history and asks: are there ASSUMED CONSTRAINTS killing ideas, not real physics limits?
Example: 100 ideas killed for "junction temp too high" → hypothesis: "the 105°C JEDEC limit
is an engineering assumption, not Carnot. What if direct bonding relaxes it?"
Saves hypotheses to Supabase → injected into next Cycle 1's PhysicsLimitMapper.
→ This is the system challenging its own assumptions.

---

### New DB Modules

#### `db/lineage.py` (NEW)
Tracks which source URL / finding generated each idea.
After 2+ weeks: `get_source_diamond_yield()` ranks venues by diamond yield.
Top-yielding venues are injected into Cycle 1 researcher context.
→ The system learns WHICH sources to trust.

#### `db/feedback.py` (NEW)
Human-in-the-loop rating system (1-5 stars per idea).
`apply_feedback_to_scores()` blends human ratings into ChiefScientist final scores.
Rating=1 forces kill. Rating=5 adds 15% boost toward 10.0.
→ The strongest learning signal available — human expert judgment.

---

### Modified Files

#### `agents/research/physics_limit_mapper.py`
- Added `limit_type: "physics" | "engineering"` to every mapped limit
- Added `assumption_if_engineering` field — what assumption makes an engineering limit look immovable
- Added `engineering_limits_worth_challenging` output section
- Now receives `open_hypotheses` from DB — previous cycle's hypotheses guide what assumptions to challenge
- System prompt updated: distinguishes Carnot/Landauer (immovable) from JEDEC/manufacturing specs (challengeable)

#### `agents/management/chief_scientist.py`
- Now receives `feedback_signals` (human ratings) and integrates them into scoring
- Receives `top_yielding_sources` — venues that produced diamonds get novelty bonus
- Receives `coupling_map` from CrossDomainSynthesizer
- Receives `engineering_limits_worth_challenging` from PhysicsLimitMapper

#### `db/supabase_client.py`
Added functions:
- `load_killed_ideas_full()` — full kill data for HypothesisGenerator
- `save_hypotheses()` — persist hypotheses to DB
- `load_open_hypotheses()` — load hypotheses for injection into Cycle 1
- `save_weekly_report()` — save to weekly_reports table
- `get_week_stats()` — aggregate weekly statistics
- `cleanup_weak_ideas()` — archive/delete weak ideas (was missing from original)

#### `db/schema.sql`
Added tables:
- `idea_feedback` — human 1-5 star ratings
- `idea_lineage` — source URL → idea tracking
- `hypotheses` — HypothesisGenerator output
- `weekly_reports` — weekly analyst reports

Added columns to `ideas`:
- `source_venue` — denormalized venue for fast queries
- `physics_kill_detail` — detailed kill reason from sim engine
- `human_rating` — denormalized human rating for fast sorting
- `hypothesis_id` — which hypothesis this idea was generated to test

Added PostgreSQL view: `source_performance_view`
Added PostgreSQL function: `_extract_venue_sql()`

#### `main.py` — Cycle 1 (run_cycle_1_harvest)
- Loads `open_hypotheses` from DB → injects into PhysicsLimitMapper context
- Loads `top_yielding_sources` from lineage tracking
- Runs `CrossDomainSynthesizer` after all 4 researchers
- Saves `coupling_map` + `engineering_limits_worth_challenging` to strategy
- Injects `engineering_limits` and `open_hypotheses` into all researcher contexts

#### `main.py` — Cycle 4 (run_cycle_4_director)
- Loads `feedback_signals` from `idea_feedback` table
- Applies feedback adjustments BEFORE writing final scores
- Runs `HypothesisGenerator` after ChiefScientist
- Saves hypotheses to DB
- High-priority hypothesis searches added to next-cycle keywords

#### `dashboard/index.html`
- Added "🧪 Hypotheses" tab — shows open hypotheses with priority, challenge, and testability
- Added "📡 Sources" tab — shows source performance by diamond yield (meaningful after week 2)
- Added human feedback rating UI in detail panel (1-5 stars + reason textarea)
- Ratings POST directly to Supabase `idea_feedback` table via REST API
- `human_rating` shown in idea detail panel when available

---

## Learning Architecture (v5)

```
Week 1:
  Cycle 4 → HypothesisGenerator reads kills → saves hypotheses
  ↓
Week 2 Cycle 1:
  PhysicsLimitMapper reads hypotheses → focuses on engineering limits
  CrossDomainSynthesizer finds couplings → saves coupling_map
  Lineage tracks which sources → will rank after enough data
  ↓
Week 2 Cycle 4:
  ChiefScientist receives: feedback + source yields + coupling map + eng limits
  HypothesisGenerator compares this week's kills vs last week → new hypotheses
  ↓
Month 1:
  Source yield data meaningful → system prioritizes high-yield venues
  Hypothesis database grows → system challenges more assumptions per cycle
  Feedback calibration → FEEDBACK_WEIGHT could be auto-adjusted if accuracy tracked
```

## Backward Compatibility
- All original agents untouched
- No schema columns removed (only added)
- New tables created with IF NOT EXISTS
- All new features fail silently (try/except, non-blocking)
- ZIP runs identically to v4 if new tables don't exist (graceful fallback)
