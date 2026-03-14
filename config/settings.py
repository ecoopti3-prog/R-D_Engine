import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── LLM API Keys ──────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY", "")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
MISTRAL_API_KEY   = os.environ.get("MISTRAL_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
COHERE_API_KEY    = os.environ.get("COHERE_API_KEY", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")   # free — github.com/settings/tokens
SAM_GOV_API_KEY   = os.environ.get("SAM_GOV_API_KEY", "DEMO_KEY")  # free — sam.gov (DEMO_KEY = 1000 req/day without registration)

# ── Supabase ──────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# ── LLM definitions (shared pool) ────────────────────────
_GROQ = {
    "name":               "groq",
    "base_url":           "https://api.groq.com/openai/v1",
    "api_key_env":        GROQ_API_KEY,
    "model":              "llama-3.3-70b-versatile",
    "max_tokens":         4000,
    "temperature":        0.3,
    "supports_json_mode": True,
}
_SAMBANOVA = {
    "name":               "sambanova",
    "base_url":           "https://api.sambanova.ai/v1",
    "api_key_env":        SAMBANOVA_API_KEY,
    "model":              "DeepSeek-V3.2",
    "max_tokens":         4000,
    "temperature":        0.3,
    "supports_json_mode": False,   # SambaNova compat layer ignores response_format
}
_FIREWORKS = {
    "name":               "fireworks",
    "base_url":           "https://api.fireworks.ai/inference/v1",
    "api_key_env":        FIREWORKS_API_KEY,
    "model":              "accounts/fireworks/models/deepseek-v3p2",
    "max_tokens":         4000,
    "temperature":        0.3,
    "supports_json_mode": True,
}
_MISTRAL = {
    "name":               "mistral",
    "base_url":           "https://api.mistral.ai/v1",
    "api_key_env":        MISTRAL_API_KEY,
    "model":              "mistral-large-latest",
    "max_tokens":         4000,
    "temperature":        0.3,
    "supports_json_mode": True,
}
_GEMINI = {
    "name":               "gemini",
    "base_url":           "https://generativelanguage.googleapis.com/v1beta/openai",
    "api_key_env":        GEMINI_API_KEY,
    "model":              "gemini-2.5-flash",
    "max_tokens":         4000,
    "temperature":        0.3,
    "supports_json_mode": False,  # Gemini OpenAI-compat ignores response_format and returns ~15 tokens
                                  # Use strict system prompt approach instead (handled in llm_router)
}
_COHERE = {
    "name":               "cohere",
    "base_url":           "https://api.cohere.com/compatibility/v1",
    "api_key_env":        COHERE_API_KEY,
    "model":              "command-a-reasoning-08-2025",
    "max_tokens":         4000,
    "temperature":        0.3,
    "supports_json_mode": False,   # Cohere compat endpoint doesn't honour response_format
}

# ── REASONING chain ───────────────────────────────────────
# Used by: Physics agents, Critics, Devil's Advocate, Chief Scientist
# Priority: strong reasoning models first, preserve Groq quota for these
REASONING_CHAIN = [_GROQ, _SAMBANOVA, _FIREWORKS, _MISTRAL, _COHERE, _GEMINI]

# ── RESEARCH chain ────────────────────────────────────────
# Used by: Harvest (Cycle 1), Extractors, Market/Cost/Competition analysts
# Priority: Gemini Flash FIRST — 1M token context window is decisive for
# reading many papers/patents at once. Groq second (fastest fallback).
RESEARCH_CHAIN = [_GEMINI, _GROQ, _SAMBANOVA, _FIREWORKS, _MISTRAL, _COHERE]

# ── Legacy alias (backward compat — not used internally) ──
LLM_CHAIN = REASONING_CHAIN

# ── Runtime ───────────────────────────────────────────────
AGENT_DELAY_SECONDS    = 3
GEMINI_RPM_DELAY       = 5   # extra sleep after each Gemini call (15 RPM limit on free tier)
HEARTBEAT_INTERVAL_SEC = 180
HEARTBEAT_TIMEOUT_SEC  = 600
RETRY_TIMEOUT_SEC      = 60
MAX_TOKENS             = 4000

# ── Physics Gate ──────────────────────────────────────────
PHYSICS_KILL_THRESHOLD = 0
PHYSICS_GATE_WEIGHT    = 0.40   # gate score weight in blend with agent scores
PHYSICS_AGENT_WEIGHT   = 0.60   # agent score weight in blend with gate score

# ── HypothesisGenerator guard ─────────────────────────────
# Skip HypothesisGenerator if fewer than this many killed ideas exist.
# Avoids wasting a reasoning LLM call on empty/near-empty kill history.
MIN_KILLS_FOR_HYPOTHESIS = 10

# ── Novelty Detection ─────────────────────────────────────
NOVELTY_AUTO_KILL_THRESHOLD    = 0.85
NOVELTY_SIMILAR_FLAG_THRESHOLD = 0.75
EMBEDDING_MODEL                = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM                  = 384

# ── Diamond Score weights ─────────────────────────────────
DIAMOND_WEIGHTS = {
    "physics":     0.35,
    "market":      0.30,
    "novelty":     0.20,
    "scalability": 0.15,
}

# ── Score thresholds ──────────────────────────────────────
SCORE_KILL     = 3.0
SCORE_ARCHIVE  = 5.0
SCORE_ACTIVE   = 7.0
SCORE_PRIORITY = 9.0
