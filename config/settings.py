import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── LLM API Keys ──────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY", "")
CEREBRAS_API_KEY  = os.environ.get("CEREBRAS_API_KEY", "")
MODELS_TOKEN      = os.environ.get("MODELS_TOKEN", "")       # GitHub Models — models.inference.ai.azure.com
MISTRAL_API_KEY   = os.environ.get("MISTRAL_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
COHERE_API_KEY    = os.environ.get("COHERE_API_KEY", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")       # free — github.com/settings/tokens
LENS_API_KEY      = os.environ.get("LENS_API_KEY", "")       # free — lens.org → user → subscriptions → Scholar API
# NASA Tech Transfer: public endpoint, no key needed — technology.nasa.gov/api/api/patent/{keyword}
OPENALEX_EMAIL    = os.environ.get("OPENALEX_EMAIL", "")     # free — openalex.org polite pool (10 req/s vs 1 req/s without)
HF_TOKEN          = os.environ.get("HF_TOKEN", "")           # free — huggingface.co/settings/tokens

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
    "supports_json_mode": False,
}
_SAMBANOVA = {
    "name":               "sambanova",
    "base_url":           "https://api.sambanova.ai/v1",
    "api_key_env":        SAMBANOVA_API_KEY,
    "model":              "DeepSeek-V3.2",
    "max_tokens":         4000,   # FIX: lowered from 7000 — total context limit is 8192, input alone exceeds it at 7000 output
    "temperature":        0.3,
    "supports_json_mode": False,  # SambaNova compat layer ignores response_format
}
_CEREBRAS = {
    "name":               "cerebras",
    "base_url":           "https://api.cerebras.ai/v1",
    "api_key_env":        CEREBRAS_API_KEY,
    "model":              "qwen-3-235b-a22b-instruct-2507",  # 235B, 65536 context — best available on this account
    "max_tokens":         4000,
    "temperature":        0.3,
    "supports_json_mode": True,
}
_GITHUB_MODELS = {
    "name":               "github-models",
    "base_url":           "https://models.inference.ai.azure.com",
    "api_key_env":        MODELS_TOKEN,
    "model":              "Llama-3.3-70B-Instruct",
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
    "supports_json_mode": False,  # Gemini OpenAI-compat ignores response_format → returns ~15 tokens
}
_COHERE = {
    "name":               "cohere",
    "base_url":           "https://api.cohere.com/compatibility/v1",
    "api_key_env":        COHERE_API_KEY,
    "model":              "command-a-reasoning-08-2025",
    "max_tokens":         4000,
    "temperature":        0.3,
    "supports_json_mode": False,
}

# ── REASONING chain ───────────────────────────────────────
# Cerebras first — qwen-3-235b has 65536 context, handles long reasoning prompts
# Groq second — fastest fallback for short prompts
# Sambanova moved to last — 8192 context limit causes constant 400 errors
REASONING_CHAIN = [_CEREBRAS, _GROQ, _MISTRAL, _COHERE, _SAMBANOVA, _GEMINI]

# ── RESEARCH chain ────────────────────────────────────────
# Cerebras first — 65536 context solves the "context too long" problem that was
# causing Sambanova to fail on every research agent call
# Sambanova moved to last — its 8192 limit is too small for research prompts
RESEARCH_CHAIN = [_CEREBRAS, _GROQ, _MISTRAL, _COHERE, _SAMBANOVA, _GEMINI]

# ── Legacy alias (backward compat — not used internally) ──
LLM_CHAIN = REASONING_CHAIN

# ── Runtime ───────────────────────────────────────────────
AGENT_DELAY_SECONDS    = 6
GEMINI_RPM_DELAY       = 7   # extra sleep after each Gemini call (15 RPM limit on free tier)
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
MIN_KILLS_FOR_HYPOTHESIS = 3

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
