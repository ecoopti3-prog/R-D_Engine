"""
check_keys.py — בדיקת כל המפתחות לפני הרצה
הרץ: python check_keys.py
"""
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = []
failed = []
warnings = []


def ok(name, detail=""):
    passed.append(name)
    print(f"  {GREEN}✓{RESET}  {name}" + (f"  {YELLOW}({detail}){RESET}" if detail else ""))


def fail(name, detail=""):
    failed.append(name)
    print(f"  {RED}✗{RESET}  {name}" + (f"  — {detail}" if detail else ""))


def warn(name, detail=""):
    warnings.append(name)
    print(f"  {YELLOW}⚠{RESET}  {name}" + (f"  — {detail}" if detail else ""))


# ─────────────────────────────────────────────
# 1. SUPABASE
# ─────────────────────────────────────────────
print(f"\n{BOLD}[ Supabase ]{RESET}")
supa_url = os.environ.get("SUPABASE_URL", "")
supa_key = os.environ.get("SUPABASE_KEY", "")

if not supa_url:
    fail("SUPABASE_URL", "missing")
elif not supa_url.startswith("https://"):
    fail("SUPABASE_URL", f"must start with https://  got: {supa_url[:30]}")
else:
    ok("SUPABASE_URL", supa_url[:40])

if not supa_key:
    fail("SUPABASE_KEY", "missing")
elif len(supa_key) < 100:
    fail("SUPABASE_KEY", "too short — use service role key, not anon key")
else:
    ok("SUPABASE_KEY", f"length={len(supa_key)}")

if supa_url and supa_key:
    try:
        r = requests.get(
            f"{supa_url}/rest/v1/research_cycles?select=id&limit=1",
            headers={"apikey": supa_key, "Authorization": f"Bearer {supa_key}"},
            timeout=8,
        )
        if r.status_code == 200:
            ok("Supabase connection", f"HTTP {r.status_code}")
        elif r.status_code == 401:
            fail("Supabase connection", "401 Unauthorized — wrong key?")
        else:
            fail("Supabase connection", f"HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:
        fail("Supabase connection", str(e)[:80])


# ─────────────────────────────────────────────
# 2. LLM KEYS
# ─────────────────────────────────────────────
print(f"\n{BOLD}[ LLM API Keys ]{RESET}")

def test_llm(name, url, key, model, headers_extra=None):
    if not key:
        fail(f"{name} key", "missing from .env")
        return
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if headers_extra:
        headers.update(headers_extra)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 5,
    }
    try:
        r = requests.post(f"{url}/chat/completions", json=body, headers=headers, timeout=15)
        if r.status_code == 200:
            ok(name, f"model={model}")
        elif r.status_code == 401:
            fail(name, "401 Unauthorized — invalid key")
        elif r.status_code == 429:
            warn(name, "429 Rate limit — key is valid but throttled")
        elif r.status_code == 404:
            fail(name, f"404 — model '{model}' not found on this account")
        else:
            fail(name, f"HTTP {r.status_code}: {r.text[:100]}")
    except requests.exceptions.Timeout:
        warn(name, "timeout — endpoint slow, key may still be valid")
    except Exception as e:
        fail(name, str(e)[:80])

# Groq
test_llm(
    "Groq",
    "https://api.groq.com/openai/v1",
    os.environ.get("GROQ_API_KEY", ""),
    "llama-3.3-70b-versatile",
)
time.sleep(0.5)

# Gemini
test_llm(
    "Gemini",
    "https://generativelanguage.googleapis.com/v1beta/openai",
    os.environ.get("GEMINI_API_KEY", ""),
    "gemini-2.5-flash",
)
time.sleep(0.5)

# Mistral
test_llm(
    "Mistral",
    "https://api.mistral.ai/v1",
    os.environ.get("MISTRAL_API_KEY", ""),
    "mistral-large-latest",
)
time.sleep(0.5)

# SambaNova
test_llm(
    "SambaNova",
    "https://api.sambanova.ai/v1",
    os.environ.get("SAMBANOVA_API_KEY", ""),
    "DeepSeek-V3.2",
)
time.sleep(0.5)

# Cerebras — qwen-3-235b is the best model available on this account (235B, 65536 context)
test_llm(
    "Cerebras",
    "https://api.cerebras.ai/v1",
    os.environ.get("CEREBRAS_API_KEY", ""),
    "qwen-3-235b-a22b-instruct-2507",
)
time.sleep(0.5)

# Cohere
cohere_key = os.environ.get("COHERE_API_KEY", "")
test_llm(
    "Cohere",
    "https://api.cohere.com/compatibility/v1",
    cohere_key,
    "command-a-reasoning-08-2025",
)


# ─────────────────────────────────────────────
# 3. OPTIONAL KEYS
# ─────────────────────────────────────────────
print(f"\n{BOLD}[ Optional Keys ]{RESET}")

# GitHub Token
gh_token = os.environ.get("GITHUB_TOKEN", "")
if not gh_token:
    warn("GITHUB_TOKEN", "missing — summaries won't be pushed to git")
else:
    try:
        r = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {gh_token}"},
            timeout=8,
        )
        if r.status_code == 200:
            ok("GITHUB_TOKEN", f"user={r.json().get('login','?')}")
        else:
            fail("GITHUB_TOKEN", f"HTTP {r.status_code}")
    except Exception as e:
        fail("GITHUB_TOKEN", str(e)[:60])

# HuggingFace Token
hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    warn("HF_TOKEN", "missing — unauthenticated HF requests (rate limited, but works)")
else:
    try:
        r = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {hf_token}"},
            timeout=8,
        )
        if r.status_code == 200:
            ok("HF_TOKEN", f"user={r.json().get('name','?')}")
        else:
            fail("HF_TOKEN", f"HTTP {r.status_code} — invalid token")
    except Exception as e:
        fail("HF_TOKEN", str(e)[:60])

# PatentsView — removed (no key in use)

# OpenAlex email (polite pool)
openalex_email = os.environ.get("OPENALEX_EMAIL", "")
if not openalex_email:
    warn("OPENALEX_EMAIL", "missing — limited to 1 req/s (polite pool: 10 req/s, 100k/day)")
else:
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={
                "search":   "thermal management GPU",
                "per-page": "1",
                "mailto":   openalex_email,
            },
            headers={"User-Agent": f"rd-engine-check/1.0 (mailto:{openalex_email})"},
            timeout=10,
        )
        if r.status_code == 200:
            meta  = r.json().get("meta", {})
            count = meta.get("count", "?")
            ok("OPENALEX_EMAIL", f"polite pool active — {count} works indexed for test query")
        elif r.status_code == 403:
            fail("OPENALEX_EMAIL", "403 — email may be blocked or malformed")
        else:
            warn("OPENALEX_EMAIL", f"HTTP {r.status_code} — email set but unexpected response")
    except Exception as e:
        warn("OPENALEX_EMAIL", f"timeout or error: {str(e)[:60]}")

# NASA Tech Transfer — endpoint changed, no API key required
# New endpoint: https://technology.nasa.gov/api/api/patent/{keyword}
print("[NASA] Checking technology.nasa.gov endpoint...")
try:
    r = requests.get(
        "https://technology.nasa.gov/api/api/patent/thermal",
        headers={"User-Agent": "rd-engine-check/1.0", "Accept": "application/json"},
        timeout=15,
    )
    if r.status_code == 200:
        body = r.text.strip()
        if not body:
            warn("NASA_API_KEY", "200 OK but empty body")
        else:
            try:
                count = len(r.json().get("results", []))
                ok("NASA (technology.nasa.gov)", f"live — {count} patents for 'thermal'")
                if os.environ.get("NASA_API_KEY"):
                    warn("NASA_API_KEY", "key set but not needed — new endpoint is public (no key required)")
            except Exception:
                warn("NASA (technology.nasa.gov)", f"200 OK but invalid JSON: {body[:60]}")
    else:
        fail("NASA (technology.nasa.gov)", f"HTTP {r.status_code}: {r.text[:80]}")
except Exception as e:
    warn("NASA (technology.nasa.gov)", f"timeout or error: {str(e)[:60]}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"{BOLD}RESULTS{RESET}")
print(f"  {GREEN}✓ Passed:  {len(passed)}{RESET}")
if warnings:
    print(f"  {YELLOW}⚠ Warnings: {len(warnings)}{RESET}")
if failed:
    print(f"  {RED}✗ Failed:  {len(failed)}{RESET}")
    print(f"\n{RED}Fix these before running:{RESET}")
    for f in failed:
        print(f"  • {f}")
    print(f"\n{BOLD}Where to get missing keys:{RESET}")
    key_urls = {
        "Groq":           "https://console.groq.com/keys",
        "Gemini":         "https://aistudio.google.com/app/apikey",
        "Mistral":        "https://console.mistral.ai/api-keys",
        "SambaNova":      "https://cloud.sambanova.ai/apis",
        "Cohere":         "https://dashboard.cohere.com/api-keys",
        "GITHUB_TOKEN":   "https://github.com/settings/tokens → classic → public_repo",
        "HF_TOKEN":       "https://huggingface.co/settings/tokens",
        "OPENALEX_EMAIL": "https://openalex.org — set any valid email, no registration needed",
        "NASA_API_KEY":   "https://api.nasa.gov — click 'Generate API Key', free, instant",
    }
    for f in failed:
        for k, url in key_urls.items():
            if k.lower() in f.lower():
                print(f"  • {f}: {url}")
                break
    sys.exit(1)
else:
    print(f"\n{GREEN}{BOLD}All required keys OK — ready to run!{RESET}")
    sys.exit(0)
