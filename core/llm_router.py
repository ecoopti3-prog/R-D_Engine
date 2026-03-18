"""
llm_router.py — 6-model fallback chain with Exponential Backoff.

Priority order: Groq → SambaNova → Fireworks → Mistral → Gemini → Cohere
Rules:
  429 / 500          → next LLM immediately
  Timeout            → retry once same LLM → then next LLM
  Broken JSON (200)  → retry with fix instruction same LLM → then next LLM
  Context too long   → compress input → retry same LLM
  All 6 fail         → raise AllLLMsFailedError
"""
from __future__ import annotations
import json, time, logging
from typing import Optional
from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError
from config.settings import REASONING_CHAIN, RESEARCH_CHAIN, MAX_TOKENS, GEMINI_RPM_DELAY

logger = logging.getLogger(__name__)

# Track suspended accounts (412) — skip for entire process lifetime
_suspended_llms: set = set()


class AllLLMsFailedError(Exception):
    pass

class JSONParseError(Exception):
    pass


def _client(llm_cfg: dict) -> OpenAI:
    return OpenAI(
        api_key=llm_cfg["api_key_env"],
        base_url=llm_cfg["base_url"],
        timeout=60.0,
    )


def _call_single(llm_cfg: dict, system_prompt: str, user_prompt: str,
                 force_json: bool = False) -> tuple[str, int, int, int]:
    """One LLM call. Returns (raw_text, tokens_in, tokens_out, total_tokens). Raises on error."""
    client = _client(llm_cfg)

    kwargs = dict(
        model=llm_cfg["model"],
        max_tokens=llm_cfg.get("max_tokens", MAX_TOKENS),
        temperature=llm_cfg.get("temperature", 0.3),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    # Native JSON mode — supported by Groq, Gemini (via openai-compat), Mistral, Fireworks.
    # Forces the model to emit valid JSON without Markdown wrapping.
    # SambaNova and Cohere may not honour this; they fall back gracefully via _parse_json.
    if force_json and llm_cfg.get("supports_json_mode", True):
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    tokens_in  = resp.usage.prompt_tokens if resp.usage else 0
    tokens_out = resp.usage.completion_tokens if resp.usage else 0
    total      = resp.usage.total_tokens if resp.usage else 0
    return text, tokens_in, tokens_out, total


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("```", 2)[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.rsplit("```", 1)[0]
    return json.loads(clean.strip())


def call_llm(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    compress_if_long: bool = True,
    chain: str = "reasoning",
) -> tuple:
    """
    Try each LLM in the chain until one succeeds.
    Returns (parsed_output, llm_name_used, tokens_in, tokens_out, total_tokens).
    Raises AllLLMsFailedError if all fail.
    """
    active_chain = RESEARCH_CHAIN if chain == "research" else REASONING_CHAIN
    last_error = None

    for llm_cfg in active_chain:
        name = llm_cfg["name"]
        if not llm_cfg.get("api_key_env"):
            logger.debug(f"[Router] Skipping {name} — no API key")
            continue
        if name in _suspended_llms:
            logger.debug(f"[Router] Skipping {name} — account suspended (412)")
            continue

        logger.info(f"[Router] Trying {name}")

        try:
            text, tok_in, tok_out, total = _call_single(
                llm_cfg, system_prompt, user_prompt,
                force_json=expect_json,
            )
        except RateLimitError:
            logger.warning(f"[Router] {name} → 429 Rate Limit — next LLM")
            continue
        except APIStatusError as e:
            if e.status_code == 412:
                # 412 = account suspended (spending limit / unpaid invoice)
                # Skip permanently for this session — retrying wastes time
                logger.warning(f"[Router] {name} → 412 Account Suspended — skipping permanently")
                _suspended_llms.add(name)
                continue
            if e.status_code == 400 and compress_if_long:
                logger.warning(f"[Router] {name} → 400 context too long — compressing")
                user_prompt = user_prompt[:int(len(user_prompt) * 0.6)] + "\n[TRUNCATED]"
                try:
                    text, tok_in, tok_out, total = _call_single(
                        llm_cfg, system_prompt, user_prompt,
                        force_json=expect_json,
                    )
                except Exception as inner:
                    logger.warning(f"[Router] {name} failed after compress: {inner}")
                    continue
            elif e.status_code >= 500:
                logger.warning(f"[Router] {name} → {e.status_code} server error — next LLM")
                continue
            else:
                logger.warning(f"[Router] {name} → API error {e.status_code}: {e} — next LLM")
                continue
        except APITimeoutError:
            logger.warning(f"[Router] {name} → timeout — retrying once")
            time.sleep(5)
            try:
                text, tok_in, tok_out, total = _call_single(
                    llm_cfg, system_prompt, user_prompt,
                    force_json=expect_json,
                )
            except Exception:
                logger.warning(f"[Router] {name} → timeout on retry — next LLM")
                continue
        except Exception as e:
            logger.warning(f"[Router] {name} → unexpected error: {e} — next LLM")
            last_error = e
            continue

        if expect_json:
            # Reject suspiciously thin responses — a real agent output needs substance
            if tok_out < 50:
                logger.warning(f"[Router] {name} returned only {tok_out} output tokens — likely empty response, next LLM")
                continue
            try:
                parsed = _parse_json(text)
                logger.info(f"[Router] {name} OK ({total} tokens in={tok_in} out={tok_out})")
                if name == "gemini":
                    time.sleep(GEMINI_RPM_DELAY)
                return parsed, name, tok_in, tok_out, total
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"[Router] {name} returned invalid JSON — retrying")
                strict_system = (
                    system_prompt
                    + "\n\nCRITICAL: Return ONLY valid JSON. No markdown, no backticks. Start with {{ end with }}."
                )
                try:
                    text2, tok_in2, tok_out2, total2 = _call_single(
                        llm_cfg, strict_system, user_prompt,
                        force_json=True,
                    )
                    parsed = _parse_json(text2)
                    logger.info(f"[Router] {name} OK after JSON fix ({total2} tokens)")
                    if name == "gemini":
                        time.sleep(GEMINI_RPM_DELAY)
                    return parsed, name, tok_in2, tok_out2, total2
                except Exception:
                    logger.warning(f"[Router] {name} still invalid JSON — next LLM")
                    continue
        else:
            logger.info(f"[Router] {name} OK ({total} tokens)")
            if name == "gemini":
                time.sleep(GEMINI_RPM_DELAY)
            return text, name, tok_in, tok_out, total

    raise AllLLMsFailedError(
        f"All LLMs in chain failed. Last error: {last_error}"
    )
