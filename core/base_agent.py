"""
base_agent.py — All agents inherit from BaseAgent.
Handles: LLM call, JSON validation, retry, fallback, logging, timing.

Chunking support:
  Agents that process lists of ideas support run_chunked(ideas, chunk_size=5).
  This keeps every LLM call small and safe regardless of how many ideas exist.
"""
from __future__ import annotations
import time, logging, json, math
from datetime import datetime, timezone
from typing import Optional
from pydantic import ValidationError
from core.llm_router import call_llm, AllLLMsFailedError
from core.schemas import AgentOutput, Finding, Idea, Kill

logger = logging.getLogger(__name__)

# Ideas per LLM call — keeps input+output safely under 4000 tokens
CHUNK_SIZE = 5


class BaseAgent:
    """
    Subclass this and implement:
      - AGENT_NAME: str
      - SYSTEM_PROMPT: str
      - build_user_prompt(context: dict) -> str
      - parse_output(raw: dict, cycle_id: str) -> AgentOutput

    For batch agents (extractors, physics, market, critics):
      Call run_chunked(ideas, chunk_size, cycle_id, extra_context)
      instead of run() directly.
    """

    AGENT_NAME: str = "base_agent"
    SYSTEM_PROMPT: str = ""
    CHAIN_TYPE: str = "reasoning"  # "reasoning" | "research" — set per agent class

    # ── Single call ───────────────────────────────────────────────────────────
    def run(self, context: dict, cycle_id: str) -> AgentOutput:
        """Main entry point. Never raises — returns failed AgentOutput on error."""
        start = time.time()
        logger.info(f"[{self.AGENT_NAME}] Starting — cycle {cycle_id}")

        user_prompt = self.build_user_prompt(context)

        try:
            raw, llm_used, tok_in, tok_out, total_tokens = call_llm(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                expect_json=True,
                chain=self.CHAIN_TYPE,
            )
        except AllLLMsFailedError as e:
            logger.error(f"[{self.AGENT_NAME}] All LLMs failed: {e}")
            return self._failed_output(cycle_id, str(e), None, int((time.time()-start)*1000))

        duration_ms = int((time.time() - start) * 1000)
        try:
            output = self.parse_output(raw, cycle_id)
            output.llm_used    = llm_used
            output.tokens_used = total_tokens
            output.duration_ms = duration_ms
            output.status      = "done"
            logger.info(
                f"[{self.AGENT_NAME}] Done — "
                f"{len(output.findings)} findings, "
                f"{len(output.ideas)} ideas, "
                f"{len(output.kills)} kills | "
                f"{output.duration_ms}ms | {llm_used} in={tok_in} out={tok_out}"
            )
            # v7: async cost logging (non-blocking)
            try:
                import db.supabase_client as _db
                from db.sim_feedback_loop import log_api_cost
                _llm_cfg = next(
                    (c for chain in [__import__('config.settings', fromlist=['REASONING_CHAIN']).REASONING_CHAIN,
                                     __import__('config.settings', fromlist=['RESEARCH_CHAIN']).RESEARCH_CHAIN]
                     for c in chain if c["name"] == llm_used), None
                )
                if _llm_cfg:
                    log_api_cost(
                        _db.get_client(), cycle_id, self.AGENT_NAME,
                        llm_used, _llm_cfg["model"],
                        tok_in, tok_out, duration_ms,
                    )
            except Exception:
                pass  # cost logging is never blocking
            return output
        except (ValidationError, KeyError, TypeError) as e:
            logger.error(f"[{self.AGENT_NAME}] Output parse error: {e}\nRaw: {json.dumps(raw)[:500]}")
            return self._failed_output(cycle_id, str(e), llm_used, duration_ms)

    # ── Chunked batch call ────────────────────────────────────────────────────
    def run_chunked(
        self,
        ideas: list,
        cycle_id: str,
        chunk_size: int = CHUNK_SIZE,
        extra_context: dict = None,
    ) -> AgentOutput:
        """
        Split ideas into chunks of chunk_size, call run() per chunk,
        then merge all outputs into a single AgentOutput.

        extra_context: additional keys passed alongside each chunk's ideas
        (e.g. findings, market_assessments).
        """
        if not ideas:
            return self._empty_output(cycle_id)

        extra_context = extra_context or {}
        n_chunks = math.ceil(len(ideas) / chunk_size)
        logger.info(
            f"[{self.AGENT_NAME}] Chunked run: {len(ideas)} ideas "
            f"→ {n_chunks} chunks of {chunk_size}"
        )

        merged = self._empty_output(cycle_id)
        total_tokens = 0

        for i in range(n_chunks):
            chunk = ideas[i * chunk_size : (i + 1) * chunk_size]
            context = {"ideas": chunk, **extra_context}

            output = self.run(context, cycle_id)

            # Merge results
            merged.findings.extend(output.findings)
            merged.ideas.extend(output.ideas)
            merged.kills.extend(output.kills)
            total_tokens += (output.tokens_used or 0)

            if output.llm_used and not merged.llm_used:
                merged.llm_used = output.llm_used

            # Merge metadata lists (assessments, scored_ideas, etc.)
            for key, val in output.metadata.items():
                if isinstance(val, list):
                    merged.metadata.setdefault(key, [])
                    merged.metadata[key].extend(val)
                elif isinstance(val, dict) and key in merged.metadata:
                    merged.metadata[key].update(val)
                elif key not in merged.metadata:
                    merged.metadata[key] = val

            # Respect rate limits between chunks
            if i < n_chunks - 1:
                time.sleep(3)

        merged.tokens_used = total_tokens
        merged.status      = "done"
        logger.info(
            f"[{self.AGENT_NAME}] Chunked done — "
            f"{len(merged.findings)} findings, "
            f"{len(merged.ideas)} ideas, "
            f"{len(merged.kills)} kills | "
            f"{total_tokens} total tokens"
        )
        return merged

    # ── Helpers ───────────────────────────────────────────────────────────────
    def build_user_prompt(self, context: dict) -> str:
        raise NotImplementedError

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        raise NotImplementedError

    def _empty_output(self, cycle_id: str) -> AgentOutput:
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="done",
            findings=[],
            ideas=[],
            kills=[],
            metadata={},
        )

    def _failed_output(self, cycle_id: str, error: str, llm_used: Optional[str], duration_ms: int) -> AgentOutput:
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="failed",
            findings=[],
            ideas=[],
            kills=[],
            metadata={"error": error},
            llm_used=llm_used,
            duration_ms=duration_ms,
        )

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()
