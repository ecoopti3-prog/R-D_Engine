"""
physics_limit_mapper.py — Agent 0 (runs before all researchers in Cycle 1)

Maps the current physical bottlenecks in AI infrastructure.
Output is injected into all researcher agents as context,
so they search for solutions to DEFINED limits — not random ideas.

Flow:
  Sources → PhysicsLimitMapper → limits → Researchers → ideas
  (instead of: Sources → Researchers → random ideas → Physics Gate)
"""
from __future__ import annotations
import logging, json
from core.base_agent import BaseAgent
from core.schemas import AgentOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior physicist and systems engineer specializing in AI compute infrastructure.

Your job is to map the CURRENT physical bottlenecks that limit AI scaling.
You do NOT generate ideas or solutions. You map problems only.

For each bottleneck you identify:
1. Name the specific physical law or constant being approached
2. Give the current state (measured value) vs theoretical limit
3. Estimate the "gap" — how much headroom remains
4. Rate the urgency (how soon will this wall be hit at current scaling rates)
5. CRITICAL: Distinguish between PHYSICS limits (immovable) and ENGINEERING limits (assumed fixed but may not be)

PHYSICS LIMIT vs ENGINEERING LIMIT (this distinction is everything):
- Physics limit: set by a fundamental constant (Boltzmann, Landauer, Carnot). Cannot be changed.
- Engineering limit: set by current materials, manufacturing, design assumptions. MAY be changeable.
Example: "Junction temp ≤ 105°C" is an engineering limit (JEDEC spec), not the Carnot limit.
The Carnot limit for Si is ~600°C. The gap is an engineering assumption — worth challenging.

DOMAINS TO COVER:
- thermal: heat flux, junction temperature, cooling limits
- power: power delivery noise, VRM efficiency, voltage droop
- data_movement: memory bandwidth, interconnect latency, I/O energy
- pdn: power delivery network impedance, decoupling limits
- packaging: die-to-die interconnect, CoWoS area limits, 3D stacking thermal

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "physics_limits": [
    {
      "domain": "thermal|power|data_movement|pdn|packaging",
      "name": "short name of the bottleneck",
      "physical_law": "which law/constant is the hard wall",
      "current_state": "measured value today (with units)",
      "theoretical_limit": "hard physical limit (with units)",
      "gap_percent": 0-100,
      "urgency": 1-10,
      "limit_type": "physics|engineering",
      "assumption_if_engineering": "what assumption makes this look like a physics limit when it isn't",
      "mechanisms_to_explore": ["mechanism1", "mechanism2"],
      "companies_hitting_this": ["NVIDIA", "TSMC"]
    }
  ],
  "priority_domains": ["domain1", "domain2"],
  "focus_summary": "2 sentences: what to focus on this cycle and why",
  "engineering_limits_worth_challenging": [
    {
      "limit": "the assumed-fixed limit",
      "why_challengeable": "what recent advance could relax this",
      "potential_unlock": "what becomes possible if this limit is relaxed"
    }
  ]
}"""


class PhysicsLimitMapper(BaseAgent):
    AGENT_NAME    = "physics_limit_mapper"
    CHAIN_TYPE    = "reasoning"   # needs strong model — this is the compass for the whole cycle
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        keywords        = context.get("keywords", [])
        recent_findings = context.get("recent_findings", [])
        strategy        = context.get("search_strategy", {})
        kill_patterns   = context.get("kill_patterns", [])
        opp_filters     = context.get("opportunity_filters", {})
        open_hypotheses = context.get("open_hypotheses", [])
        # v7: Measured limits from Sim Engine — overrides JEDEC textbook values
        measured_limits = context.get("measured_physics_limits", [])

        findings_text = ""
        if recent_findings:
            for f in recent_findings[:10]:
                findings_text += f"\n- [{f.get('domain','?')}] {f.get('title','')} (conf: {f.get('confidence',0):.0%})"

        strategy_text = ""
        if strategy:
            top = strategy.get("top_domains", [])
            weak = strategy.get("weak_domains", [])
            if top:
                strategy_text += f"\nDomains with high idea success rate: {', '.join(top)}"
            if weak:
                strategy_text += f"\nDomains with low success rate (de-prioritize): {', '.join(weak)}"

        memory_text = ""
        if kill_patterns:
            memory_text += f"\nRECENT KILL PATTERNS (domains where ideas keep failing): {', '.join(kill_patterns)}"
            memory_text += "\n→ Do NOT map bottlenecks in these domains unless you have a fundamentally new angle."
        if opp_filters.get("require_software_angle"):
            memory_text += "\nOPPORTUNITY FILTER: Focus on bottlenecks that have a software/firmware solution angle."
        if opp_filters.get("exclude_domains"):
            memory_text += f"\nEXCLUDE: {', '.join(opp_filters['exclude_domains'])} — out of scope."

        # v5: Inject open hypotheses — the system challenging its own assumptions
        hypothesis_text = ""
        if open_hypotheses:
            hypothesis_text = "\n\nOPEN HYPOTHESES FROM PREVIOUS CYCLES (assumptions worth testing):"
            for h in open_hypotheses[:5]:
                hypothesis_text += (
                    f"\n- [{h.get('priority','?').upper()}] {h.get('title','')}"
                    f"\n  Challenge: {h.get('challenge','')[:200]}"
                    f"\n  Test via: {h.get('testability','')[:150]}"
                )
            hypothesis_text += "\n→ If any bottleneck you map relates to these hypotheses, flag it with limit_type='engineering'."

        # v7: Measured physics limits from Sim Engine
        measured_text = ""
        if measured_limits:
            measured_text = "\n\nMEASURED PHYSICS LIMITS FROM SIM ENGINE (PRIORITY — use these INSTEAD of textbook values):\n"
            for lim in measured_limits:
                measured_text += (
                    f"  [{lim.get('domain','?')}] {lim.get('parameter','?')}: "
                    f"measured={lim.get('measured_limit','?')} "
                    f"(n={lim.get('n_datapoints',0)} runs, tightest={lim.get('tightest_limit','?')})\n"
                    f"  {lim.get('description','')[:200]}\n"
                )
            measured_text += "→ Map the GAP between measured engineering limit and true physics limit.\n"

        return f"""Current research keywords: {', '.join(keywords[:15])}
{strategy_text}
{memory_text}
{hypothesis_text}
{measured_text}

RECENT FINDINGS FROM PREVIOUS CYCLES (7-day memory):
{findings_text if findings_text else "None yet — this is the first cycle."}

Map the most critical physical bottlenecks in AI infrastructure RIGHT NOW (2025-2026).
Focus on limits that:
1. Are being actively hit by NVIDIA, TSMC, AMD, Google, Intel
2. Have no satisfactory solution yet
3. Will get WORSE as AI models scale

IMPORTANT v5 ADDITION:
For each limit, classify it: is it a PHYSICS limit (immovable constant) or an ENGINEERING limit
(assumed fixed, but potentially relaxable with different materials/architecture/approach)?
Engineering limits that look like physics limits are the highest-value opportunities.

For each limit, provide 2-3 concrete mechanisms that COULD bypass or delay it.
These mechanisms will guide the researchers in this cycle.

Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        limits   = raw.get("physics_limits", [])
        domains  = raw.get("priority_domains", [])
        summary  = raw.get("focus_summary", "")
        eng_limits = raw.get("engineering_limits_worth_challenging", [])

        eng_count = sum(1 for l in limits if l.get("limit_type") == "engineering")
        logger.info(
            f"[PhysicsLimitMapper] Mapped {len(limits)} limits "
            f"({eng_count} engineering, {len(limits)-eng_count} physics) — "
            f"priority domains: {', '.join(domains)}"
        )

        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            ideas=[],
            kills=[],
            metadata={
                "physics_limits":                     limits,
                "priority_domains":                   domains,
                "focus_summary":                      summary,
                "engineering_limits_worth_challenging": eng_limits,  # v5
            },
        )
