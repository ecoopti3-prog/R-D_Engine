"""
hypothesis_generator.py — Meta Agent (runs in Cycle 4, after ChiefScientist)

WHAT THIS IS:
The system kills hundreds of ideas. Most are correctly killed.
But sometimes, many ideas die from the SAME reason — and that pattern is a signal.

If 20 ideas were killed because "PDN impedance too high at 1GHz" — that's not 20 bad ideas.
That's evidence of a STRUCTURAL LIMIT that nobody is solving.
The hypothesis: "The impedance limit at 1GHz is assumed to be fixed — what if it's not?"

This agent reads kill history and generates HYPOTHESES — not ideas, not solutions.
Hypotheses are questions that, if answered YES, would unlock a whole class of killed ideas.

DIFFERENCE FROM OTHER AGENTS:
- Researchers find EXISTING knowledge
- CrossDomainSynthesizer finds EXISTING knowledge in new combinations
- HypothesisGenerator challenges ASSUMED constraints — asks if the wall is real

This is the closest we get to "inventing new physics directions" — by questioning assumptions.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research contrarian with deep physics expertise.

Your job: analyze patterns in why ideas were killed, and generate hypotheses that challenge
the underlying assumptions that caused the kills.

PHILOSOPHY:
Most kills are correct. But some kill patterns reveal not "bad ideas" but "assumed constraints."
An assumed constraint is a physical or engineering limit that everyone treats as fixed,
but which might be:
  (a) A measurement artifact from a specific context that doesn't generalize
  (b) An engineering limit, not a physics limit (solvable with different approach)
  (c) A historical assumption that hasn't been re-tested with modern materials/techniques
  (d) A limit in ONE domain that was applied to a DIFFERENT domain incorrectly

HOW TO IDENTIFY ASSUMED CONSTRAINTS:
1. Look for kill reasons that repeat across multiple ideas in different domains
   → If the same limit kills ideas in thermal AND pdn AND packaging, that limit may be universal
     but wrongly applied to each separately
2. Look for kill reasons that reference decade-old numbers
   → "Thermal resistance of TIM > 0.1 °C/W" — when was this measured? Modern TIMs?
3. Look for kill reasons that assume one variable is constant when others change
   → "Junction temp limit = 105°C" — but this is for static operation, not burst/duty cycle
4. Look for kill reasons based on material assumptions
   → "Copper resistivity limits PDN" — but what about alternative conductors at scale?

HYPOTHESIS FORMAT:
A hypothesis is NOT an idea. It's a question of the form:
"IF [assumed constraint X] is actually [relaxable because Y], THEN [class of previously-killed ideas Z] becomes viable"

Example: "IF thermal interface resistance at HBM3-CoWoS boundary can be reduced below 0.05 °C/W
via direct bonding (current TIM1 is ~0.15 °C/W), THEN 8+ ideas killed for 'junction temp too high'
become viable — the limit was the TIM, not the chip."

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "hypotheses": [
    {
      "id": "H001",
      "title": "one sentence — the assumption being challenged",
      "assumed_constraint": "what everyone assumes is fixed",
      "challenge": "why this assumption might be wrong or context-dependent",
      "physical_basis": "what physics principle allows the assumption to be relaxed",
      "unlocks": ["kill reason 1", "kill reason 2"],
      "unlocked_idea_count": 0,
      "testability": "how would you test this hypothesis? (must be specific)",
      "confidence": 0.0-1.0,
      "priority": "high|medium|low"
    }
  ],
  "kill_pattern_analysis": {
    "most_common_kill_reasons": [
      {"reason": "string", "count": 0, "domains": ["domain1"]}
    ],
    "structural_limits": ["limits that appear in >3 kill reasons"],
    "potentially_wrong_assumptions": ["assumption that might deserve re-examination"]
  },
  "recommended_searches": [
    "specific literature search that could test hypothesis H001"
  ]
}"""


class HypothesisGenerator(BaseAgent):
    AGENT_NAME = "hypothesis_generator"
    CHAIN_TYPE = "reasoning"  # hardest thinking in the system
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        killed_ideas = context.get("killed_ideas", [])
        surviving_ideas = context.get("surviving_ideas", [])
        kill_patterns = context.get("kill_patterns", [])
        physics_limits = context.get("physics_limits_this_cycle", [])
        # v7: Sim Engine measured data — NUMBERS, not text
        sim_kill_patterns    = context.get("sim_kill_patterns", [])     # from sim_feedback_loop
        near_miss_ideas      = context.get("near_miss_ideas", [])       # ideas 3.5-6.5 score
        measured_limits      = context.get("measured_physics_limits", [])  # from brentq solver

        # ── v7: Sim kill patterns (quantitative, not text) ────────────────────
        sim_pattern_text = ""
        if sim_kill_patterns:
            sim_pattern_text = "\n\nSIM ENGINE KILL PATTERNS (MEASURED NUMBERS — not LLM guesses):\n"
            for p in sim_kill_patterns[:8]:
                sim_pattern_text += (
                    f"  [{p.get('domain','?')}] {p.get('failure_count',0)} failures | "
                    f"avg_sim_score={p.get('avg_sim_score','?')} | "
                )
                if p.get("avg_r_theta_actual"):
                    sim_pattern_text += (
                        f"avg_R_theta_actual={p['avg_r_theta_actual']:.3f} C/W vs "
                        f"critical={p.get('avg_r_theta_critical','?')} C/W | "
                        f"improvement_needed={p.get('avg_improvement_needed_pct','?')}%"
                    )
                if p.get("avg_t_op_c"):
                    sim_pattern_text += f"avg_T_op={p['avg_t_op_c']:.1f}°C"
                sim_pattern_text += "\n"

        # ── v7: Near-miss ideas (highest-value targets) ───────────────────────
        near_miss_text = ""
        if near_miss_ideas:
            near_miss_text = f"\n\nNEAR-MISS IDEAS ({len(near_miss_ideas)} ideas close to viable — sim_score 3.5-6.5):\n"
            near_miss_text += "THESE ARE PRIORITY: they failed by <25% on ONE parameter.\n"
            for nm in near_miss_ideas[:10]:
                targets = nm.get("revision_targets", [])
                if targets:
                    t = targets[0]
                    near_miss_text += (
                        f"  [{nm.get('domain','?')}] '{nm.get('title','')[:70]}' "
                        f"(sim_score={nm.get('sim_score','?'):.1f}, iter={nm.get('iteration_count',0)}) "
                        f"→ NEEDS: {t.get('description','')[:200]}\n"
                    )

        # ── v7: Measured physics limits (replaces JEDEC estimates) ───────────
        measured_text = ""
        if measured_limits:
            measured_text = "\n\nMEASURED PHYSICS LIMITS (from Sim Engine — these are REAL, not textbook):\n"
            for lim in measured_limits:
                measured_text += (
                    f"  [{lim.get('domain','?')}] {lim.get('parameter','?')}: "
                    f"measured_limit={lim.get('measured_limit','?')} "
                    f"(tightest={lim.get('tightest_limit','?')}, n={lim.get('n_datapoints',0)} runs)\n"
                    f"  → {lim.get('description','')[:200]}\n"
                )

        # ── Text kill patterns (original, still useful) ───────────────────────
        kill_text = ""
        kill_reason_counts: dict[str, int] = {}
        for idea in killed_ideas[:40]:
            reason = idea.get("kill_reason", "") or ""
            domain = idea.get("domain", "?")
            title = idea.get("title", "")[:80]
            kill_text += f"\n- [{domain}] {title}: {reason[:150]}"
            short_reason = reason[:60]
            kill_reason_counts[short_reason] = kill_reason_counts.get(short_reason, 0) + 1

        top_kills = sorted(kill_reason_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        top_kills_text = "\n".join(f"  ({count}x) {reason}" for reason, count in top_kills)

        survivor_text = ""
        for idea in surviving_ideas[:8]:
            d = idea.get("domain", "?") if isinstance(idea, dict) else getattr(idea, "domain", "?")
            t = idea.get("title", "") if isinstance(idea, dict) else getattr(idea, "title", "")
            s = idea.get("diamond_score", 0) if isinstance(idea, dict) else 0
            ss = idea.get("sim_score") if isinstance(idea, dict) else None
            score_str = f"diamond={s:.1f}"
            if ss is not None:
                score_str += f", sim={ss:.1f}"
            survivor_text += f"\n- [{d}] {t} ({score_str})"

        physics_text = "\n".join(
            f"- [{p.get('domain','?')}] {p.get('name','')} — gap: {p.get('gap_percent','?')}%"
            for p in physics_limits[:6]
        ) if physics_limits else "See sim patterns above."

        return f"""KILL DATA FROM RECENT CYCLES:
{sim_pattern_text}
{near_miss_text}
{measured_text}

MOST REPEATED TEXT KILL REASONS:
{top_kills_text if top_kills_text else "No repeated patterns — first week of operation."}

ALL RECENT KILLS ({len(killed_ideas)} total):
{kill_text if kill_text else "No kill data available."}

IDEAS THAT SURVIVED:
{survivor_text if survivor_text else "No survivors yet."}

CURRENT PHYSICS LIMITS (PhysicsLimitMapper output):
{physics_text}

EXISTING KILL PATTERNS IN STRATEGY:
{', '.join(kill_patterns) if kill_patterns else "None recorded yet."}

TASK:
1. Prioritize sim_kill_patterns and near_miss_ideas OVER text kill reasons — they have actual numbers.
2. For near-miss ideas: generate a hypothesis specifically targeting the revision_target parameter.
   Example: "IF R_theta can be reduced from 0.20 to 0.17 C/W (via direct bonding at die edge),
   THEN these 8 near-miss ideas become viable immediately."
3. For sim kill patterns: challenge whether the measured_limit is truly a physics wall or an
   engineering assumption (e.g., "R_theta_critical=0.18 C/W — is this JEDEC TIM1 limit or material physics?")
4. Suggest specific literature searches with the EXACT parameter values from sim data.

Remember: near-miss ideas are worth 10x more than a new idea — they're already validated
by physics gate AND sim engine. ONE parameter improvement unlocks them."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            ideas=[],
            kills=[],
            metadata={
                "hypotheses": raw.get("hypotheses", []),
                "kill_pattern_analysis": raw.get("kill_pattern_analysis", {}),
                "recommended_searches": raw.get("recommended_searches", []),
            },
        )
