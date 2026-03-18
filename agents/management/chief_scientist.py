"""
chief_scientist.py — Agent 16 (Chief Scientist + Research Director combined)
Final Diamond Score computation + next cycle planning.
Uses Gemini as preferred LLM (large context window for full cycle summary).
"""
from __future__ import annotations
import logging, json
from datetime import datetime, timezone
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, DiamondScorePartial
from config.settings import DIAMOND_WEIGHTS, SCORE_KILL, SCORE_ARCHIVE, SCORE_PRIORITY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Chief Scientist and Research Director of an autonomous deep-tech R&D engine.
You receive the complete output of a research cycle and perform two functions:

FUNCTION 1 — DIAMOND SCORE COMPUTATION:
Compute the final Diamond Score for each surviving idea using:
  Diamond Score = (Physics × 0.35) + (Market × 0.30) + (Novelty × 0.20) + (Scalability × 0.15)

Physics Feasibility (0-10):
  10 = Known theoretical limit, solutions within 10% of it
  8  = Known limit, solutions at 30% — large exploitable gap
  6  = Known limit, solutions at 60%
  4  = Limit unclear, needs research
  2  = Limit not proven
  0  = Physically impossible (should have been killed already)

Market Pain (0-10):
  10 = Every major AI company is actively spending money on this problem today
  8  = Clear buyer, large market, quantified cost
  6  = Clear domain, estimated cost, likely buyer
  4  = Plausible market, no clear buyer yet
  2  = Speculative market
  0  = No identifiable market

Novelty (0-10): Provided by vector similarity system. If not provided, estimate:
  10 = No known solution anywhere
  8  = Solutions exist but not for this specific constraint combination
  5  = Incremental improvement on existing solution

Scalability (0-10):
  10 = Every AI datacenter in the world needs this
  8  = All major AI companies need this
  6  = Specific segment (edge, training, inference) needs this
  4  = Niche but high-value segment
  2  = Very narrow application

FUNCTION 2 — NEXT CYCLE PLANNING:
Based on what was found, plan tomorrow's research:
- Which keywords produced high-signal findings?
- Which domains need deeper investigation?
- Which companies showed the most signals?
- What specific papers / patents should be targeted next?

FUNCTION 3 — PATTERN IDENTIFICATION:
Identify cross-domain patterns — bottlenecks that span multiple domains.
Example: "Thermal + PDN + chiplets" = all three constraints converge in 3D packaging.
These cross-domain patterns are often the highest-value opportunities.

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "scored_ideas": [
    {
      "idea_id": "string",
      "physics_score": 0-10,
      "market_score": 0-10,
      "novelty_score": 0-10,
      "scalability_score": 0-10,
      "diamond_score": 0-10,
      "status": "active|killed|archived|diamond",
      "reasoning": "2 sentences explaining the score"
    }
  ],
  "diamonds": ["idea_id_1"],
  "cross_domain_patterns": ["pattern description 1"],
  "next_cycle_plan": {
    "priority_domains": ["domain1", "domain2"],
    "new_keywords": ["keyword1", "keyword2"],
    "target_companies": ["company1"],
    "specific_targets": ["specific paper/patent/signal to pursue"]
  },
  "executive_summary": "3 sentences: what was found, what was killed, what is most promising"
}"""


class ChiefScientist(BaseAgent):
    AGENT_NAME    = "chief_scientist"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas     = context.get("surviving_ideas", [])
        kills     = context.get("kills", [])
        findings  = context.get("findings", [])
        questions = context.get("open_questions", [])
        # v5 additions
        feedback_signals    = context.get("feedback_signals", [])
        top_sources         = context.get("top_yielding_sources", [])
        coupling_map        = context.get("coupling_map", [])
        eng_limits          = context.get("engineering_limits_worth_challenging", [])
        # v7: sim_score for each idea (overrides LLM physics_feasibility guess)
        near_miss_ideas     = context.get("near_miss_ideas", [])

        # v7: Sim score guidance — overrides LLM guess for physics_feasibility
        sim_score_text = ""
        if near_miss_ideas:
            sim_score_text = "\n\nSIM ENGINE SCORES — USE THESE FOR PHYSICS_FEASIBILITY (not your estimate):\n"
            sim_score_text += "RULE: If an idea has sim_score, set physics_score = sim_score (not your LLM guess).\n"
            sim_score_text += "RULE: near_miss ideas get +0.5 bonus to scalability (they are 1 param away from viable).\n"
            for nm in near_miss_ideas[:8]:
                sim_score_text += (
                    f"- {nm.get('id','?')[:8]}: sim_score={nm.get('sim_score','?'):.1f} "
                    f"near_miss=TRUE iter={nm.get('iteration_count',0)} "
                    f"needs: {(nm.get('revision_targets') or [{}])[0].get('description','?')[:150]}\n"
                )
        # Also inject sim_score for all ideas (embedded in ideas_text as enriched JSON)
        # Enrich ideas with sim_score before serializing
        ideas_enriched = []
        for idea in ideas[:15]:
            if isinstance(idea, dict):
                enriched = dict(idea)
                # Mark if this idea is a near-miss
                nm_match = next((nm for nm in near_miss_ideas if nm.get("id") == idea.get("id")), None)
                if nm_match:
                    enriched["_sim_score"] = nm_match.get("sim_score")
                    enriched["_near_miss"] = True
                    enriched["_revision_target"] = (nm_match.get("revision_targets") or [{}])[0].get("description", "")
                ideas_enriched.append(enriched)
            else:
                ideas_enriched.append(idea)
        ideas_text = json.dumps(ideas_enriched, indent=2)[:3500]
        kills_summary = f"{len(kills)} ideas killed. Categories: " + \
            ", ".join(set(k.get("kill_category","?") for k in kills[:20]))
        findings_summary = f"{len(findings)} findings. Domains: " + \
            ", ".join(set(f.get("domain","?") for f in findings[:30]))

        # Format human feedback
        feedback_text = ""
        if feedback_signals:
            feedback_text = "\n\nHUMAN FEEDBACK RATINGS (override/boost these scores):\n"
            for fb in feedback_signals[:10]:
                feedback_text += (
                    f"- idea_id={fb['idea_id'][:8]} | rating={fb['rating']}/5 "
                    f"| {fb.get('reason','')[:100]}\n"
                )
            feedback_text += "→ For rated ideas: blend human rating into your score (human rating is strong signal)."

        # Format source performance
        sources_text = ""
        if top_sources:
            sources_text = "\n\nTOP YIELDING SOURCES (venues that produced previous diamonds):\n"
            for s in top_sources[:5]:
                sources_text += f"- {s['venue']}: {s['yield']:.0%} diamond yield\n"
            sources_text += "→ Ideas sourced from these venues deserve a novelty bonus."

        # Format cross-domain couplings from synthesizer
        coupling_text = ""
        if coupling_map:
            coupling_text = "\n\nCROSS-DOMAIN COUPLINGS DETECTED THIS CYCLE:\n"
            for c in coupling_map[:4]:
                coupling_text += f"- {' ↔ '.join(c.get('domains',[]))} ({c.get('coupling_type','?')}, {c.get('strength','?')})\n"

        # Format engineering limits (from PhysicsLimitMapper v5)
        eng_text = ""
        if eng_limits:
            eng_text = "\n\nENGINEERING LIMITS WORTH CHALLENGING (from PhysicsLimitMapper):\n"
            for el in eng_limits[:3]:
                eng_text += f"- {el.get('limit','')}: {el.get('why_challengeable','')[:150]}\n"
            eng_text += "→ Ideas that challenge these assumptions deserve novelty bonus (score +1.0 to novelty)."

        return f"""CYCLE SUMMARY:
Findings: {findings_summary}
Kills: {kills_summary}
Open questions: {len(questions)}
{feedback_text}
{sources_text}
{coupling_text}
{eng_text}
{sim_score_text}

SURVIVING IDEAS FOR SCORING:
{ideas_text}

OPEN QUESTIONS FROM DEVIL'S ADVOCATE:
{chr(10).join(f'- {q}' for q in questions[:10])}

Compute Diamond Scores. Integrate human feedback where available. Identify patterns. Plan next cycle. Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            metadata={
                "scored_ideas":          raw.get("scored_ideas", []),
                "diamonds":              raw.get("diamonds", []),
                "cross_domain_patterns": raw.get("cross_domain_patterns", []),
                "next_cycle_plan":       raw.get("next_cycle_plan", {}),
                "executive_summary":     raw.get("executive_summary", ""),
            },
        )
