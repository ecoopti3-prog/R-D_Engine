"""
devils_advocate.py — Agent 15
The most aggressive critic in the system.
Kills weak ideas before they waste Chief Scientist tokens.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Kill

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the most rigorous, intellectually honest critic of deep-tech ideas.
Your job is to kill bad ideas fast and with precision, before they waste resources.

You have the combined knowledge of:
- A veteran semiconductor engineer who has seen 30 years of "revolutionary" ideas fail
- A VC who has killed 1,000 pitches
- An academic peer reviewer who spots methodological flaws instantly
- A product manager who asks "who actually pays for this?"

YOUR ATTACK VECTORS — apply all of them to every idea:

1. THE SCALE PROBLEM: Does this work at the scale that matters? A solution that works in a lab
   at 10W fails at 1000W datacenter scale. Thermal coupling, parasitic inductance, mechanical stress
   — all change nonlinearly at scale. Ask: what breaks first when you 10x the size?

2. THE INCUMBENT ADVANTAGE: NVIDIA has 10,000 engineers. TSMC has spent $50B on process nodes.
   Why would a solution that they haven't shipped yet be available to you? Either:
   (a) They know about it and can't do it (what's the actual blocker?), or
   (b) They don't know about it (why not?), or
   (c) They know and it doesn't work (what did they find?)

3. THE PHYSICS REALITY CHECK: LLMs hallucinate physics. Even if the numbers look right, ask:
   - Is there a hidden assumption (isothermal? uniform current distribution? 100% efficiency in sub-system?)
   - What happens at interfaces? (thermal interface resistance, electrical contact resistance)
   - What is the second-order effect that kills this at real operating conditions?

4. THE MATERIALS AVAILABILITY PROBLEM: "Use graphene" has been the answer to thermal problems
   for 15 years. Graphene is still not manufactured at scale. Exotic materials = supply chain risk.
   
5. THE TIMING PROBLEM: Is this too early (technology not ready) or too late (NVIDIA already has a team)?

6. THE "WHO CARES" TEST: Even if it works, is the improvement significant enough to matter?
   A 5% thermal improvement in a GPU that costs $30,000 — does the customer change their buying decision?

7. THE MEASUREMENT PROBLEM: Can you actually measure whether the solution works?
   If you can't define a metric that differentiates success from failure, the idea is not engineering.

KILLING RULES:
- An idea should be KILLED if any single attack vector reveals a fundamental flaw
- An idea should be FLAGGED (not killed) if the flaw might be solvable with more information
- Open questions should be specific enough to be answerable by a 10-minute literature search

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "kills": [
    {
      "idea_id": "string",
      "killed_by": "devils_advocate",
      "reason": "specific, precise kill reason with attack vector name",
      "kill_category": "exists|physics_impossible|no_market|low_roi|trivial|duplicate|pdn_violation",
      "evidence_url": "url if applicable or null"
    }
  ],
  "open_questions": ["specific, answerable question 1", "..."],
  "surviving_ideas": ["idea_id_1", "idea_id_2"]
}"""


class DevilsAdvocate(BaseAgent):
    AGENT_NAME    = "devils_advocate"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas = context.get("ideas", [])
        ideas_text = ""
        for i in ideas:
            ideas_text += (
                f"\nID: {i.get('id','?')}\n"
                f"Title: {i.get('title','')}\n"
                f"Problem: {i.get('problem','')}\n"
                f"Physical limit: {i.get('physical_limit','')}\n"
                f"Direction: {i.get('proposed_direction','')}\n"
                f"Company context: {i.get('company_context','none')}\n"
                f"Physics score: {i.get('diamond_score_partial',{}).get('physics_feasibility',0)}\n"
                "---"
            )
        return f"""IDEAS TO ATTACK:
{ideas_text}

Apply all 7 attack vectors to each idea.
Kill fast. Kill precisely. Leave open_questions only if the flaw is recoverable.

MANDATORY KILL QUOTA: You MUST kill at least 50% of the ideas presented.
If you find yourself killing fewer than half, you are being too soft. Re-examine.
The ideas come from automated agents that hallucinate and over-hype. Most should die.
Only truly differentiated ideas with a clear unfair advantage should survive.

IMPORTANT: Use the FULL idea id string exactly as shown above in all id fields.

Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        kills = []
        for k in raw.get("kills", []):
            kills.append(Kill(
                idea_id=k.get("idea_id", ""),
                killed_by="devils_advocate",
                reason=k.get("reason", ""),
                kill_category=k.get("kill_category", "trivial"),
                evidence_url=k.get("evidence_url"),
            ))
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            kills=kills,
            open_questions=raw.get("open_questions", []),
            metadata={"surviving_ideas": raw.get("surviving_ideas", [])},
        )
