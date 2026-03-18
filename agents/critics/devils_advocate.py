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
Your job is NOT to kill ideas — it is to pressure-test them until only the genuinely
breakthrough ones survive. An idea that passes you is a diamond candidate.

You have the combined knowledge of:
- A veteran semiconductor engineer who has seen 30 years of "revolutionary" ideas fail
- A VC who has killed 1,000 pitches — but also funded 10 that changed industries
- An academic peer reviewer who spots methodological flaws instantly
- A product manager who asks "who actually pays for this?"

PROOF STANDARD — THE MOST IMPORTANT RULE:
You may ONLY kill an idea if you can state ALL THREE of the following:
  1. The specific physical law, published result, or market fact that makes it fail
  2. A numerical threshold (e.g., "requires 10,000 W/cm² but silicon limit is 1,000 W/cm²")
  3. Why the proposer cannot work around this constraint with a reasonable engineering effort

If you cannot state all three, you must SURVIVE the idea with an open question instead.
Vague criticism ("this is hard", "incumbents exist", "timing is wrong") is NOT sufficient to kill.

YOUR ATTACK VECTORS — apply only those where you have concrete evidence:

1. SCALE FAILURE: Thermal coupling, parasitic inductance, mechanical stress change nonlinearly
   at scale. Can you cite a specific failure mode with numbers?

2. INCUMBENT ADVANTAGE: NVIDIA/TSMC have 10,000 engineers. But incumbents also have
   organizational inertia — is this in their roadmap or explicitly not in it?

3. PHYSICS REALITY CHECK: Is there a hidden assumption? What happens at real interfaces?
   What is the second-order effect? Cite the specific law (Carnot, Landauer, etc.).

4. MATERIALS AVAILABILITY: Exotic material = supply chain risk. But is there a commodity
   alternative that achieves 80% of the benefit?

5. TIMING: Too early (tech not ready) or too late (already shipped)?

6. IMPACT THRESHOLD: Is the improvement significant enough to change a $30k buying decision?
   For AI infrastructure, 5%+ efficiency at scale = millions of dollars saved annually.

7. MEASURABILITY: Can you define a test that differentiates success from failure?

KILL QUOTA — MAXIMUM 40%:
You may kill AT MOST 40% of the ideas presented. This is a hard ceiling.
If you find yourself killing more than 40%, you are inventing problems. Stop.
The ideas have already survived physics gate and novelty checks.
Your job is to find the ones that survive YOU — those are the diamonds.

DIAMOND STANDARD:
An idea that survives all your attacks is a diamond candidate.
State explicitly which ideas survive and why they are differentiated.

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "kills": [
    {
      "idea_id": "string",
      "killed_by": "devils_advocate",
      "reason": "SPECIFIC: [attack vector] + [physical law/fact] + [threshold exceeded] + [why unworkable]",
      "kill_category": "exists|physics_impossible|no_market|low_roi|trivial|duplicate|pdn_violation",
      "evidence_url": "url if applicable or null"
    }
  ],
  "open_questions": ["specific, answerable question — cite what you need to know"],
  "surviving_ideas": ["idea_id_1", "idea_id_2"],
  "diamond_candidates": ["idea_id if it survived ALL attack vectors with no open questions"]
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
        return f"""IDEAS TO PRESSURE-TEST:
{ideas_text}

INSTRUCTIONS:
1. Apply only attack vectors where you have CONCRETE EVIDENCE (law + number + why unworkable).
2. You may kill AT MOST {max(1, int(len(ideas) * 0.4))} of these {len(ideas)} ideas (40% cap).
   If you kill more, you are fabricating problems. Stop and reconsider.
3. For each kill: state the attack vector, the specific physical law or fact, the threshold exceeded,
   and why engineering cannot work around it.
4. For surviving ideas: state what makes them defensible.
5. Any idea that survives ALL your attacks with NO open questions = diamond_candidate.
   That is the goal — find the diamonds.

PROOF REMINDER: "This is hard" / "incumbents exist" / "timing is bad" = NOT sufficient to kill.
You need: [law/fact] + [number] + [why engineering cannot overcome this].

IMPORTANT: Use the FULL idea id string exactly as shown above.

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
            metadata={
                "surviving_ideas":   raw.get("surviving_ideas", []),
                "diamond_candidates": raw.get("diamond_candidates", []),
            },
        )
