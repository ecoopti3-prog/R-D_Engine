"""
competition_analyst.py — Agent 14
Kills ideas that already exist or are already being solved by well-funded players.
The "exists" kill filter — before devil's advocate.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Kill

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a technology competitive intelligence specialist. Your job is to
determine whether an idea is truly novel or is already being pursued by well-funded incumbents.

THE CORE QUESTION for each idea: "Is there a team with >$10M in funding already building this?"

If yes → kill it. Not because the idea is bad — but because you can't win against NVIDIA's
10,000-engineer team if they're already on it.

If no → flag it as "open" and identify exactly WHY the giants aren't doing it.

THE "WHY ISN'T NVIDIA DOING THIS?" QUESTION — possible answers:
1. They ARE doing it (and you should find evidence)
2. They tried and failed (find the failed products/papers)
3. It's not in their core business model (they won't do it even if it works)
4. The solution requires a technology they don't have (your window)
5. It's too small for them to care (your window at a different scale)

SOURCES TO CHECK FOR EACH IDEA:
- NVIDIA, AMD, Intel, Google, TSMC research publications
- Recent acquisitions in the space (acquisition = they know the problem exists)
- Series A/B/C startups in the same domain (Crunchbase signals)
- Academic groups that are the feeders for these companies (MIT, Stanford, CMU, ETH Zurich)

KILLING RULES:
- KILL if: 2+ papers from major companies directly address the same constraint
- KILL if: A funded startup (>$5M) is already in market with this approach
- FLAG if: Related work exists but no one is attacking the specific combination
- PASS if: You cannot find a direct competitor after thorough search

DO NOT kill based on broad similarities. "Someone else does thermal management" is not a kill.
"NVIDIA published 3 papers on exactly this cooling approach for 3D stacking" IS a kill.

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "competition_assessments": [
    {
      "idea_id": "string",
      "action": "kill|flag|pass",
      "competition_level": "none|emerging|established|saturated",
      "key_competitors": [
        {"name": "company or team", "evidence": "paper/product/funding", "url": "url_or_null"}
      ],
      "why_giants_not_doing_it": "explanation if action=pass",
      "window_of_opportunity": "null or specific opportunity window",
      "kill_reason": "null or specific reason"
    }
  ]
}"""


class CompetitionAnalyst(BaseAgent):
    AGENT_NAME    = "competition_analyst"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas = context.get("ideas", [])
        ideas_text = ""
        for i in ideas:
            ideas_text += (
                f"\nID: {i.get('id','?')}\n"
                f"Title: {i.get('title','')}\n"
                f"Domain: {i.get('domain','')}\n"
                f"Problem: {i.get('problem','')[:200]}\n"
                f"Direction: {( i.get('proposed_direction') or '' )[:150]}\n"
                f"Company context: {i.get('company_context','none')}\n"
                "---"
            )

        return f"""IDEAS TO CHECK FOR COMPETITION:
{ideas_text}

For each idea:
1. Search for direct competitors (same problem + same approach)
2. Check major company research publications (NVIDIA, TSMC, AMD, Intel, Google)
3. Look for funded startups in the same space (last 3 years)
4. If the giants are already on it → KILL with evidence
5. If no one is doing it → WHY? Explain the reason precisely.

Focus on ideas in technical domains where NVIDIA/TSMC/AMD are active.
Remember: they have 10,000+ engineers. If they care about this, they're already building it.

IMPORTANT: Use the FULL idea id string exactly as shown above in all id fields.

Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        kills = []
        open_questions = []

        for a in raw.get("competition_assessments", []):
            if a.get("action") == "kill" and a.get("kill_reason"):
                kills.append(Kill(
                    idea_id=a.get("idea_id", ""),
                    killed_by="competition_analyst",
                    reason=a.get("kill_reason", ""),
                    kill_category="exists",
                    evidence_url=a.get("key_competitors", [{}])[0].get("url") if a.get("key_competitors") else None,
                ))
            elif a.get("action") == "flag":
                open_questions.append(
                    f"Idea {a.get('idea_id','?')[:8]}: verify competition — {a.get('key_competitors')}"
                )

        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            kills=kills,
            open_questions=open_questions,
            metadata={"assessments": raw.get("competition_assessments", [])},
        )
