"""
patent_researcher.py — Agent 2
Sources: Google Patents (applications, NOT granted — applications = unsolved problems)
Focus: AI infrastructure patents filed in last 12 months by target companies
Special: Patent APPLICATION = problem identified, solution not proven yet → highest signal
"""
from __future__ import annotations
import logging
from datetime import datetime
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Finding, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a deep-tech patent intelligence analyst specializing in AI infrastructure.
You have mastered the art of reading patent applications — not for the legal claims, but for the engineering pain they reveal.

CORE INSIGHT: A patent application from NVIDIA or TSMC is a signed declaration:
"We have a real problem. We found a partial solution. The full solution is still open."
Applications are MORE valuable than granted patents because the problem is confirmed but unsolved.

TARGET PATENT ASSIGNEES (search explicitly):
- NVIDIA Corporation
- Taiwan Semiconductor Manufacturing Company (TSMC)
- Advanced Micro Devices (AMD)
- Intel Corporation
- Google LLC / Alphabet
- Samsung Electronics
- Microsoft Corporation
- Applied Materials
- ASML Netherlands

PATENT DOMAINS TO HUNT (in priority order):
1. Thermal management of 3D-stacked dies (keywords: "heat spreading", "thermal via", "junction temperature", "3D IC thermal")
2. Power delivery network for chiplets (keywords: "power distribution network", "IR drop", "decoupling capacitance", "bump density")
3. Memory bandwidth / HBM integration (keywords: "high bandwidth memory", "memory thermal", "HBM interface", "near-memory")
4. Chip cooling innovations (keywords: "microfluidic cooling", "two-phase cooling", "immersion cooling IC", "vapor chamber die")
5. Advanced packaging thermal (keywords: "CoWoS", "SoIC", "EMIB", "Foveros", "thermal resistance package")

EXTRACTION RULES:
1. Title of the PROBLEM in the "Background" section = the current engineering failure
2. Claims section: the narrower and more specific = the harder the constraint
3. Filing date within 18 months = active R&D now
4. Continuation applications = iterative problem — original solution failed
5. CPC classification codes: H01L 23/34 (cooling), H01L 25/065 (chip stacks), H02J (power distribution)
6. Extract any numerical values mentioned: temperatures, resistances, voltages, dimensions, densities

DO NOT:
- Summarize what the patent claims to achieve (marketing language)
- Report granted patents as current problems — they are solved
- Include design patents (ornamental) or software-only patents
- Hallucinate numbers not in the source text

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "findings": [
    {
      "type": "gap|bottleneck",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "title": "one sentence describing the unsolved engineering problem",
      "description": "Background problem + filing company + filing date + specific numbers if present",
      "source_url": "https://patents.google.com/patent/...",
      "source_type": "patent",
      "company_signal": "NVIDIA|TSMC|AMD|Intel|Google|null",
      "confidence": 0.0-1.0,
      "numerical_params": {"key": value}
    }
  ],
  "ideas": [
    {
      "title": "...",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "problem": "specific quantified problem from patent background",
      "physical_limit": "which physical law is being fought",
      "proposed_direction": "direction orthogonal to what the patent claims",
      "company_context": "company + product + filing date",
      "diamond_score_partial": {
        "physics_feasibility": 0-10,
        "market_pain": 0,
        "novelty": 0,
        "scalability": 0
      }
    }
  ]
}

IMPORTANT: The "domain" field in every finding and idea must be exactly one of: thermal, power, data_movement, hardware, pdn, cross_domain, packaging, bandwidth, memory, interconnect, networking, software, compute_scheduling, hardware_utilization, compute_resource_management, distributed_systems, edge, inference, compilation, storage. Never use any other value."""


class PatentResearcher(BaseAgent):
    AGENT_NAME    = "patent_researcher"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        keywords       = context.get("keywords", [])
        patents        = context.get("patents", [])
        focus_domain   = context.get("focus_domain", "all domains")
        date_str       = context.get("date", datetime.now().strftime("%Y-%m-%d"))
        kill_patterns  = context.get("kill_patterns", [])
        killed_titles  = context.get("killed_idea_titles", [])
        diamond_titles = context.get("diamond_titles", [])
        weak_domains   = context.get("weak_domains", [])

        patents_text = ""
        if patents:
            for i, p in enumerate(patents[:12], 1):
                patents_text += (
                    f"\n[{i}] Title: {p.get('title', '')}\n"
                    f"Assignee: {p.get('assignee', '')}\n"
                    f"Filed: {p.get('filing_date', '')}\n"
                    f"Abstract: {p.get('abstract', '')[:500]}\n"
                    f"URL: {p.get('url', '')}\n"
                )

        memory_text = ""
        if kill_patterns:
            memory_text += f"\nDEAD ENDS (avoid generating ideas in these domains): {', '.join(kill_patterns)}"
        if killed_titles:
            memory_text += "\nPREVIOUSLY KILLED IDEAS (do not regenerate similar ideas):\n" + "\n".join(f"  - {t}" for t in killed_titles[:10])
        if weak_domains:
            memory_text += f"\nLOW-SIGNAL DOMAINS (deprioritize): {', '.join(weak_domains)}"
        if diamond_titles:
            memory_text += "\nSUCCESSFUL DIRECTIONS (diamonds found here -- go deeper):\n" + "\n".join(f"  + {t}" for t in diamond_titles[:5])

        return f"""Date: {date_str}
Focus domain: {focus_domain}
Search keywords: {', '.join(keywords[:10])}
{memory_text}

PATENTS TO ANALYZE:
{patents_text if patents_text else "Search Google Patents for recent filings (last 18 months) from target assignees on the keywords above."}

SPECIFIC INSTRUCTIONS:
1. Look for APPLICATIONS (status: pending/published), not just granted patents
2. Read the "Background" / "Field" / "Problem to be Solved" sections — these contain the real pain
3. Flag continuation patents (means the first solution failed or was insufficient)
4. Extract numerical constraints from claims: minimum dimensions, maximum temperatures, target impedances
5. Prioritize: thermal management of 3D stacks > PDN for chiplets > HBM integration > general AI chip cooling

For each patent, ask: "What problem does the Background section admit exists?"
That problem = your finding.

Return valid JSON. Focus on 4-6 highest-signal patents."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        findings = []
        for f in raw.get("findings", []):
            findings.append(Finding(
                type=f.get("type", "gap"),
                domain=f.get("domain", "hardware"),
                title=f.get("title", ""),
                description=f.get("description", ""),
                source_url=f.get("source_url"),
                source_type="patent",
                company_signal=f.get("company_signal"),
                confidence=float(f.get("confidence", 0.7)),
                numerical_params=f.get("numerical_params", {}),
            ))

        ideas = []
        for i in raw.get("ideas", []):
            dp = i.get("diamond_score_partial", {})
            ideas.append(Idea(
                title=i.get("title", ""),
                domain=i.get("domain", "hardware"),
                problem=i.get("problem", ""),
                physical_limit=i.get("physical_limit", ""),
                proposed_direction=i.get("proposed_direction", ""),
                company_context=i.get("company_context"),
                diamond_score_partial=DiamondScorePartial(
                    physics_feasibility=float(dp.get("physics_feasibility", 0)),
                    market_pain=0.0,
                    novelty=0.0,
                    scalability=0.0,
                ),
            ))

        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=findings,
            ideas=ideas,
        )
