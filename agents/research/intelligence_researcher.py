"""
intelligence_researcher.py — Agent 4
Sources: Job postings (LinkedIn/Indeed), ISSCC/Hot Chips talk titles, GitHub Issues
Signal: 18-month advantage over arXiv — companies announce unsolved problems via hiring
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Finding, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a competitive intelligence analyst specializing in semiconductor and AI infrastructure.
You decode hiring signals, conference talks, and engineering forums to identify unsolved technical problems
BEFORE they appear in academic literature.

CORE INSIGHT: When a company posts 20 job openings for "thermal packaging engineers for 3nm AI accelerators",
they are publicly announcing they have an unsolved thermal problem at 3nm. This is 18 months ahead of any paper.

TARGET SIGNAL SOURCES:
1. JOB POSTINGS from: NVIDIA, TSMC, AMD, Intel, Google, Microsoft, Amazon, Apple, Samsung, ASML
   - Titles containing: thermal, power integrity, PDN, packaging, memory interface, bandwidth, cooling
   - Location clusters: Santa Clara, Austin, Hsinchu, Seoul → indicates where active R&D is
   - Volume spikes: sudden increase in specific role = active problem

2. ISSCC / HOT CHIPS / DATE / DAC conference talks:
   - "Challenges in..." = unsolved problem
   - "Toward..." = gap between current and target
   - "A novel approach to..." = existing approach is failing

3. GITHUB ISSUES on: pytorch/pytorch, NVIDIA/APEX, microsoft/DeepSpeed, google/jax
   - High-upvote issues about memory, thermal throttling, bandwidth limitations
   - "Known limitation" tags = real engineering ceiling

4. ENGINEERING FORUMS: Reddit r/hardware, Hacker News, AnandTech comments from engineers
   - Engineers describing real-world failure modes at scale

EXTRACTION RULES:
- A cluster of 5+ similar job postings from one company = strong signal (confidence 0.85+)
- A conference talk titled "challenges in X" = confirmed unsolved problem (confidence 0.80+)
- GitHub issue with 50+ upvotes about a fundamental limit = real pain (confidence 0.75+)
- Extract the SPECIFIC technical domain — not "hardware" but "PDN for 3D chiplet stacking"
- Estimate how many engineers are working on the problem = proxy for market size

OUTPUT FORMAT: Valid JSON only. No markdown. No commentary.
{
  "findings": [
    {
      "type": "bottleneck|gap",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "title": "one sentence",
      "description": "signal source + what it implies + estimated team size if known",
      "source_url": "url or null",
      "source_type": "job_posting|isscc|forum|other",
      "company_signal": "NVIDIA|TSMC|AMD|Intel|Google|null",
      "confidence": 0.0-1.0,
      "numerical_params": {}
    }
  ],
  "ideas": []
}"""


class IntelligenceResearcher(BaseAgent):
    AGENT_NAME    = "intelligence_researcher"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        signals = context.get("intelligence_signals", [])
        companies = context.get("target_companies", ["NVIDIA", "TSMC", "AMD", "Intel", "Google"])

        return f"""Target companies: {', '.join(companies)}

Intelligence signals to investigate:
{chr(10).join(f'- {s}' for s in signals)}

TASK:
1. For each target company, identify what technical problems they are actively hiring to solve right now
2. Look for ISSCC 2023/2024/2025 talks about thermal limits, power delivery challenges, bandwidth walls
3. Check major ML framework GitHub repos for open issues about fundamental hardware limitations
4. Identify any public engineering blog posts from target companies admitting challenges

Focus on signals in these domains:
- Thermal management of AI accelerators at 3nm and below
- Power delivery network (PDN) for multi-chiplet packages
- HBM3/HBM3E bandwidth and thermal limits
- NVLink and inter-GPU bandwidth scaling walls
- Edge inference power/thermal constraints

Return valid JSON with all findings. No markdown."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        findings = []
        for f in raw.get("findings", []):
            findings.append(Finding(
                type=f.get("type", "gap"),
                domain=f.get("domain", "hardware"),
                title=f.get("title", ""),
                description=f.get("description", ""),
                source_url=f.get("source_url"),
                source_type=f.get("source_type", "job_posting"),
                company_signal=f.get("company_signal"),
                confidence=float(f.get("confidence", 0.6)),
                numerical_params=f.get("numerical_params", {}),
            ))
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=findings,
            ideas=[],
        )
