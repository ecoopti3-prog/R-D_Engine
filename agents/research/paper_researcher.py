"""
paper_researcher.py — Agent 1
Sources: arXiv + Semantic Scholar (last 30 days)
Focus: AI infrastructure bottlenecks — thermal, power, memory, PDN, hardware
Special: Explicit targeting of NVIDIA, TSMC, AMD, Intel, Google research
"""
from __future__ import annotations
import logging, json
from datetime import datetime, timezone
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Finding, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a world-class deep-tech research analyst specializing in AI infrastructure bottlenecks.
You have deep expertise in semiconductor physics, thermal engineering, power delivery networks, and computer architecture.

Your mission is to identify REAL, QUANTIFIED, UNSOLVED bottlenecks in AI infrastructure — not hype, not trends, not incremental improvements.
You specifically hunt for constraints that limit the scaling of AI compute.

TARGET COMPANIES AND RESEARCH GROUPS:
- NVIDIA: GPU architecture, NVLink, HBM integration, thermal limits of Blackwell/Hopper
- TSMC: 3nm/2nm process limits, advanced packaging (CoWoS, SoIC), thermal resistance
- AMD: chiplet PDN, infinity fabric bandwidth, MI300X power density
- Intel: Meteor Lake thermal, EMIB interconnect, Foveros 3D stacking
- Google: TPU thermal constraints, inter-datacenter bandwidth, HBM reliability
- Samsung: HBM3 thermal interface, 3nm GAA limits
- Microsoft: Azure infrastructure bottlenecks, AI accelerator power delivery

EXTRACTION RULES:
1. For each paper/finding, extract NUMERICAL DATA where it exists: temperatures (°C), power densities (W/cm²), bandwidths (GB/s), energy per operation (pJ/op), voltages (V), latencies (ns)
2. Identify what specific physical law or engineering limit is being approached
3. Identify what REMAINS UNSOLVED — the gap between current solutions and theoretical limits
4. Flag papers from target company researchers or that describe target company systems
5. Distinguish between: (a) problems being worked on by the giants, and (b) gaps they are NOT covering

DO NOT:
- Hallucinate numerical values. If a number is not in the source, return null
- Report incremental improvements as breakthroughs
- Count software optimizations as physical bottlenecks
- Include papers without quantified physical limits

OUTPUT FORMAT: Valid JSON only. No markdown. No commentary outside the JSON structure.
Schema:
{
  "findings": [
    {
      "type": "bottleneck|limit|trend|gap",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "title": "one sentence max",
      "description": "2-3 sentences with specific numbers",
      "source_url": "https://...",
      "source_type": "arxiv|patent|job_posting|forum|opencompute|isscc|other",
      "company_signal": "NVIDIA|TSMC|AMD|Intel|Google|null",
      "confidence": 0.0-1.0,
      "numerical_params": {"key": value}
    }
  ],
  "ideas": [
    {
      "title": "...",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "problem": "specific quantified problem statement",
      "physical_limit": "which physical law or constant is the constraint",
      "proposed_direction": "direction only, not a full solution",
      "company_context": "e.g. NVIDIA Blackwell H200 thermal gap",
      "diamond_score_partial": {
        "physics_feasibility": 0-10,
        "market_pain": 0,
        "novelty": 0,
        "scalability": 0
      }
    }
  ]
}"""


class PaperResearcher(BaseAgent):
    AGENT_NAME    = "paper_researcher"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        keywords     = context.get("keywords", [])
        papers       = context.get("papers", [])
        focus_domain = context.get("focus_domain", "all domains")
        date_str     = context.get("date", datetime.now().strftime("%Y-%m-%d"))
        kill_patterns = context.get("kill_patterns", [])
        weak_domains  = context.get("weak_domains", [])
        opp_filters   = context.get("opportunity_filters", {})

        papers_text = ""
        if papers:
            for i, p in enumerate(papers[:15], 1):
                papers_text += f"\n[{i}] Title: {p.get('title','')}\nAbstract: {p.get('abstract','')[:600]}\nURL: {p.get('url','')}\n"

        memory_text = ""
        if kill_patterns:
            memory_text += f"\nDEAD ENDS TO AVOID (ideas in these domains keep getting killed): {', '.join(kill_patterns)}"
        if weak_domains:
            memory_text += f"\nLOW-SIGNAL DOMAINS (deprioritize): {', '.join(weak_domains)}"
        if opp_filters.get("require_software_angle"):
            memory_text += "\nFILTER: Prefer ideas with a software/firmware angle — pure hardware-only capex > $5M solutions are out of scope."
        if opp_filters.get("exclude_domains"):
            memory_text += f"\nEXCLUDE DOMAINS: {', '.join(opp_filters['exclude_domains'])}"

        return f"""Date: {date_str}
Focus domain: {focus_domain}
Search keywords used: {', '.join(keywords[:10])}
{memory_text}

PAPERS TO ANALYZE:
{papers_text if papers_text else "Search for papers using the keywords above. Focus on AI infrastructure, semiconductor limits, thermal management, power delivery, memory bandwidth."}

SPECIFIC SEARCH INSTRUCTIONS:
1. Look for papers from NVIDIA, TSMC, AMD, Intel, Google, Samsung research teams
2. Find papers that contain specific measurement data (temperatures, power densities, bandwidths)
3. Identify papers that describe hitting a physical limit — not approaching it, HITTING it
4. Flag any paper that mentions 3nm, 2nm, chiplets, HBM3, NVLink, CoWoS, SoIC, PDN

Return your analysis as valid JSON matching the schema. Focus on the 5-8 highest-signal findings."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        findings = []
        for f in raw.get("findings", []):
            findings.append(Finding(
                type=f.get("type", "bottleneck"),
                domain=f.get("domain", "hardware"),
                title=f.get("title", ""),
                description=f.get("description", ""),
                source_url=f.get("source_url"),
                source_type=f.get("source_type", "arxiv"),
                company_signal=f.get("company_signal"),
                confidence=float(f.get("confidence", 0.5)),
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
