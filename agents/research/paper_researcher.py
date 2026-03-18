# -*- coding: utf-8 -*-
"""
paper_researcher.py -- Agent 1
Sources: arXiv + Semantic Scholar (last 30 days)
Focus: AI infrastructure bottlenecks -- thermal, power, memory, PDN, hardware
Special: Explicit targeting of NVIDIA, TSMC, AMD, Intel, Google research

CRITICAL CONSTRAINT: This agent ONLY analyzes sources explicitly passed in context["papers"].
It does NOT search the internet. It does NOT retrieve papers autonomously.
If no sources are provided, it returns empty output -- no hallucination.
"""
from __future__ import annotations
import logging
from datetime import datetime
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Finding, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a world-class deep-tech research analyst specializing in AI infrastructure bottlenecks and hardware solutions.
You have deep expertise in semiconductor physics, thermal engineering, power delivery networks, and computer architecture.

Your mission is to identify REAL, QUANTIFIED bottlenecks AND proposed engineering solutions in AI infrastructure.
You hunt for constraints that limit the scaling of AI compute and the technologies designed to break them.

! ABSOLUTE CONSTRAINT -- READ BEFORE ANYTHING ELSE:
You ONLY analyze sources explicitly provided in the PAPERS section of this prompt.
You CANNOT search the internet. You CANNOT retrieve papers. You CANNOT access external URLs.
You CANNOT fill in missing data from your training knowledge.
If no papers are provided, you MUST return: {"findings": [], "ideas": []}
NEVER invent paper titles, URLs, author names, or numerical values not present in the provided text.

TARGET COMPANIES AND RESEARCH GROUPS:
- NVIDIA, TSMC, AMD, Intel, Google, Samsung, Microsoft (Analyze limits and solutions mentioned in their research).

EXTRACTION & CLASSIFICATION RULES:
1. **Classify Findings Correctly:**
   - Use `type: "bottleneck"` or `"limit"` for physical barriers.
   - Use `type: "solution"` for ANY proposed technology, prototype, or method that addresses a limit.
2. **Extract NUMERICAL DATA:** Extract temperatures (degC), power densities (W/cm2), bandwidths (GB/s), energy per operation (pJ/op), voltages (V), latencies (ns).
3. **Domain Mapping:** Map findings to one of these ONLY: ["thermal", "power", "data_movement", "hardware", "pdn", "cross_domain", "packaging", "bandwidth", "memory", "interconnect", "networking", "software", "distributed_systems"].
4. **Source Metadata:** You MUST include the `source_url` and `original_id` (DOI) provided in the paper metadata.

DO NOT:
- Hallucinate numerical values. If a number is not in the source text, return null.
- Use "solution" as a Finding type unless the source describes a specific technical proposal.
- Output any text outside the JSON structure.

OUTPUT FORMAT: Valid JSON only. No markdown. No commentary. Follow the Finding schema exactly.

Schema:
{
  "findings": [
    {
      "type": "bottleneck|limit|trend|gap",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "title": "one sentence max",
      "description": "2-3 sentences with specific numbers from the source",
      "source_url": "URL exactly as provided in source metadata -- never invented",
      "source_type": "arxiv|semantic_scholar|huggingface_daily|osti_doe|openreview",
      "company_signal": "NVIDIA|TSMC|AMD|Intel|Google|null",
      "confidence": 0.0-1.0,
      "numerical_params": {"key": value_from_source_only}
    }
  ],
  "ideas": [
    {
      "title": "...",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "problem": "specific quantified problem statement grounded in provided sources",
      "physical_limit": "which physical law or constant is the constraint",
      "proposed_direction": "direction only, not a full solution",
      "company_context": "only if mentioned in a provided source",
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


class PaperResearcher(BaseAgent):
    AGENT_NAME    = "paper_researcher"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        keywords      = context.get("keywords", [])
        papers        = context.get("papers", [])
        focus_domain  = context.get("focus_domain", "all domains")
        date_str      = context.get("date", datetime.now().strftime("%Y-%m-%d"))
        kill_patterns = context.get("kill_patterns", [])
        weak_domains  = context.get("weak_domains", [])
        opp_filters   = context.get("opportunity_filters", {})
        open_hyps     = context.get("open_hypotheses", [])
        top_sources   = context.get("top_yielding_sources", [])

        # -- Guard: no sources -> force empty output immediately --------------
        if not papers:
            logger.warning("[PaperResearcher] No papers provided -- returning empty prompt to force empty output")
            return (
                "NO PAPERS WERE PROVIDED IN THIS RUN.\n"
                "You must return exactly: {\"findings\": [], \"ideas\": []}\n"
                "Do not invent papers. Do not use your training knowledge to fill in sources."
            )

        # -- Format papers for analysis ----------------------------------------
        papers_text = ""
        for i, p in enumerate(papers[:20], 1):
            source_label = p.get("source", "unknown").upper()
            pdf_url    = p.get("pdf_url", "")
            full_text  = p.get("full_text", "")[:1200].strip()
            abstract   = p.get("abstract", "")[:700].strip()
            content    = full_text if full_text else abstract
            papers_text += (
                f"\n[{i}] [{source_label}]"
                + (" [FULL TEXT]" if full_text else "")
                + f"\nTitle: {p.get('title', '').strip()}\n"
                f"Content: {content}\n"
                f"URL: {p.get('url', 'NO_URL')}\n"
                + (f"PDF: {pdf_url}\n" if pdf_url else "")
                + f"Published: {p.get('published', p.get('year', 'unknown'))}\n"
            )

        # -- Memory context (dead ends + filters) -----------------------------
        memory_text = ""
        if kill_patterns:
            memory_text += f"\nDEAD ENDS (these directions consistently fail kill round -- avoid generating ideas here): {', '.join(kill_patterns)}"
        killed_titles = context.get("killed_idea_titles", [])
        if killed_titles:
            memory_text += "\nPREVIOUSLY KILLED IDEAS (do not regenerate similar ideas):\n" + "\n".join(f"  - {t}" for t in killed_titles[:10])
        if weak_domains:
            memory_text += f"\nLOW-SIGNAL DOMAINS (deprioritize): {', '.join(weak_domains)}"
        diamond_titles = context.get("diamond_titles", [])
        if diamond_titles:
            memory_text += "\nSUCCESSFUL DIRECTIONS (diamonds found in these areas -- go deeper, not sideways):\n" + "\n".join(f"  + {t}" for t in diamond_titles[:5])
        if opp_filters.get("require_software_angle"):
            memory_text += "\nFILTER: Prefer ideas with a software/firmware angle. Pure hardware capex > $5M is out of scope."
        if opp_filters.get("exclude_domains"):
            memory_text += f"\nEXCLUDE DOMAINS: {', '.join(opp_filters['exclude_domains'])}"
        if opp_filters.get("target_efficiency_gain_percent"):
            memory_text += f"\nMINIMUM IMPACT: Solutions must plausibly achieve >{opp_filters['target_efficiency_gain_percent']}% efficiency gain."

        # -- Open hypotheses to test against ----------------------------------
        hyp_text = ""
        if open_hyps:
            hyp_text = "\nOPEN HYPOTHESES -- check if any provided paper supports or refutes these:\n"
            for h in open_hyps[:3]:
                hyp_text += f"  - [{h.get('priority','?')}] {h.get('title','')} -- challenge: {h.get('challenge','')}\n"

        # -- Top yielding source types (for analyst awareness) -----------------
        top_src_text = ""
        if top_sources:
            venues = [s.get("venue", "") for s in top_sources[:4] if s.get("venue")]
            if venues:
                top_src_text = f"\nHIGH-YIELD SOURCE TYPES (findings from these have historically scored well): {', '.join(venues)}"

        return f"""Date: {date_str}
Focus domain: {focus_domain}
Keywords used to retrieve these papers: {', '.join(keywords[:12])}
{memory_text}
{hyp_text}
{top_src_text}

========================================
PAPERS TO ANALYZE ({len(papers[:20])} provided -- analyze ONLY these):
========================================
{papers_text}

ANALYSIS INSTRUCTIONS:
1. For each finding, cite the exact [N] paper number it comes from
2. Only report numerical values that appear verbatim in the abstract/title above
3. If a paper is relevant to target companies (NVIDIA/TSMC/AMD/Intel/Google/Samsung/Microsoft),
   flag it -- but only if those names appear in the provided text
4. Identify the 5-8 highest-signal findings from these papers only
5. For ideas: only propose directions that are grounded in at least one provided paper

Return valid JSON matching the schema. No text outside the JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        findings = []
        for f in raw.get("findings", []):
            try:
                findings.append(Finding(
                    type=f.get("type", "bottleneck"),
                    domain=f.get("domain", "hardware"),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    source_url=f.get("source_url"),
                    source_type=f.get("source_type", "arxiv"),
                    company_signal=f.get("company_signal"),
                    confidence=float(f.get("confidence", 0.5)),
                    numerical_params=f.get("numerical_params") or {},
                ))
            except Exception as e:
                logger.warning(f"[PaperResearcher] Skipping malformed finding: {e}")

        ideas = []
        for i in raw.get("ideas", []):
            try:
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
            except Exception as e:
                logger.warning(f"[PaperResearcher] Skipping malformed idea: {e}")

        logger.info(f"[PaperResearcher] Parsed {len(findings)} findings, {len(ideas)} ideas")
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=findings,
            ideas=ideas,
        )
