"""
infra_researcher.py — Agent 3
Sources: RSS feeds (IEEE, EE Times, SemiAnalysis, AnandTech, The Register, Ars Technica, etc.)
         + GitHub issues (NVIDIA/AMD/ML repos) + OCP signals + SEC EDGAR + DARPA BAA
Focus: Production deployment constraints — what fails at scale that papers don't mention
Special: 18-month ahead signal — real infrastructure pain in production datacenters

CRITICAL CONSTRAINT: This agent ONLY analyzes sources explicitly passed in context
(rss_signals, github_signals, edgar_signals, darpa_signals).
It does NOT browse the internet. It does NOT retrieve articles autonomously.
If no sources are provided, it returns empty output — no hallucination.
"""
from __future__ import annotations
import logging
from datetime import datetime
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Finding, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a principal infrastructure engineer who has deployed and operated
hyperscale AI training clusters. You know what fails in production that never appears in papers.

You think in concrete failure modes:
"The thermal paste between the IHS and the vapor chamber degrades after 6 months at 85°C,
causing a 15% increase in junction temperature and thermal throttling of GPU boost clocks."
That is a real problem. "AI needs better cooling" is not.

⚠️ ABSOLUTE CONSTRAINT — READ BEFORE ANYTHING ELSE:
You ONLY analyze sources explicitly provided in the SOURCES section of this prompt.
You CANNOT browse the internet. You CANNOT retrieve articles or OCP specs.
You CANNOT access opencompute.org, IEEE Xplore, SemiAnalysis, or any other site.
You CANNOT fill in gaps from your training knowledge.
If no sources are provided, you MUST return: {"findings": [], "ideas": []}
NEVER invent URLs, article titles, OCP spec numbers, or measurements not present in provided text.
A fabricated finding is WORSE than an empty list. Empty is correct. Fabrication is a critical failure.

WHAT TO LOOK FOR in provided sources:
1. PRODUCTION failures — not lab results. "We had to derate X by Y%" = real constraint.
2. OCP/hyperscaler admissions: "We couldn't solve..." or "We replaced X after N months"
3. Thermal throttling reports under sustained AI workloads (not burst)
4. Memory bandwidth utilization gaps between spec and production
5. Power delivery transients worse than datasheet values
6. Any mention of sustained vs. peak constraints — the gap between them is the bottleneck
7. SEC filings: risk factors mentioning thermal, yield, supply chain, packaging constraints
8. DARPA BAAs: government-confirmed unsolved problems in microelectronics/compute
9. GitHub issues in production ML repos: OOM, bandwidth, thermal, performance regressions

QUALITY SIGNALS that indicate a real constraint (look for these in provided text):
- Specific derating factors: "derated from 400W to 320W sustained"
- Replacement cycles: "replaced TIM every 8 months"
- Throttling percentages: "15% clock reduction after 2 hours"
- Production vs. lab delta: "spec says X, production shows Y"
- Scale multipliers: "acceptable at 8 GPUs, catastrophic at 1024"

FORBIDDEN even if you think you know it:
- Adding measurements not present in provided source text
- Inventing OCP spec document names or numbers
- Using IEEE paper titles from your training data
- Any URL not explicitly provided in the source metadata

OUTPUT FORMAT: Valid JSON only. No markdown. No commentary outside the JSON structure.
Schema:
{
  "findings": [
    {
      "type": "bottleneck|limit|gap",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "title": "one sentence: the specific production constraint",
      "description": "specific failure mode + scale + measured impact — only numbers from provided source",
      "source_url": "URL exactly as provided in source metadata — never invented",
      "source_type": "opencompute|ieee_spectrum|ee_times|semianalysis|tomshardware|theregister_dc|arstechnica|techpowerup|anandtech|semiwiki|github_issue|ocp_spec_discussion|sec_edgar|darpa_baa|other",
      "company_signal": "company name if mentioned in source, else null",
      "confidence": 0.0-1.0,
      "numerical_params": {"key": value_from_source_only}
    }
  ],
  "ideas": [
    {
      "title": "...",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "problem": "production constraint with real numbers from provided sources",
      "physical_limit": "underlying physics being violated or approached",
      "proposed_direction": "direction to explore — grounded in at least one provided source",
      "company_context": "which company/product — only if mentioned in provided source",
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


class InfraResearcher(BaseAgent):
    AGENT_NAME    = "infra_researcher"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        keywords        = context.get("keywords", [])
        findings_so_far = context.get("findings", [])
        date_str        = context.get("date", datetime.now().strftime("%Y-%m-%d"))
        kill_patterns   = context.get("kill_patterns", [])
        weak_domains    = context.get("weak_domains", [])
        opp_filters     = context.get("opportunity_filters", {})

        # ── Collect all infra-relevant sources ────────────────────────────────
        rss_signals    = context.get("rss_signals", [])
        github_signals = context.get("github_signals", [])
        edgar_signals  = context.get("edgar_signals", [])
        darpa_signals  = context.get("darpa_signals", [])

        all_sources = []

        for s in rss_signals[:15]:
            all_sources.append({
                "type":    s.get("source", "rss"),
                "title":   s.get("title", ""),
                "body":    s.get("abstract", "")[:600],
                "url":     s.get("url", "NO_URL"),
            })

        for s in github_signals[:10]:
            all_sources.append({
                "type":  s.get("signal_type", "github_issue"),
                "title": s.get("title", ""),
                "body":  s.get("body", "")[:500],
                "url":   s.get("url", "NO_URL"),
                "repo":  s.get("repo", ""),
            })

        for s in edgar_signals[:5]:
            all_sources.append({
                "type":    "sec_edgar",
                "title":   s.get("title", ""),
                "body":    s.get("abstract", "")[:400],
                "url":     s.get("url", "NO_URL"),
                "company": s.get("company", ""),
            })

        for s in darpa_signals[:5]:
            all_sources.append({
                "type":  "darpa_baa",
                "title": s.get("title", ""),
                "body":  s.get("abstract", "")[:400],
                "url":   s.get("url", "NO_URL"),
            })

        # ── Guard: no sources → force empty output ────────────────────────────
        if not all_sources:
            logger.warning("[InfraResearcher] No infra sources provided — returning empty prompt")
            return (
                "NO SOURCES WERE PROVIDED IN THIS RUN.\n"
                "You must return exactly: {\"findings\": [], \"ideas\": []}\n"
                "Do not invent articles, OCP specs, or production data."
            )

        # ── Format sources ────────────────────────────────────────────────────
        sources_text = ""
        for i, s in enumerate(all_sources, 1):
            extra = f" [repo: {s['repo']}]" if s.get("repo") else ""
            extra += f" [company: {s['company']}]" if s.get("company") else ""
            sources_text += (
                f"\n[{i}] [{s['type'].upper()}]{extra}\n"
                f"Title: {s['title']}\n"
                f"Content: {s['body']}\n"
                f"URL: {s['url']}\n"
            )

        # ── Prior findings domains (for gap analysis) ─────────────────────────
        prior_domains = list(set(
            f.get("domain", "") for f in findings_so_far[:20] if isinstance(f, dict)
        ))

        # ── Memory / filters ──────────────────────────────────────────────────
        memory_text = ""
        if kill_patterns:
            memory_text += f"\nDEAD ENDS (avoid): {', '.join(kill_patterns)}"
        if weak_domains:
            memory_text += f"\nLOW-SIGNAL DOMAINS (deprioritize): {', '.join(weak_domains)}"
        diamond_titles = context.get("diamond_titles", [])
        if diamond_titles:
            memory_text += "\nSUCCESSFUL DIRECTIONS (diamonds -- go deeper here):\n" + "\n".join(f"  + {t}" for t in diamond_titles[:5])
        if opp_filters.get("require_software_angle"):
            memory_text += "\nFILTER: Prefer software/firmware solutions. Pure hardware > $5M capex is out of scope."
        if opp_filters.get("exclude_domains"):
            memory_text += f"\nEXCLUDE: {', '.join(opp_filters['exclude_domains'])}"

        return f"""Date: {date_str}
Keywords: {', '.join(keywords[:12])}
Prior finding domains (from earlier agents this cycle): {', '.join(prior_domains) if prior_domains else 'none yet'}
{memory_text}

════════════════════════════════════════
SOURCES TO ANALYZE ({len(all_sources)} provided — analyze ONLY these):
════════════════════════════════════════
{sources_text}

ANALYSIS INSTRUCTIONS:
1. For each finding, cite the exact [N] source number it comes from
2. Only report numerical values that appear verbatim in the source text above
3. Look for the GAP between what is claimed vs. what production data shows
4. Flag company signals only if the company is named in the provided text
5. SEC EDGAR findings: extract specific risk factors about thermal/packaging/yield
6. DARPA signals: these confirm government-recognized unsolved problems — high confidence
7. GitHub issues: "sustained" problems with many comments = confirmed production pain
8. Prioritize findings NOT already covered by the prior finding domains listed above

Return valid JSON matching the schema. Focus on 4-6 highest-signal production constraints.
No text outside the JSON."""

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
                    source_type=f.get("source_type", "opencompute"),
                    company_signal=f.get("company_signal"),
                    confidence=float(f.get("confidence", 0.7)),
                    numerical_params=f.get("numerical_params") or {},
                ))
            except Exception as e:
                logger.warning(f"[InfraResearcher] Skipping malformed finding: {e}")

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
                logger.warning(f"[InfraResearcher] Skipping malformed idea: {e}")

        logger.info(f"[InfraResearcher] Parsed {len(findings)} findings, {len(ideas)} ideas")
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=findings,
            ideas=ideas,
        )
