"""
infra_researcher.py — Agent 3
Sources: OpenCompute Project, IEEE Xplore, SemiAnalysis, AnandTech, ServeTheHome
Focus: Production deployment constraints — what fails at scale that papers don't mention
Special: 18-month ahead signal — real infrastructure pain in production datacenters
"""
from __future__ import annotations
import logging
from datetime import datetime
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Finding, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a principal infrastructure engineer who has deployed and operated
hyperscale AI training clusters. You know what fails in production that never appears in papers.

You think in concrete failure modes: "The thermal paste between the IHS and the vapor chamber
degrades after 6 months at 85°C, causing a 15% increase in junction temperature and thermal
throttling of GPU boost clocks." That's a real problem. "AI needs better cooling" is not.

TARGET SOURCES (in priority order):
1. OPEN COMPUTE PROJECT (opencompute.org)
   - Technical specifications that describe real constraints
   - OCP Summit presentations: engineers admitting what they couldn't solve
   - Hardware specs that reveal bottlenecks: rack power limits, cooling capacity, interconnect

2. IEEE XPLORE — infrastructure papers (not pure research):
   - "Challenges in deploying...", "Lessons learned from...", "Operational experience with..."
   - Data center thermal management in practice vs. theory
   - Power delivery at rack and PDU scale for AI workloads

3. SEMIANALYSIS / ANANDTECH / SERVETHEHOME technical articles:
   - Real measurements of deployed hardware
   - Thermal performance under sustained AI workloads
   - Memory subsystem bottlenecks in production

4. HYPERSCALER TECHNICAL BLOGS:
   - Google: cloud.google.com/blog, research.google
   - Meta: engineering.fb.com
   - Microsoft: techcommunity.microsoft.com
   - Amazon: aws.amazon.com/blogs (re:Invent papers)

EXTRACTION RULES:
1. Focus on PRODUCTION failures, not lab failures
2. "We had to derate..." = real constraint, extract the derating factor
3. "We replaced X with Y after N months" = X failed under real conditions
4. OCP rack specs: max power per rack, max inlet temperature, thermal budget
5. Any mention of "thermal throttling at scale" = confirmed thermal bottleneck
6. Memory bandwidth utilization statistics from production workloads = real roofline data
7. Production di/dt transients are always worse than datasheets — flag when this is mentioned

FORBIDDEN:
- Marketing white papers (no measurements)
- Simulated results presented as production data
- Papers older than 24 months unless they describe a problem still unsolved today

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "findings": [
    {
      "type": "bottleneck|limit|gap",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "title": "one sentence: the production constraint",
      "description": "specific failure mode + scale + measured impact + source",
      "source_url": "url",
      "source_type": "opencompute|forum|other",
      "company_signal": "company name or null",
      "confidence": 0.0-1.0,
      "numerical_params": {"measured_values": numbers_from_source}
    }
  ],
  "ideas": [
    {
      "title": "...",
      "domain": "thermal|power|data_movement|hardware|pdn|cross_domain",
      "problem": "production constraint with real numbers",
      "physical_limit": "underlying physics being violated or approached",
      "proposed_direction": "direction to explore",
      "company_context": "which company / which product in production",
      "diamond_score_partial": {
        "physics_feasibility": 0-10,
        "market_pain": 0,
        "novelty": 0,
        "scalability": 0
      }
    }
  ]
}"""


class InfraResearcher(BaseAgent):
    AGENT_NAME    = "infra_researcher"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        keywords     = context.get("keywords", [])
        findings_so_far = context.get("findings", [])
        date_str     = context.get("date", datetime.now().strftime("%Y-%m-%d"))

        prior_domains = list(set(f.get("domain", "") for f in findings_so_far[:20]))

        return f"""Date: {date_str}
Search keywords: {', '.join(keywords[:10])}
Prior findings domains: {', '.join(prior_domains) if prior_domains else 'none yet'}

SEARCH TASKS:
1. OpenCompute Summit {datetime.now().year - 1}/{datetime.now().year} presentations: what thermal/power problems were admitted?
2. IEEE papers on production AI cluster thermal management (2023-2025)
3. Hyperscaler engineering blog posts about infrastructure limits they hit
4. OCP hardware specs: what are the actual rack-level thermal and power constraints?
5. Production HBM reliability data: what fails first and at what temperature?

CRITICAL: Look for the GAP between what papers claim and what production data shows.
Example: A paper claims a cooling solution handles 300W/chip. 
OCP deployment data shows sustained AI workloads throttle at 250W after 2 hours.
That 50W gap = real bottleneck.

For any numerical constraint found:
- Convert to standard units (W/cm², °C, GB/s, mV)
- Note if this is a sustained vs. peak constraint
- Note if the constraint was tighter in production than in lab

Return valid JSON. Focus on 4-6 production constraints not covered by academic papers."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        findings = []
        for f in raw.get("findings", []):
            findings.append(Finding(
                type=f.get("type", "bottleneck"),
                domain=f.get("domain", "hardware"),
                title=f.get("title", ""),
                description=f.get("description", ""),
                source_url=f.get("source_url"),
                source_type=f.get("source_type", "opencompute"),
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
