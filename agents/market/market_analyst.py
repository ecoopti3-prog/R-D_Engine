"""
market_analyst.py — Agent 12
Evaluates market pain and buyer identification for surviving ideas.
Produces market_pain score (0-10) for Diamond Score computation.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a deep-tech market analyst who has evaluated hundreds of semiconductor
and AI infrastructure opportunities. You think like a venture partner at a top-tier fund that
specifically invests in deep-tech infrastructure.

YOUR CONVICTION: The best deep-tech opportunities are where:
1. A problem is CONFIRMED (not speculative) — the giants are spending money on it today
2. The buyer is IDENTIFIED — someone has a budget line item for this
3. The cost is QUANTIFIED — we can calculate the ROI precisely

MARKET SCORING FRAMEWORK (0-10):

SCORE 10 = "Every hyperscaler is bleeding money on this today"
  Evidence: NVIDIA/AMD/Google/Microsoft/Amazon all have teams working on this.
  Quantified cost: can calculate $ lost per hour of thermal throttling.
  Decision maker: VP of Infrastructure. Budget already allocated.

SCORE 8 = "Clear buyer, large market, measurable pain"
  Evidence: At least 2 major players confirmed spending. Market size calculable.
  Example: HBM thermal management — every HBM-enabled chip has this problem.
  Cost model: power cost + chip derating cost can be estimated.

SCORE 6 = "Identifiable domain with likely buyers but not confirmed spending"
  The problem domain is clear (e.g., edge inference cooling) but buyer budgets uncertain.
  Market size estimated from proxy data (e.g., edge device shipments × cost premium).

SCORE 4 = "Plausible market but no clear buyer"
  The problem exists technically. Who pays for the solution is unclear.
  Multiple potential buyers, none confirmed, none with obvious budget.

SCORE 2 = "Speculative market"
  Market depends on a future technology reaching commercial maturity.
  Example: "When quantum computing scales, this cooling approach will matter"

SCORE 0 = "No identifiable market"
  Technically interesting but no one will write a check for this.

BUYER TAXONOMY for AI infrastructure:
- TIER 1 (largest checks): AWS, Azure, GCP, Oracle Cloud — infrastructure capex
- TIER 2: NVIDIA, AMD, Intel, Google — chip design + packaging R&D
- TIER 3: TSMC, Samsung, Micron — process + manufacturing R&D
- TIER 4: Meta, Apple, Tesla — custom silicon teams
- TIER 5: OEMs (Dell, HPE, Supermicro) — server thermal solutions

ROI CALCULATION METHOD:
1. GPU cost: H100 SXM5 ≈ $30,000-40,000
2. Revenue per GPU-hour (cloud): ~$2-4/hr
3. If thermal throttling reduces throughput 10%: $0.20-0.40/hr revenue lost
4. For 1,000 GPU cluster: $200-400/hr, ~$1.7M-3.5M/year in lost revenue
5. A solution that eliminates thermal throttling → worth up to 10% of that = $170K-350K/year/cluster

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "market_assessments": [
    {
      "idea_id": "string",
      "market_pain_score": 0-10,
      "primary_buyer": "company tier + specific team",
      "secondary_buyers": ["list"],
      "market_size_usd_bn": estimated_TAM_or_null,
      "roi_model": "one sentence quantified ROI estimate",
      "roi_at_10pct_adoption": "$ amount",
      "roi_at_50pct_adoption": "$ amount",
      "budget_evidence": "what confirms this buyer has budget for this",
      "time_to_market_years": estimated_years,
      "reasoning": "2-3 sentences"
    }
  ]
}"""


class MarketAnalyst(BaseAgent):
    AGENT_NAME    = "market_analyst"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas = context.get("ideas", [])
        findings = context.get("findings", [])

        # Company signals from findings = confirmation that buyers exist
        company_signals = list(set(
            f.get("company_signal", "") for f in findings if f.get("company_signal")
        ))

        ideas_text = ""
        for i in ideas:
            ideas_text += (
                f"\nID: {i.get('id','?')}\n"
                f"Title: {i.get('title','')}\n"
                f"Domain: {i.get('domain','')}\n"
                f"Problem: {i.get('problem','')[:200]}\n"
                f"Company context: {i.get('company_context','none')}\n"
                f"Direction: {( i.get('proposed_direction') or '' )[:150]}\n"
                "---"
            )

        return f"""COMPANY SIGNALS FROM RESEARCH (confirmed buyers):
{', '.join(company_signals) if company_signals else 'None confirmed yet'}

IDEAS TO EVALUATE FOR MARKET PAIN:
{ideas_text}

For each idea:
1. Who is the primary buyer? Be specific (e.g., "NVIDIA packaging engineering team" not "hardware companies")
2. What is the annual cost of NOT solving this problem? Use the GPU cost/revenue model.
3. What evidence from the findings confirms this buyer has budget for this?
4. Is the market large enough to justify a company? (Need TAM > $1B for VC interest, $100M+ for bootstrapped)
5. What is the time-to-market? Physics validated ideas still need 2-5 years to commercialize.

Be conservative. Do not score market_pain > 6 unless there is confirmed company spending evidence.

IMPORTANT: Use the FULL idea id string exactly as shown above in all id fields.

Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for a in raw.get("market_assessments", []):
            idea = Idea(
                id=a.get("idea_id", ""),
                title=f"Market assessment for {a.get('idea_id','?')[:8]}",
                domain="hardware",
                problem="Market pain assessment",
                physical_limit="",
                proposed_direction="",
                diamond_score_partial=DiamondScorePartial(
                    physics_feasibility=0.0,
                    market_pain=float(a.get("market_pain_score", 0)),
                    novelty=0.0,
                    scalability=0.0,
                ),
            )
            ideas.append(idea)

        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            ideas=ideas,
            metadata={"market_assessments": raw.get("market_assessments", [])},
        )
