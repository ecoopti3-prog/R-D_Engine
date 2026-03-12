"""
cost_analyst.py — Agent 13
ROI and scalability scoring for Diamond Score computation.
Produces scalability score (0-10) and validates market_pain ROI estimates from market_analyst.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a deep-tech cost analyst and financial modeler specializing in
semiconductor infrastructure investments. You've built financial models for hardware startups
and know exactly where the numbers fall apart.

YOUR JOB: Validate and refine the scalability score and ROI estimates. Be skeptical of optimistic
market sizing. Your models should be defensible to a CFO, not just a product manager.

SCALABILITY SCORING (0-10):

SCORE 10 = "Every AI datacenter globally needs this"
  Total addressable market: all GPU/AI accelerator deployments worldwide.
  Scale: 100,000+ units/year addressable. No geographic or workload constraints.

SCORE 8 = "All major hyperscalers need this"
  Market: AWS, Azure, GCP, and hyperscalers = ~$500B+ annual capex.
  The constraint affects training AND inference at hyperscale.

SCORE 6 = "Large market segment"
  Specific but large segment: e.g., "all training clusters above 1000 GPUs".
  ~$50B+ TAM. 3-5 major customers.

SCORE 4 = "Niche but valuable"
  Specific application: e.g., "edge inference for automotive at high temperature".
  $5-50B TAM. Concentrated customer base, high switching cost.

SCORE 2 = "Narrow application"
  Very specific constraints needed: only works for one type of workload or chip.
  <$1B TAM. Single customer risk.

SCORE 0 = "No path to scale"
  Problem is too narrow, or solution requires per-customer customization.
  Not a product company — would be a consulting engagement.

COST STRUCTURE ANALYSIS:
For hardware solutions, the key question is: BOM cost vs. value delivered.
- Customer pays in one of two ways: cost reduction OR capability improvement
- Cost reduction: if solution costs $X, it must save >3X in operational costs to get purchased
- Capability improvement: must unlock new revenue or prevent revenue loss

KEY COST BENCHMARKS:
- NVIDIA H100 SXM5: ~$30,000-40,000 per chip
- HBM3 stack: ~$4,000-8,000 per stack
- Advanced packaging (CoWoS): ~$500-2,000 premium per chip
- Data center power cost: ~$0.05-0.12 per kWh (hyperscale)
- 1 MW of cooling infrastructure: ~$1-3M CapEx
- GPU server (8×H100): ~$300,000-400,000

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "cost_assessments": [
    {
      "idea_id": "string",
      "scalability_score": 0-10,
      "tam_usd_bn": estimated_TAM_or_null,
      "cost_per_unit_usd": estimated_solution_cost_or_null,
      "value_per_unit_usd": estimated_value_delivered_or_null,
      "roi_multiple": cost_value_ratio_or_null,
      "customer_acquisition_challenge": "low|medium|high",
      "manufacturing_scalability": "commodity|specialized|exotic",
      "competitive_moat": "patent|know_how|network_effects|none",
      "reasoning": "2 sentences on scalability"
    }
  ]
}"""


class CostAnalyst(BaseAgent):
    AGENT_NAME    = "cost_analyst"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas = context.get("ideas", [])
        market_data = context.get("market_assessments", {})

        ideas_text = ""
        for i in ideas:
            idea_id = i.get("id", "?")
            idea_id_short = idea_id[:8]
            mkt = market_data.get(idea_id_short, {})
            ideas_text += (
                f"\nID: {idea_id}\n"
                f"Title: {i.get('title','')}\n"
                f"Domain: {i.get('domain','')}\n"
                f"Problem: {i.get('problem','')[:200]}\n"
                f"Primary buyer: {mkt.get('primary_buyer','unknown')}\n"
                f"Market pain score: {i.get('diamond_score_partial',{}).get('market_pain',0)}\n"
                f"Market size estimate: ${mkt.get('market_size_usd_bn','?')}B\n"
                f"ROI model: {mkt.get('roi_model','none provided')}\n"
                "---"
            )

        return f"""IDEAS FOR SCALABILITY AND COST ASSESSMENT:
{ideas_text}

For each idea:
1. Validate the TAM estimate from market_analyst. Is it realistic or optimistic?
2. Estimate the likely solution cost (BOM + manufacturing premium)
3. Calculate value delivered per unit (cost saved or revenue protected)
4. What is the ROI multiple? (<3× = hard sell, >10× = strong case)
5. How scalable is the manufacturing? Can it reach 100,000 units/year?
6. What's the real moat? (IP, process know-how, network effects, or none)

Be conservative on TAM — most hardware TAMs are overstated by 3-5×.

IMPORTANT: Use the FULL idea id string exactly as shown above in all id fields.

Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for a in raw.get("cost_assessments", []):
            idea = Idea(
                id=a.get("idea_id", ""),
                title=f"Cost assessment for {a.get('idea_id','?')[:8]}",
                domain="hardware",
                problem="Cost and scalability assessment",
                physical_limit="",
                proposed_direction="",
                diamond_score_partial=DiamondScorePartial(
                    physics_feasibility=0.0,
                    market_pain=0.0,
                    novelty=0.0,
                    scalability=float(a.get("scalability_score", 0)),
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
            metadata={"cost_assessments": raw.get("cost_assessments", [])},
        )
