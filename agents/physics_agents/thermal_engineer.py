"""
thermal_engineer.py — Agent 10
Physics validation specialist: thermal limits.
Evaluates ideas against fundamental thermal physics limits.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a principal thermal engineer who has designed cooling systems for
AI accelerators at companies like NVIDIA, Google, and TSMC. You have direct experience with
thermal failures at the 3nm node and below.

You know the REAL thermal limits:
- Copper heat spreader practical limit: ~100 W/cm² (convective cooling)
- Copper theoretical: ~300 W/cm² (requires exotic cooling)
- Diamond substrate: ~2000 W/cm² theoretical, ~500 W/cm² practical
- JEDEC T_junction_max = 125°C standard silicon, 150°C automotive
- HBM3 thermal limit: 85°C junction (constrained by solder reliability)
- Carnot COP ceiling: T_cold / (T_hot - T_cold) — nothing beats this
- Interface thermal resistance: TIM1 ~0.1°C·cm²/W, TIM2 ~0.5°C·cm²/W
- Spreading resistance for die sizes: R_spread = 1 / (2 × π × k × √(A))

SECOND-ORDER EFFECTS THAT KILL IDEAS IN PRODUCTION:
1. Thermal cycling fatigue: ΔT > 50°C per cycle → solder joint failure in 1000 cycles
2. CTE mismatch: Si (~2.6 ppm/K) vs Cu (~17 ppm/K) → stress at interfaces
3. Non-uniform heat generation in GPU dies → local hot spots 30-40°C above average
4. Thermal spreading resistance increases with smaller die area at constant power
5. Two-phase cooling (vapor chambers, heat pipes): limited by condenser capacity, not evaporator

YOUR JOB: For each idea, evaluate thermal physics score (0-10).

SCORING:
SCORE 10: Operates well within thermal limits. Gap between claimed performance and physical limit is >40%.
SCORE 8: Approaching limits (60-80% of limit). Feasible with careful engineering.
SCORE 6: At or near practical limit. Would require exotic cooling or materials.
SCORE 4: Thermal feasibility unclear — missing key temperature or heat flux data.
SCORE 2: Approaching theoretical limit. Second-order effects likely problematic.
SCORE 0: Violates a confirmed thermal law. Carnot violated. Heat flux exceeds material limit.

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "thermal_assessments": [
    {
      "idea_id": "string",
      "physics_score": 0-10,
      "reasoning": "specific calculation or physical law cited with numbers",
      "limiting_thermal_factor": "which thermal constraint is binding",
      "hotspot_risk": "low|medium|high",
      "cooling_approach_required": "air|liquid|two_phase|microfluidic|exotic",
      "kill_recommendation": false,
      "flags": ["specific thermal concerns"]
    }
  ]
}"""


class ThermalEngineer(BaseAgent):
    AGENT_NAME    = "thermal_engineer"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas = context.get("ideas", [])
        ideas_text = ""
        for i in ideas:
            tp = i.get("thermal_params") or {}
            pp = i.get("power_params") or {}
            ideas_text += (
                f"\nID: {i.get('id','?')}\n"
                f"Title: {i.get('title','')}\n"
                f"Problem: {i.get('problem','')}\n"
                f"Physical limit: {i.get('physical_limit','')}\n"
                f"Domain: {i.get('domain','')}\n"
                f"Total power: {pp.get('watt') or pp.get('tdp_watt')} W\n"
                f"Thermal: T_j={tp.get('t_junction_c')}°C, "
                f"flux={tp.get('heat_flux_w_cm2')} W/cm², "
                f"Rth={tp.get('thermal_resistance_c_per_w')} °C/W, "
                f"T_amb={tp.get('t_ambient_c')}°C, "
                f"COP_claimed={tp.get('cop_claimed')}, "
                f"material={tp.get('material')}\n"
                "---"
            )

        return f"""IDEAS FOR THERMAL PHYSICS ASSESSMENT:
{ideas_text}

For each idea with thermal parameters, evaluate:
1. Is T_junction within JEDEC limits for the stated conditions?
2. Does the claimed COP violate Carnot? (COP_max = T_cold_K / (T_hot_K - T_cold_K))
3. Is the heat flux within the material's practical limit?
4. What cooling approach is required, and is it commercially viable?
5. Are there hot-spot concerns given non-uniform power distribution in AI dies?

For ideas WITHOUT thermal parameters: score 4.0 and flag missing values.

IMPORTANT: Use the FULL idea id string exactly as shown above in all id fields.

Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for a in raw.get("thermal_assessments", []):
            idea = Idea(
                id=a.get("idea_id", ""),
                title=f"Thermal assessment for {a.get('idea_id','?')[:8]}",
                domain="thermal",
                problem="Thermal physics assessment",
                physical_limit=a.get("limiting_thermal_factor", ""),
                proposed_direction="",
                diamond_score_partial=DiamondScorePartial(
                    physics_feasibility=float(a.get("physics_score", 4.0)),
                    market_pain=0.0,
                    novelty=0.0,
                    scalability=0.0,
                ),
            )
            if a.get("kill_recommendation"):
                idea.physics_kill_detail = a.get("reasoning", "")
            ideas.append(idea)

        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            ideas=ideas,
            metadata={"assessments": raw.get("thermal_assessments", [])},
        )
