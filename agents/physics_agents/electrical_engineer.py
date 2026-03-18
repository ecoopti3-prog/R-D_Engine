"""
electrical_engineer.py — Agent 9
Physics validation specialist: power, PDN, electrical limits.
Uses extracted PowerParams + PDNParams to evaluate how close an idea is to fundamental electrical limits.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a principal electrical engineer who has designed PDNs and power delivery systems
for advanced AI accelerators at TSMC 5nm and below.

You know the REAL electrical limits — not the textbook ones. You know that:
- Landauer limit (0.0174 aJ at 300K) is irrelevant; the real kill is supply noise at <0.8V
- IR drop > 5% of VDD = performance degradation, not just warning
- Electromigration current density limit for copper at 100°C = ~1 mA/μm²
- PDN impedance at 1 GHz must be below target_voltage * 5% / peak_di_dt
- Decoupling capacitor resonance can make PDN worse at specific frequencies

YOUR JOB: For each idea, evaluate the physics score from the ELECTRICAL perspective.
You receive ideas that have already passed or have no electrical parameters yet.

EVALUATION FRAMEWORK (assign a score 0-10 and justify):

SCORE 10: Solution operates at 10-40% below a known fundamental electrical limit.
         The gap is exploitable. The math closes. No second-order effects that kill it.

SCORE 8: Solution approaches a known limit (40-70% of limit). Gap exists but narrowing.
         Could work if the specific constraint mentioned is addressed.

SCORE 6: Limit is known but the solution's position relative to it is unclear.
         Missing one or two critical numerical values to verify.

SCORE 4: Electrical feasibility is speculative. Physics is plausible but unverified.
         No hard numbers, only claims.

SCORE 2: Known electrical limit is being approached too closely.
         Second-order effects (contact resistance, parasitic inductance) likely kill this.

SCORE 0: Violates a confirmed electrical limit. IR drop math doesn't close.
         Electromigration at proposed current density = chip failure in 6 months.

KEY CHECKS TO RUN (mentally):
1. IR drop: V_drop = I × R_pdn. Must be < 5% of VDD.
2. Landauer limit: E_per_bit = kT × ln(2) = 0.0174 aJ at 300K. No operation can cost less.
3. CMOS power density: P = α × C × V² × f. At 3nm, typical switching power = 50-100 W/cm².
4. PDN impedance: Z_target = (VDD × allowed_droop%) / di_dt. Verify if feasible.
5. Electromigration: J_max for Cu interconnect ≈ 1 mA/μm² at 100°C (JEDEC JEP122).

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "electrical_assessments": [
    {
      "idea_id": "string",
      "physics_score": 0-10,
      "reasoning": "specific calculation or limit cited with numbers",
      "limiting_factor": "which electrical constraint is binding",
      "gap_to_limit_pct": 0-100,
      "flags": ["list of concerns that need verification"],
      "kill_recommendation": false
    }
  ]
}"""


class ElectricalEngineer(BaseAgent):
    AGENT_NAME    = "electrical_engineer"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas = context.get("ideas", [])
        ideas_text = ""
        for i in ideas:
            pp  = i.get("power_params") or {}
            pdn = i.get("pdn_params") or {}
            ideas_text += (
                f"\nID: {i.get('id','?')}\n"
                f"Title: {i.get('title','')}\n"
                f"Problem: {i.get('problem','')}\n"
                f"Physical limit: {i.get('physical_limit','')}\n"
                f"Domain: {i.get('domain','')}\n"
                f"Power params: watt={pp.get('watt')}, tdp={pp.get('tdp_watt')}, "
                f"density={pp.get('power_density_w_cm2')} W/cm², "
                f"energy={pp.get('energy_per_op_pj')} pJ/op, "
                f"V={pp.get('voltage_v')}, I={pp.get('current_a')} A\n"
                f"PDN params: IR={pdn.get('ir_drop_mv')} mV, VDD={pdn.get('vdd_v')} V, "
                f"Z={pdn.get('pdn_impedance_mohm')} mΩ, "
                f"di/dt={pdn.get('di_dt_a_per_ns')} A/ns, "
                f"decap={pdn.get('decap_nf')} nF\n"
                "---"
            )

        return f"""IDEAS FOR ELECTRICAL PHYSICS ASSESSMENT:
{ideas_text}

For each idea with electrical/PDN parameters, evaluate:
1. Does the IR drop budget close at the stated VDD and current?
2. Is the power density within CMOS limits for the implied process node?
3. Is the energy per operation above the Landauer limit?
4. Are PDN impedance targets physically achievable with known decap densities?

For ideas WITHOUT electrical parameters: score them 4.0 (speculative) and flag the missing values.

IMPORTANT: Use the FULL idea id string exactly as shown above in all id fields.

Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for a in raw.get("electrical_assessments", []):
            idea = Idea(
                id=a.get("idea_id", ""),
                title=f"Electrical assessment for {a.get('idea_id','?')[:8]}",
                domain="power",
                problem="Electrical physics assessment",
                physical_limit=a.get("limiting_factor", ""),
                proposed_direction="",
                diamond_score_partial=DiamondScorePartial(
                    physics_feasibility=float(a.get("physics_score", 4.0)),
                    market_pain=0.0,
                    novelty=0.0,
                    scalability=0.0,
                ),
            )
            idea.physics_kill_detail = (
                a.get("reasoning", "")
                if a.get("kill_recommendation") else None
            )
            ideas.append(idea)

        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            ideas=ideas,
            metadata={
                "assessments": raw.get("electrical_assessments", [])
            },
        )
