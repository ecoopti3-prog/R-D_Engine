"""
systems_architect.py — Agent 11
Cross-domain physics integrator. Identifies constraint interactions that single-domain agents miss.
The "second opinion" — catches ideas that pass individual physics checks but fail at system level.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a distinguished systems architect with 20 years of experience designing
AI accelerators and data center infrastructure. You see constraints others miss because you think
at the system level, not the component level.

YOUR UNIQUE VALUE: You find the "constraint coupling" problems:
- An idea improves thermal performance by 30% but increases PDN complexity by 60% → net loss
- A bandwidth solution requires 3× more power → thermal wall gets hit first
- A chiplet integration approach solves electrical isolation but creates new thermal hot spots
- A cooling improvement works at chip level but the rack-level power distribution doesn't scale

SYSTEM-LEVEL CHECKLIST (apply to every idea):

1. THERMAL-POWER COUPLING: If power increases to solve bandwidth, does thermal wall get hit?
   Rule: Power density × thermal resistance = ΔT. Check if both move in the right direction.

2. PDN-THERMAL COUPLING: Higher current density → more heat from I²R losses in PDN.
   A PDN solution that reduces impedance by adding copper also changes thermal resistance.

3. BANDWIDTH-POWER TRADE: Higher bandwidth usually means more I/O drivers, more power.
   SerDes power per Gbps: ~5-15 mW/Gbps for modern interfaces. Scale accordingly.

4. SCALING ANALYSIS: Does the physics get better or worse when you go from 1 die to a 64-die system?
   Thermal: Power density stays same but cooling infrastructure must scale.
   PDN: Voltage regulation across 64 chiplets = 64× harder.
   Bandwidth: Bisection bandwidth becomes the new bottleneck.

5. MANUFACTURING YIELD: Exotic materials (diamond, graphene) have yield problems at scale.
   A solution that requires 99% yield for each of 64 chiplets has system yield of 0.99^64 = 52%.

6. RELIABILITY UNDER AI WORKLOADS: AI training runs 24/7 at maximum utilization.
   MTTF calculations assume 30-50% utilization. At 100%: all failure modes accelerate.
   Electromigration: current duration matters. Thermal fatigue: cycle count matters.

7. INTEGRATION REALITY: Does this require a new manufacturing process or can it use existing fabs?
   If it requires a new fab process: add 5 years and $2B to the timeline.

SCORING OVERRIDE: You can RAISE or LOWER the physics score by ±2 points based on system-level effects.
If you lower it below 3: recommend kill with justification.

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "system_assessments": [
    {
      "idea_id": "string",
      "physics_score_adjustment": -2 to +2,
      "final_physics_score": 0-10,
      "coupling_problems": ["thermal-power: ...", "pdn-thermal: ..."],
      "scaling_verdict": "scales_well|scales_poorly|untested",
      "manufacturing_readiness": "existing_process|requires_new_process|exotic_materials",
      "reliability_concern": "low|medium|high",
      "kill_recommendation": false,
      "kill_reason": "null or specific reason",
      "system_level_opportunity": "any cross-domain opportunity identified"
    }
  ]
}"""


class SystemsArchitect(BaseAgent):
    AGENT_NAME    = "systems_architect"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas = context.get("ideas", [])
        physics_assessments = context.get("physics_assessments", {})

        ideas_text = ""
        for i in ideas:
            idea_id = i.get("id", "?")
            idea_id_short = idea_id[:8]   # only for display/lookup in assessments dict
            elec  = physics_assessments.get(f"electrical_{idea_id_short}", {})
            therm = physics_assessments.get(f"thermal_{idea_id_short}", {})

            pp  = i.get("power_params") or {}
            pdn = i.get("pdn_params") or {}
            tp  = i.get("thermal_params") or {}
            dm  = i.get("data_movement_params") or {}

            ideas_text += (
                f"\nID: {idea_id}\n"
                f"Title: {i.get('title','')}\n"
                f"Domain: {i.get('domain','')}\n"
                f"Problem: {i.get('problem','')[:200]}\n"
                f"Direction: {( i.get('proposed_direction') or '' )[:150]}\n"
                f"Physics score so far: {i.get('diamond_score_partial',{}).get('physics_feasibility',4)}\n"
                f"Power: {pp.get('watt') or pp.get('tdp_watt')} W, "
                f"density: {pp.get('power_density_w_cm2')} W/cm²\n"
                f"Thermal: T_j={tp.get('t_junction_c')}°C, flux={tp.get('heat_flux_w_cm2')} W/cm²\n"
                f"PDN: IR={pdn.get('ir_drop_mv')} mV, di/dt={pdn.get('di_dt_a_per_ns')} A/ns\n"
                f"Bandwidth: {dm.get('bandwidth_gb_s')} GB/s, compute={dm.get('compute_tflops')} TFLOPS\n"
                "---"
            )

        return f"""IDEAS FOR SYSTEM-LEVEL PHYSICS REVIEW:
{ideas_text}

For each idea:
1. Check thermal-power coupling: if the idea touches power, check if thermal becomes the new bottleneck
2. Check PDN-thermal coupling: higher current = more I²R heat in interconnects
3. Evaluate if the physics gets better or worse at 10× scale
4. Identify manufacturing readiness (existing TSMC process vs. exotic)
5. Assess reliability under 100% utilization AI workloads

Be specific about the coupling. "Thermal-power coupling" without a calculation is not useful.
Calculate or estimate: if power = X watts and R_th = Y °C/W, then ΔT = X×Y °C.

IMPORTANT: Use the FULL idea id string exactly as shown above in all id fields.

Return valid JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        kills = []
        from core.schemas import Kill

        for a in raw.get("system_assessments", []):
            idea = Idea(
                id=a.get("idea_id", ""),
                title=f"Systems assessment for {a.get('idea_id','?')[:8]}",
                domain="cross_domain",
                problem="System-level physics assessment",
                physical_limit=", ".join(a.get("coupling_problems", [])),
                proposed_direction=a.get("system_level_opportunity", ""),
                diamond_score_partial=DiamondScorePartial(
                    physics_feasibility=float(a.get("final_physics_score", 4.0)),
                    market_pain=0.0,
                    novelty=0.0,
                    scalability=0.0,
                ),
            )
            ideas.append(idea)

            if a.get("kill_recommendation") and a.get("kill_reason"):
                kills.append(Kill(
                    idea_id=a.get("idea_id", ""),
                    killed_by="systems_architect",
                    reason=a.get("kill_reason", ""),
                    kill_category="physics_impossible",
                ))

        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            ideas=ideas,
            kills=kills,
            metadata={"assessments": raw.get("system_assessments", [])},
        )
