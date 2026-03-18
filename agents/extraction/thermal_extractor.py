"""
thermal_extractor.py — Agent 6
Extracts ThermalParams from findings. Extraction only — no evaluation, no guessing.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, ThermalParams

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precision data extraction system specializing in thermal parameters.

YOUR ONLY JOB: Extract numerical thermal values from provided text.
DO NOT calculate. DO NOT evaluate. DO NOT guess.
If a value is not explicitly stated in the source text → return null for that field.

THERMAL PARAMETERS TO EXTRACT:
- t_junction_c: Junction temperature in Celsius (look for "T_j", "junction temperature", "°C max")
- heat_flux_w_cm2: Heat flux in watts per cm² (look for "W/cm²", "heat flux", "power flux")
- thermal_resistance_c_per_w: Thermal resistance (look for "θ_JA", "θ_JC", "R_th", "°C/W", "K/W")
- delta_t_c: Temperature difference (look for "ΔT", "delta T", "temperature rise", "temperature gradient")
- t_ambient_c: Ambient temperature (look for "T_ambient", "T_air", "inlet temperature", "coolant temperature")
- cop_claimed: Claimed coefficient of performance for cooling (look for "COP", "efficiency of cooling")
- material: Thermal interface or substrate material (look for "TIM", "IHS material", "substrate", "diamond", "copper")

CONTEXT CLUES FOR THERMAL DATA:
- Packaging papers always mention junction temperature limits
- Cooling solution papers cite heat flux they handle
- JEDEC papers quote θ_JC or θ_JA values
- Chip test reports mention max operating temperature
- Thermal Interface Material papers cite thermal resistance reduction

UNIT NORMALIZATION:
- Always use Celsius (not Kelvin) for temperatures
- If Kelvin → subtract 273.15
- W/m² → W/cm² (divide by 10,000)
- K/W = °C/W (same thing)

MATERIAL NORMALIZATION: Standardize to: copper, diamond, silicon_carbide, alumina, silicon, aluminum, graphene, default

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "thermal_extractions": [
    {
      "idea_id": "string matching one of the input ideas",
      "thermal_params": {
        "t_junction_c": number_or_null,
        "heat_flux_w_cm2": number_or_null,
        "thermal_resistance_c_per_w": number_or_null,
        "delta_t_c": number_or_null,
        "t_ambient_c": number_or_null,
        "cop_claimed": number_or_null,
        "material": "string_or_null",
        "source_ref": "paper title or URL"
      }
    }
  ]
}"""


class ThermalExtractor(BaseAgent):
    AGENT_NAME    = "thermal_extractor"
    CHAIN_TYPE    = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        ideas    = context.get("ideas", [])
        findings = context.get("findings", [])

        findings_text = "\n".join(
            f"- [{f.get('domain','')}] {f.get('title','')}: {f.get('description','')[:300]}"
            for f in findings[:20]
        )
        ideas_text = "\n".join(
            f"id={i.get('id','?')} title={i.get('title','')} domain={i.get('domain','')}"
            for i in ideas[:10]
        )
        return f"""IDEAS TO EXTRACT THERMAL DATA FOR:
{ideas_text}

SOURCE FINDINGS (extract thermal values from these):
{findings_text}

INSTRUCTIONS:
For each idea in the thermal or cross_domain domains, scan ALL findings for thermal data.
Only extract values explicitly mentioned. Return null if not found.
Key sources: packaging papers, cooling solution papers, chip test reports.
JEDEC JESD51 standard values: T_junction_max = 125°C (standard), 150°C (automotive).
HBM3 thermal limit: ~85°C. GPU junction: ~83-95°C nominal operating.

IMPORTANT: Use the FULL idea id string exactly as shown above (not truncated).

Return valid JSON only."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for extraction in raw.get("thermal_extractions", []):
            p = extraction.get("thermal_params", {})
            idea = Idea(
                id=extraction.get("idea_id", ""),
                title=f"Thermal extraction for {extraction.get('idea_id','?')[:8]}",
                domain="thermal",
                problem="Thermal parameter extraction",
                physical_limit="Thermal physics limits",
                proposed_direction="",
                thermal_params=ThermalParams(
                    t_junction_c=p.get("t_junction_c"),
                    heat_flux_w_cm2=p.get("heat_flux_w_cm2"),
                    thermal_resistance_c_per_w=p.get("thermal_resistance_c_per_w"),
                    delta_t_c=p.get("delta_t_c"),
                    t_ambient_c=p.get("t_ambient_c"),
                    cop_claimed=p.get("cop_claimed"),
                    material=p.get("material"),
                    source_ref=p.get("source_ref"),
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
        )
