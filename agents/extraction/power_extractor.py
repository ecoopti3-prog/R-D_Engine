"""
power_extractor.py — Agent 5
Extracts PowerParams from findings. Extraction only — no evaluation, no guessing.
Parallel to pdn_extractor but for power/electrical parameters.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, PowerParams

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precision data extraction system specializing in power and electrical parameters.

YOUR ONLY JOB: Extract numerical power/electrical values from provided text.
DO NOT calculate. DO NOT evaluate. DO NOT guess.
If a value is not explicitly stated in the source text → return null for that field.

POWER PARAMETERS TO EXTRACT:
- watt: Total power consumption in watts (look for "TDP", "total power", "power draw", "W")
- tdp_watt: Thermal Design Power specifically (look for "TDP", "thermal envelope")
- power_density_w_cm2: Power per unit die area (look for "W/cm²", "power density", "W per cm2")
- energy_per_op_pj: Energy efficiency (look for "pJ/op", "TOPS/W", "energy per operation")
- voltage_v: Supply voltage (look for "VDD", "core voltage", "supply", "V")
- current_a: Total current (look for "total current", "supply current", "A")
- efficiency_pct: Power conversion efficiency (look for "VRM efficiency", "PSU efficiency", "%")

CONTEXT CLUES FOR POWER DATA:
- AI accelerator papers always quote TDP
- Process node papers quote power density W/cm²
- Memory papers quote energy per bit or pJ/access
- Efficiency numbers appear in VRM/power converter papers
- "W per TFLOP" or "TOPS/W" → convert to energy_per_op if possible

UNIT NORMALIZATION RULES:
- Always convert to base units listed above
- TOPS/W → pJ/op = 1000 / TOPS_per_watt
- mW → W (divide by 1000)
- μW → W (divide by 1,000,000)
- W/mm² → W/cm² (multiply by 100)

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "power_extractions": [
    {
      "idea_id": "string matching one of the input ideas",
      "power_params": {
        "watt": number_or_null,
        "tdp_watt": number_or_null,
        "power_density_w_cm2": number_or_null,
        "energy_per_op_pj": number_or_null,
        "voltage_v": number_or_null,
        "current_a": number_or_null,
        "efficiency_pct": number_or_null,
        "source_ref": "paper title or URL"
      }
    }
  ]
}"""


class PowerExtractor(BaseAgent):
    AGENT_NAME    = "power_extractor"
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
        return f"""IDEAS TO EXTRACT POWER DATA FOR:
{ideas_text}

SOURCE FINDINGS (extract power values from these):
{findings_text}

INSTRUCTIONS:
For each idea, scan ALL findings for power/electrical numerical data.
Only extract values explicitly mentioned. Return null if not found.
Pay special attention to: GPU/TPU papers, AI accelerator specs, power integrity papers.
NVIDIA H100/H200/B200, AMD MI300X, Google TPU — these have published power specs.

UNIT CONVERSION: If you see TOPS/W, convert to pJ/op = 1000/TOPS_per_W.
If you see W/mm², convert to W/cm² by multiplying by 100.

IMPORTANT: Use the FULL idea id string exactly as shown above (not truncated).

Return valid JSON only."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for extraction in raw.get("power_extractions", []):
            p = extraction.get("power_params", {})
            idea = Idea(
                id=extraction.get("idea_id", ""),
                title=f"Power extraction for {extraction.get('idea_id','?')[:8]}",
                domain="power",
                problem="Power parameter extraction",
                physical_limit="Power density limits",
                proposed_direction="",
                power_params=PowerParams(
                    watt=p.get("watt"),
                    tdp_watt=p.get("tdp_watt"),
                    power_density_w_cm2=p.get("power_density_w_cm2"),
                    energy_per_op_pj=p.get("energy_per_op_pj"),
                    voltage_v=p.get("voltage_v"),
                    current_a=p.get("current_a"),
                    efficiency_pct=p.get("efficiency_pct"),
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
