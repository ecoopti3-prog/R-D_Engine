"""
pdn_extractor.py — Agent 8 (new domain)
Extracts Power Delivery Network parameters from findings.
Extraction only — no evaluation, no guessing.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, PDNParams

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precision data extraction system specializing in Power Delivery Network (PDN) parameters.

YOUR ONLY JOB: Extract numerical PDN values from the provided text. 
DO NOT calculate. DO NOT evaluate. DO NOT guess. 
If a value is not explicitly stated in the source text, return null for that field.

PDN PARAMETERS TO EXTRACT:
- ir_drop_mv: IR voltage drop in millivolts (look for "mV drop", "IR drop", "voltage droop")
- vdd_v: Supply voltage in volts (look for "VDD", "supply voltage", "core voltage")
- pdn_impedance_mohm: PDN impedance in milliohms (look for "target impedance", "Z_PDN", "mohm")
- frequency_ghz: Frequency at which impedance is measured
- bump_density_per_mm2: Power bump density (look for "bumps/mm2", "bump pitch", "C4 bumps")
- current_a: Total current in amps (look for "total current", "power delivery current")
- di_dt_a_per_ns: Current transient rate (look for "di/dt", "current slew rate")
- decap_nf: Decoupling capacitance in nanofarads
- n_chiplets: Number of chiplets in the package

CONTEXT CLUES FOR PDN DATA:
- Papers about advanced packaging (CoWoS, SoIC, EMIB, Foveros) often contain PDN data
- Chiplet papers often mention voltage domain isolation
- Power integrity papers are the richest source

OUTPUT FORMAT: Valid JSON only. No markdown. No commentary outside JSON.
{
  "pdn_extractions": [
    {
      "idea_id": "string matching one of the input ideas",
      "pdn_params": {
        "ir_drop_mv": number_or_null,
        "vdd_v": number_or_null,
        "pdn_impedance_mohm": number_or_null,
        "frequency_ghz": number_or_null,
        "bump_density_per_mm2": number_or_null,
        "current_a": number_or_null,
        "di_dt_a_per_ns": number_or_null,
        "decap_nf": number_or_null,
        "n_chiplets": integer_or_null,
        "source_ref": "paper title or URL"
      }
    }
  ]
}"""


class PDNExtractor(BaseAgent):
    AGENT_NAME    = "pdn_extractor"
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
            f"id={i.get('id','?')} title={i.get('title','')}"
            for i in ideas[:10]
        )
        return f"""IDEAS TO EXTRACT PDN DATA FOR:
{ideas_text}

SOURCE FINDINGS (extract PDN values from these):
{findings_text}

INSTRUCTIONS:
For each idea, scan ALL findings for PDN-related numerical data.
Only extract values explicitly mentioned. Return null if not found.
Pay special attention to: CoWoS papers, Foveros papers, EMIB papers, chiplet integration papers.
NVIDIA GB200, AMD MI300X, Intel Gaudi — these are known to have PDN challenges.

IMPORTANT: Use the FULL idea id string exactly as shown above (not truncated).

Return valid JSON only."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for extraction in raw.get("pdn_extractions", []):
            p = extraction.get("pdn_params", {})
            idea = Idea(
                id=extraction.get("idea_id", ""),
                title=f"PDN extraction for {extraction.get('idea_id','?')[:8]}",
                domain="pdn",
                problem="PDN parameter extraction",
                physical_limit="PDN physics limits",
                proposed_direction="",
                pdn_params=PDNParams(
                    ir_drop_mv=p.get("ir_drop_mv"),
                    vdd_v=p.get("vdd_v"),
                    pdn_impedance_mohm=p.get("pdn_impedance_mohm"),
                    frequency_ghz=p.get("frequency_ghz"),
                    bump_density_per_mm2=p.get("bump_density_per_mm2"),
                    current_a=p.get("current_a"),
                    di_dt_a_per_ns=p.get("di_dt_a_per_ns"),
                    decap_nf=p.get("decap_nf"),
                    n_chiplets=p.get("n_chiplets"),
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
