"""
data_movement_extractor.py — Agent 7
Extracts DataMovementParams from findings. Extraction only — no evaluation, no guessing.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, DataMovementParams

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precision data extraction system specializing in memory and bandwidth parameters.

YOUR ONLY JOB: Extract numerical data movement values from provided text.
DO NOT calculate. DO NOT evaluate. DO NOT guess.
If a value is not explicitly stated in the source text → return null for that field.

DATA MOVEMENT PARAMETERS TO EXTRACT:
- bandwidth_gb_s: Memory/interconnect bandwidth in GB/s (look for "GB/s", "TB/s", "bandwidth")
- latency_ns: Memory access latency in nanoseconds (look for "ns latency", "access latency", "round-trip latency")
- memory_capacity_gb: Memory capacity in gigabytes (look for "GB HBM", "memory capacity", "DRAM capacity")
- interconnect_speed_gb_s: Chip-to-chip or die-to-die link speed in GB/s (look for "NVLink", "PCIe", "die-to-die", "UCIe")
- compute_tflops: Peak compute throughput (look for "TFLOPS", "TOPS", "compute throughput")

CONTEXT CLUES FOR BANDWIDTH DATA:
- HBM3 papers always quote peak bandwidth (3.35 TB/s for H100 SXM5)
- NVLink papers quote bidirectional bandwidth
- Memory papers often quote both peak and sustained bandwidth — extract both if present
- Roofline model papers contain arithmetic intensity and bandwidth
- "Memory wall" papers specifically quantify the compute/bandwidth ratio gap

UNIT NORMALIZATION:
- TB/s → GB/s (multiply by 1000)
- MB/s → GB/s (divide by 1000)
- TOPS → note model precision (INT8 TOPS vs FP16 TFLOPS differ significantly)
- ps latency → ns (divide by 1000)

KNOWN REFERENCE VALUES (use to validate extractions, NOT to fill in missing data):
- HBM3E: ~3.9 TB/s aggregate bandwidth
- PCIe Gen5: 128 GB/s bidirectional
- NVLink 4.0: 900 GB/s bidirectional per GPU
- DDR5: ~100 GB/s per channel

OUTPUT FORMAT: Valid JSON only. No markdown.
{
  "data_movement_extractions": [
    {
      "idea_id": "string matching one of the input ideas",
      "data_movement_params": {
        "bandwidth_gb_s": number_or_null,
        "latency_ns": number_or_null,
        "memory_capacity_gb": number_or_null,
        "interconnect_speed_gb_s": number_or_null,
        "compute_tflops": number_or_null,
        "source_ref": "paper title or URL"
      }
    }
  ]
}"""


class DataMovementExtractor(BaseAgent):
    AGENT_NAME    = "data_movement_extractor"
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
        return f"""IDEAS TO EXTRACT DATA MOVEMENT PARAMS FOR:
{ideas_text}

SOURCE FINDINGS (extract bandwidth/latency/compute values from these):
{findings_text}

INSTRUCTIONS:
For each idea in data_movement or cross_domain domains, scan ALL findings.
Only extract values explicitly mentioned. Return null if not found.
Focus on: HBM bandwidth specs, NVLink/PCIe specs, memory latency measurements.
If both peak and sustained bandwidths are mentioned, extract the sustained value (more realistic).

IMPORTANT: Use the FULL idea id string exactly as shown above (not truncated).

Return valid JSON only."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for extraction in raw.get("data_movement_extractions", []):
            p = extraction.get("data_movement_params", {})
            idea = Idea(
                id=extraction.get("idea_id", ""),
                title=f"DataMovement extraction for {extraction.get('idea_id','?')[:8]}",
                domain="data_movement",
                problem="Data movement parameter extraction",
                physical_limit="Bandwidth and latency limits",
                proposed_direction="",
                data_movement_params=DataMovementParams(
                    bandwidth_gb_s=p.get("bandwidth_gb_s"),
                    latency_ns=p.get("latency_ns"),
                    memory_capacity_gb=p.get("memory_capacity_gb"),
                    interconnect_speed_gb_s=p.get("interconnect_speed_gb_s"),
                    compute_tflops=p.get("compute_tflops"),
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
