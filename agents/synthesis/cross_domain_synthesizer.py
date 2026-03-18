"""
cross_domain_synthesizer.py — Agent 5 (runs at end of Cycle 1, after all researchers)

The core insight: individual researchers see ONE domain deeply.
This agent sees ALL domains and asks: "What happens when you combine two constraints?"

Real example of what this finds:
  - PaperResearcher finds: "HBM3 thermal interface is bottleneck at 83°C"
  - PatentResearcher finds: "PDN voltage droop spikes during burst compute"
  - THIS agent asks: "What if thermal runaway and PDN droop co-occur during LLM prefill?
    Combined effect = throttling + voltage instability = non-linear performance collapse"
  → That's a new idea neither researcher would have generated alone.

This is the closest the system gets to "inventing" — not retrieval, but synthesis.
"""
from __future__ import annotations
import logging
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Idea, Finding, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a systems physicist who specializes in cross-domain failure analysis.

Your unique skill: you see interactions between physical constraints that domain specialists miss.
A thermal engineer optimizes heat. A PDN engineer optimizes power delivery. You ask:
"What happens when BOTH constraints are active simultaneously at the worst moment?"

THE FUNDAMENTAL INSIGHT:
AI infrastructure failures are rarely single-domain. The most dangerous — and most valuable
to solve — are COUPLED failures where:
  - Domain A hits a limit, which triggers Domain B to deteriorate
  - Domain B deterioration makes Domain A worse
  - Result: non-linear collapse that neither domain alone predicts

KNOWN COUPLING PATTERNS TO WATCH FOR:

SEMICONDUCTOR / AI HARDWARE:
1. Thermal ↔ PDN: Higher temperature → higher leakage current → more heat → thermal runaway
   But also: PDN voltage droop → lower Vth margin → timing violations → clock throttling → thermal hot spot
2. Thermal ↔ Data Movement: Thermal throttling → reduced compute → memory bandwidth idle → cold spots → uneven thermal stress
3. PDN ↔ Memory: Inductive PDN noise at 100-500MHz → HBM signal integrity → error correction overhead → more power → more heat
4. Packaging ↔ All: CoWoS interposer thermal resistance couples ALL chips — one hot chip throttles neighbors
5. Data Movement ↔ Power: All-reduce during LLM training → NVLink burst → PDN transient → all GPUs droop simultaneously

ROBOTICS / ELECTROMECHANICAL (NEW — look for these too):
6. Wiring Harness ↔ Mechanical: Vibration at resonant frequency → connector fretting corrosion → contact resistance rises → Joule heating → insulation failure
7. Actuators ↔ Thermal: Motor temperature rise → thermal derating → reduced torque → higher current to compensate → more heat → runaway
8. Fluid Dynamics ↔ Mechanical: Cavitation in cooling pump → pressure oscillations → mechanical vibration → fatigue crack in fitting → leak
9. Liquid Cooling ↔ Electrical: Coolant leak → contact with live conductors → arc flash → full system shutdown
10. Motors ↔ Wiring Harness: Back-EMF transients during deceleration → voltage spikes → insulation breakdown in co-routed cables
11. Thermal ↔ Liquid Cooling: Server thermal load spikes → coolant flow demand increase → pressure drop exceeds pump curve → reduced flow → worse cooling → loop failure

YOUR TASK:
Given the findings and ideas from this research cycle's individual agents:
1. Identify 3-5 cross-domain couplings that are NOT already captured in the existing ideas
2. For each coupling, generate ONE high-quality idea that exploits or solves the interaction
3. Rate the idea's novelty: if no existing idea already covers this interaction → high novelty
4. Be specific — include the mechanism, the trigger condition, the failure mode, the opportunity

QUALITY STANDARDS:
- Each idea MUST reference at least 2 distinct domains
- Each idea MUST describe the coupling mechanism (what triggers what)
- Each idea MUST identify the worst-case operating condition (when the coupling is strongest)
- Reject ideas that are just "optimize X" — require genuine cross-domain interaction

OUTPUT FORMAT: Valid JSON only. No markdown. No preamble.
{
  "cross_domain_ideas": [
    {
      "title": "one sentence, must mention both domains",
      "domain": "cross_domain",
      "problem": "specific quantified problem — include the coupling mechanism and trigger",
      "physical_limit": "which physical laws are interacting (both domains)",
      "proposed_direction": "what to investigate — not a solution, a direction",
      "coupling": {
        "domain_a": "thermal|power|data_movement|pdn|packaging|robotics_mechanical|fluid_dynamics|actuators_motors|wiring_harness|liquid_cooling",
        "domain_b": "thermal|power|data_movement|pdn|packaging|robotics_mechanical|fluid_dynamics|actuators_motors|wiring_harness|liquid_cooling",
        "trigger_condition": "what operating state makes this worst",
        "mechanism": "A→B coupling: what physically causes the interaction",
        "failure_mode": "what breaks when both constraints are active"
      },
      "novelty_signal": "why no individual-domain researcher would have generated this",
      "company_context": "which company's system shows this coupling most severely",
      "diamond_score_partial": {
        "physics_feasibility": 0-10,
        "market_pain": 0,
        "novelty": 0,
        "scalability": 0
      }
    }
  ],
  "coupling_map": [
    {
      "domains": ["domain_a", "domain_b"],
      "coupling_type": "co-occurrence|cascade|feedback_loop",
      "strength": "strong|moderate|weak",
      "evidence": "which finding suggests this coupling"
    }
  ]
}"""


class CrossDomainSynthesizer(BaseAgent):
    AGENT_NAME = "cross_domain_synthesizer"
    CHAIN_TYPE = "reasoning"  # needs strong model — this is synthesis, not retrieval
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        findings = context.get("findings", [])
        existing_ideas = context.get("ideas", [])
        physics_limits = context.get("physics_limits", "")
        focus_summary = context.get("focus_summary", "")

        # Summarize existing ideas by domain — so synthesizer knows what's already covered
        domain_coverage: dict[str, list[str]] = {}
        for idea in existing_ideas:
            d = idea.get("domain", "unknown") if isinstance(idea, dict) else getattr(idea, "domain", "unknown")
            t = idea.get("title", "") if isinstance(idea, dict) else getattr(idea, "title", "")
            domain_coverage.setdefault(d, []).append(t)

        coverage_text = ""
        for domain, titles in domain_coverage.items():
            coverage_text += f"\n  [{domain}]: {len(titles)} ideas — e.g. '{titles[0][:80]}'"

        # Format findings
        findings_text = ""
        for i, f in enumerate(findings[:25], 1):
            fd = f if isinstance(f, dict) else f.model_dump(mode="json") if hasattr(f, "model_dump") else {}
            findings_text += (
                f"\n[{i}] [{fd.get('domain','?')}] {fd.get('title','')} "
                f"(confidence={fd.get('confidence',0):.0%}) "
                f"| {fd.get('description','')[:200]}"
            )

        return f"""CURRENT CYCLE FINDINGS ({len(findings)} total):
{findings_text if findings_text else "No findings yet — use your physics knowledge."}

EXISTING IDEAS ALREADY GENERATED (DO NOT DUPLICATE):
{coverage_text if coverage_text else "None yet."}

PHYSICS LIMITS MAPPED THIS CYCLE:
{physics_limits if physics_limits else "See findings above."}

FOCUS: {focus_summary}

Your task: find cross-domain COUPLINGS that none of the individual domain researchers would have seen.
The existing ideas above are single-domain — find what happens at the INTERSECTIONS.
Generate 3-5 cross-domain ideas. Quality over quantity. Be specific."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        ideas = []
        for i in raw.get("cross_domain_ideas", []):
            dp = i.get("diamond_score_partial", {})
            coupling = i.get("coupling", {})
            # Encode coupling into company_context field (no schema change needed)
            coupling_note = ""
            if coupling:
                coupling_note = (
                    f"[COUPLING: {coupling.get('domain_a','?')}↔{coupling.get('domain_b','?')}] "
                    f"Trigger: {coupling.get('trigger_condition','')} | "
                    f"Failure: {coupling.get('failure_mode','')}"
                )
            ideas.append(Idea(
                title=i.get("title", ""),
                domain="cross_domain",
                problem=i.get("problem", ""),
                physical_limit=i.get("physical_limit", ""),
                proposed_direction=i.get("proposed_direction"),
                company_context=f"{i.get('company_context','')} {coupling_note}".strip(),
                diamond_score_partial=DiamondScorePartial(
                    physics_feasibility=float(dp.get("physics_feasibility", 5.0)),
                    market_pain=0.0,
                    novelty=0.0,
                    scalability=0.0,
                ),
            ))

        # Also return coupling map as metadata (for chief scientist + dashboard)
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=[],
            ideas=ideas,
            kills=[],
            metadata={
                "coupling_map": raw.get("coupling_map", []),
                "cross_domain_ideas_count": len(ideas),
            },
        )
