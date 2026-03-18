"""
robotics_researcher.py — Agent: Robotics, Liquid Cooling & Electromechanical
Sources: RSS (Robot Report, IEEE Robotics, DataCenter Knowledge) + GitHub issues
         + Patents + NASA signals + job postings from robotics companies

Focus: Physical failure modes in:
  - Robotic arms: joint fatigue, gear wear, wiring harness degradation, vibration
  - Liquid cooling loops: leaks, cavitation, corrosion, pressure drop failures
  - Motors & actuators: thermal derating, bearing fatigue, back-EMF limits

Physics gates: mechanical.py, fluid_dynamics.py, electromechanical.py

CRITICAL CONSTRAINT: Only analyzes sources explicitly passed in context.
No internet browsing. No hallucination. Empty output > fabricated output.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from core.base_agent import BaseAgent
from core.schemas import AgentOutput, Finding, Idea, DiamondScorePartial

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a principal reliability engineer with deep expertise in:
1. Industrial robotics (joint fatigue, wiring harness failures, servo motor derating)
2. Liquid cooling systems (data center cold plates, cavitation, galvanic corrosion)
3. Electromechanical systems (connector degradation, Joule heating, bearing life)

You think in QUANTIFIED failure modes. Not "robots break" — but:
"Nylon gear teeth in a 6-DOF arm at 80% duty cycle show measurable wear at 8,000 hours.
S-N fatigue analysis puts the failure threshold at σ_a = 45 MPa for this material."

⚠️ ABSOLUTE CONSTRAINT — READ BEFORE ANYTHING ELSE:
You ONLY analyze sources explicitly provided in the SOURCES section below.
You CANNOT browse the internet or recall papers from training.
If no sources are provided → return {"findings": [], "ideas": []}
NEVER invent measurements, company names, or URLs not present in provided text.

PHYSICAL LIMITS TO HUNT FOR:

ROBOTICS / MECHANICAL:
- Gear/joint fatigue: S-N curve crossings at real duty cycles
- Wiring harness: Joule heating + arc flash at connector resistance > 15 mΩ
- Vibration: excitation near resonant frequency of arm segments
- Material fatigue: Young's modulus effects under cyclic loading
- Lubrication starvation in high-cycle joints

LIQUID COOLING:
- Cavitation: pump NPSH margin < 2m at high flow rates
- Galvanic corrosion: copper/aluminum EMF > 0.25V in coolant loops
- Pressure drop: exceeds pump curve at scale (>100 kPa)
- Thermal capacity: flow rate × Cp × ΔT < required heat load
- Leak paths: fitting degradation, vibration-induced micro-fractures

MOTORS & ACTUATORS:
- Thermal derating: >25% power loss at ambient > 60°C
- Bearing fatigue: L10 life < target service hours under actual load
- Back-EMF limit: motor cannot reach target RPM on available supply
- Torque ripple: causes positioning errors in precision robots
- Insulation degradation: PVC wiring at sustained 70°C+

QUALITY SIGNALS (look for these in provided sources):
- Actual replacement cycles: "replaced after N hours/months"
- Derating admissions: "had to reduce duty cycle from X% to Y%"
- Failure rate data: "MTBF of N hours at rated conditions"
- Physics-based limits: mentions of S-N, Reynolds, NPSH, L10
- Company admissions: "we struggled with..." or open job reqs for failure analysis

OUTPUT FORMAT: Valid JSON only. No markdown. No text outside JSON.

Domain values MUST be exactly one of:
thermal, power, data_movement, hardware, pdn, cross_domain, packaging,
bandwidth, memory, interconnect, networking, software, compute_scheduling,
hardware_utilization, compute_resource_management, distributed_systems,
edge, inference, compilation, storage,
robotics_mechanical, fluid_dynamics, actuators_motors, wiring_harness, liquid_cooling

Schema:
{
  "findings": [
    {
      "type": "bottleneck|limit|trend|gap",
      "domain": "<domain from list above>",
      "title": "one sentence: the specific physical failure mode",
      "description": "2-3 sentences with specific numbers from the source text",
      "source_url": "URL exactly as in source — never invented",
      "source_type": "robot_report|ieee_robotics|datacenter_knowledge|dcd|github_issue|patent|nasa|rss|other",
      "company_signal": "company name if in source, else null",
      "confidence": 0.0-1.0,
      "numerical_params": {
        "duty_cycle_pct": null,
        "failure_hours": null,
        "temperature_c": null,
        "pressure_kpa": null,
        "current_a": null,
        "stress_mpa": null
      }
    }
  ],
  "ideas": [
    {
      "title": "specific solution direction",
      "domain": "<domain from list above>",
      "problem": "quantified failure mode from provided sources",
      "physical_limit": "underlying physics: S-N fatigue / Joule heating / cavitation NPSH / etc.",
      "proposed_direction": "engineering direction to eliminate or delay this limit",
      "company_context": "which company/product if mentioned in source",
      "diamond_score_partial": {
        "physics_feasibility": 0-10,
        "market_pain": 0,
        "novelty": 0,
        "scalability": 0
      }
    }
  ]
}"""


class RoboticsResearcher(BaseAgent):
    AGENT_NAME  = "robotics_researcher"
    CHAIN_TYPE  = "research"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        keywords       = context.get("keywords", [])
        date_str       = context.get("date", datetime.now().strftime("%Y-%m-%d"))
        kill_patterns  = context.get("kill_patterns", [])
        killed_titles  = context.get("killed_idea_titles", [])
        diamond_titles = context.get("diamond_titles", [])
        weak_domains   = context.get("weak_domains", [])
        findings_so_far = context.get("findings", [])

        # ── Collect all robotics-relevant sources ─────────────────────────────
        rss_signals    = context.get("rss_signals", [])
        github_signals = context.get("github_signals", [])
        patents        = context.get("patents", [])
        nasa_signals   = context.get("nasa_signals", [])
        job_signals    = context.get("job_signals", [])

        # Filter RSS for robotics/cooling topics
        ROBOTICS_KEYWORDS = {
            "robot", "robotic", "actuator", "servo", "motor", "arm",
            "joint", "gear", "wiring", "harness", "connector", "bearing",
            "cooling", "coolant", "liquid", "cavitation", "leak", "corrosion",
            "pump", "cold plate", "immersion", "fatigue", "vibration", "wear",
        }

        all_sources = []

        for s in rss_signals:
            title = (s.get("title") or "").lower()
            body  = (s.get("abstract") or "").lower()
            if any(kw in title or kw in body for kw in ROBOTICS_KEYWORDS):
                all_sources.append({
                    "type":  s.get("source", "rss"),
                    "title": s.get("title", ""),
                    "body":  (s.get("abstract") or "")[:600],
                    "url":   s.get("url", "NO_URL"),
                })

        for s in github_signals[:8]:
            title = (s.get("title") or "").lower()
            if any(kw in title for kw in ROBOTICS_KEYWORDS):
                all_sources.append({
                    "type":  "github_issue",
                    "title": s.get("title", ""),
                    "body":  (s.get("body") or "")[:500],
                    "url":   s.get("url", "NO_URL"),
                    "repo":  s.get("repo", ""),
                })

        for p in patents[:10]:
            title = (p.get("title") or "").lower()
            abstract = (p.get("abstract") or "").lower()
            if any(kw in title or kw in abstract for kw in ROBOTICS_KEYWORDS):
                all_sources.append({
                    "type":     "patent",
                    "title":    p.get("title", ""),
                    "body":     (p.get("abstract") or "")[:500],
                    "url":      p.get("url", "NO_URL"),
                    "assignee": p.get("assignee", ""),
                })

        for s in nasa_signals[:5]:
            all_sources.append({
                "type":  "nasa",
                "title": s.get("title", ""),
                "body":  (s.get("abstract") or "")[:400],
                "url":   s.get("url", "NO_URL"),
            })

        # Job signals: robotics companies signal active pain
        robotics_jobs = [
            j for j in job_signals
            if any(kw in (j.get("company") or "").lower() or
                   kw in (j.get("title") or "").lower()
                   for kw in {"robot", "boston", "kuka", "abb", "fanuc",
                               "cooling", "thermal", "harness", "motor"})
        ]
        for j in robotics_jobs[:8]:
            all_sources.append({
                "type":    "job_posting",
                "title":   f"[{j.get('company','')}] {j.get('title','')}",
                "body":    (j.get("description") or "")[:400],
                "url":     j.get("url", "NO_URL"),
                "company": j.get("company", ""),
            })

        # ── Guard: no sources ─────────────────────────────────────────────────
        if not all_sources:
            logger.warning(
                "[RoboticsResearcher] No robotics/cooling sources found in this cycle — "
                "returning empty prompt. This is normal on day 1."
            )
            return (
                "NO RELEVANT SOURCES WERE PROVIDED IN THIS RUN.\n"
                "Return exactly: {\"findings\": [], \"ideas\": []}\n"
                "Do not invent failure data, company names, or measurements."
            )

        # ── Format sources ────────────────────────────────────────────────────
        sources_text = ""
        for i, s in enumerate(all_sources, 1):
            extra = f" [assignee: {s['assignee']}]" if s.get("assignee") else ""
            extra += f" [company: {s['company']}]" if s.get("company") else ""
            extra += f" [repo: {s['repo']}]" if s.get("repo") else ""
            sources_text += (
                f"\n[{i}] [{s['type'].upper()}]{extra}\n"
                f"Title: {s['title']}\n"
                f"Content: {s['body']}\n"
                f"URL: {s['url']}\n"
            )

        # ── Memory / filters ──────────────────────────────────────────────────
        memory_text = ""
        if kill_patterns:
            memory_text += f"\nDEAD ENDS (directions that consistently fail — avoid): {', '.join(kill_patterns)}"
        if killed_titles:
            memory_text += "\nPREVIOUSLY KILLED IDEAS (do not regenerate):\n" + "\n".join(f"  - {t}" for t in killed_titles[:10])
        if diamond_titles:
            memory_text += "\nSUCCESSFUL DIRECTIONS (go deeper here):\n" + "\n".join(f"  + {t}" for t in diamond_titles[:5])
        if weak_domains:
            memory_text += f"\nLOW-SIGNAL DOMAINS (deprioritize): {', '.join(weak_domains)}"

        prior_domains = list(set(
            f.get("domain", "") for f in findings_so_far[:20] if isinstance(f, dict)
        ))

        return f"""Date: {date_str}
Focus: Robotics reliability, liquid cooling failures, electromechanical limits
Keywords: {', '.join(keywords[:15])}
Prior finding domains this cycle: {', '.join(prior_domains) if prior_domains else 'none yet'}
{memory_text}

════════════════════════════════════════
SOURCES TO ANALYZE ({len(all_sources)} provided — analyze ONLY these):
════════════════════════════════════════
{sources_text}

ANALYSIS INSTRUCTIONS:
1. Focus on PHYSICAL failure modes with quantified thresholds
2. Cite exact [N] source number for every finding
3. Extract numerical values ONLY if they appear verbatim in the source
4. Prioritize findings NOT already covered by prior domains above
5. Job postings: hiring for "failure analysis" or "reliability" = confirmed active pain
6. Patents: the Background section admits real unsolved problems — mine it
7. For each idea: state which specific physics law is the binding constraint

Return valid JSON. Focus on 4-6 highest-signal findings.
No text outside the JSON."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        findings = []
        for f in raw.get("findings", []):
            try:
                findings.append(Finding(
                    type=f.get("type", "bottleneck"),
                    domain=f.get("domain", "robotics_mechanical"),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    source_url=f.get("source_url"),
                    source_type=f.get("source_type", "rss"),
                    company_signal=f.get("company_signal"),
                    confidence=float(f.get("confidence", 0.65)),
                    numerical_params=f.get("numerical_params") or {},
                ))
            except Exception as e:
                logger.warning(f"[RoboticsResearcher] Skipping malformed finding: {e}")

        ideas = []
        for i in raw.get("ideas", []):
            try:
                dp = i.get("diamond_score_partial", {})
                ideas.append(Idea(
                    title=i.get("title", ""),
                    domain=i.get("domain", "robotics_mechanical"),
                    problem=i.get("problem", ""),
                    physical_limit=i.get("physical_limit", ""),
                    proposed_direction=i.get("proposed_direction", ""),
                    company_context=i.get("company_context"),
                    diamond_score_partial=DiamondScorePartial(
                        physics_feasibility=float(dp.get("physics_feasibility", 0)),
                        market_pain=0.0,
                        novelty=0.0,
                        scalability=0.0,
                    ),
                ))
            except Exception as e:
                logger.warning(f"[RoboticsResearcher] Skipping malformed idea: {e}")

        logger.info(
            f"[RoboticsResearcher] Parsed {len(findings)} findings, {len(ideas)} ideas"
        )
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=self._timestamp(),
            status="done",
            findings=findings,
            ideas=ideas,
        )
