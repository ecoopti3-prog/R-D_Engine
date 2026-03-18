"""
weekly_analyst.py — Meta Agent (runs every Sunday 00:00 UTC)
Two jobs:
  1. CLEANUP: Archive ideas scored 3-5, delete ideas scored < 3 (with no new activity in 7 days)
  2. ANALYSIS: Identify which directions/domains are strongest, what's wasting cycles

This is NOT an LLM agent — it's deterministic Python + one LLM call for the narrative summary.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from core.base_agent import BaseAgent
from core.schemas import AgentOutput

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
SCORE_DELETE  = 3.0   # < 3.0  AND older than 7 days → DELETE from DB
SCORE_ARCHIVE = 5.0   # 3.0–5.0 → ARCHIVE (stays in DB, excluded from active cycles)
MIN_AGE_DAYS  = 7     # don't touch ideas younger than this (still maturing)

SYSTEM_PROMPT = """You are the Research Director reviewing one week of autonomous R&D output.

You receive statistics on what was found, what was killed, and what survived.
Your job: write a concise weekly report that a technical founder can read in 2 minutes.

STRUCTURE:
1. WEEK IN NUMBERS — bullets with the raw stats
2. STRONGEST SIGNAL — the 1-2 best ideas with why they scored high
3. DEAD ENDS — what the system keeps exploring that isn't working (domains/directions to deprioritize)
4. NEXT WEEK FOCUS — 3 specific keyword/domain combinations to prioritize

RULES:
- Be brutal. If week was weak, say it.
- Cite specific idea titles and scores, not vague summaries.
- "Explore thermal management" is useless. "Pursue HBM3 junction temp at CoWoS interface (currently 83°C vs 85°C limit)" is useful.
- If diamonds were found: explain WHY they scored high — which physics gap is most exploitable?

OUTPUT FORMAT: Valid JSON only. No markdown outside the report_markdown field.
{
  "report_markdown": "full markdown report text",
  "top_domains": ["domain1", "domain2"],
  "dead_end_domains": ["domain that keeps producing low scores"],
  "priority_keywords_next_week": ["keyword1", "keyword2", "keyword3"],
  "ideas_to_watch": ["idea_id_1", "idea_id_2"],
  "week_quality": "strong|average|weak"
}"""


class WeeklyAnalyst(BaseAgent):
    AGENT_NAME    = "weekly_analyst"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def build_user_prompt(self, context: dict) -> str:
        stats     = context.get("week_stats", {})
        top_ideas = context.get("top_ideas", [])
        dead_ends = context.get("dead_end_domains", [])

        top_ideas_text = "\n".join(
            f"- [{i.get('status')}] {i.get('title','')} | "
            f"Diamond={i.get('diamond_score',0):.1f} "
            f"Physics={i.get('physics_score',0):.1f} "
            f"Market={i.get('market_score',0):.1f} "
            f"Domain={i.get('domain','')}"
            for i in top_ideas[:15]
        )

        dead_end_text = "\n".join(
            f"- {d['domain']}: {d['count']} ideas, avg score {d['avg_score']:.1f}"
            for d in dead_ends
        )

        return f"""WEEK: {context.get('week_start')} to {context.get('week_end')}

STATISTICS:
- Ideas generated: {stats.get('generated', 0)}
- Ideas killed (physics gate): {stats.get('killed_physics', 0)}
- Ideas killed (agents): {stats.get('killed_agents', 0)}
- Ideas surviving: {stats.get('surviving', 0)}
- Diamonds found: {stats.get('diamonds', 0)}
- Avg diamond score of survivors: {stats.get('avg_score', 0):.1f}
- DB cleanup: {stats.get('deleted', 0)} deleted, {stats.get('archived', 0)} archived

TOP IDEAS THIS WEEK:
{top_ideas_text if top_ideas_text else 'None'}

LOW-SIGNAL DOMAINS (high volume, low scores):
{dead_end_text if dead_end_text else 'None identified'}

Write the weekly report. Be specific and actionable."""

    def parse_output(self, raw: dict, cycle_id: str) -> AgentOutput:
        return AgentOutput(
            agent=self.AGENT_NAME,
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="done",
            metadata={
                "report_markdown":              raw.get("report_markdown", ""),
                "top_domains":                  raw.get("top_domains", []),
                "dead_end_domains":             raw.get("dead_end_domains", []),
                "priority_keywords_next_week":  raw.get("priority_keywords_next_week", []),
                "ideas_to_watch":               raw.get("ideas_to_watch", []),
                "week_quality":                 raw.get("week_quality", "average"),
            },
        )
