"""
supabase_client.py — All database operations.
State machine, idea storage, embeddings, cycle management.
"""
from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from typing import Optional
from supabase import create_client, Client
from config.settings import SUPABASE_URL, SUPABASE_KEY
from core.schemas import Idea, AgentOutput, CycleState, PhysicsVerdict, NoveltyResult

logger = logging.getLogger(__name__)


def _sanitize(obj):
    """Remove null bytes from any object before sending to PostgreSQL.
    PostgreSQL raises 22P05 on \x00/\u0000 in text fields."""
    if isinstance(obj, str):
        return obj.replace("\x00", "").replace("\u0000", "")
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return obj


_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── Cycle management ──────────────────────────────────────────────────────────

def create_cycle(cycle_id: str, cycle_number: int) -> bool:
    db = get_client()
    try:
        db.table("research_cycles").insert({
            "id": cycle_id,
            "date": datetime.now(timezone.utc).date().isoformat(),
            "cycle_number": cycle_number,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] create_cycle failed: {e}")
        return False


def update_heartbeat(cycle_id: str) -> bool:
    db = get_client()
    try:
        db.table("research_cycles").update({
            "last_heartbeat": datetime.now(timezone.utc).isoformat()
        }).eq("id", cycle_id).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] heartbeat failed: {e}")
        return False


def complete_cycle(cycle_id: str, status: str = "done") -> bool:
    db = get_client()
    try:
        db.table("research_cycles").update({
            "status": status,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", cycle_id).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] complete_cycle failed: {e}")
        return False


def is_cycle_already_done_today(cycle_number: int) -> bool:
    db = get_client()
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        res = db.table("research_cycles") \
            .select("id") \
            .eq("date", today) \
            .eq("cycle_number", cycle_number) \
            .eq("status", "done") \
            .execute()
        return len(res.data) > 0
    except Exception as e:
        logger.error(f"[DB] is_cycle_done check failed: {e}")
        return False


def is_another_worker_running(cycle_number: int) -> bool:
    """Check if another worker is currently running the same cycle."""
    db = get_client()
    try:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        res = db.table("research_cycles") \
            .select("id") \
            .eq("status", "running") \
            .eq("cycle_number", cycle_number) \
            .gte("last_heartbeat", cutoff) \
            .execute()
        return len(res.data) > 0
    except Exception as e:
        logger.error(f"[DB] worker check failed: {e}")
        return False


def is_cold_start() -> bool:
    db = get_client()
    try:
        res = db.table("ideas").select("id", count="exact").execute()
        return res.count == 0
    except Exception as e:
        logger.error(f"[DB] cold start check failed: {e}")
        return True


# ── Idea storage ──────────────────────────────────────────────────────────────

def save_idea(idea: Idea, cycle_id: str, embedding: Optional[list] = None,
              source_url: Optional[str] = None) -> bool:
    """
    Persist an idea to the database.
    source_url: optional — if provided, denormalizes venue into source_venue column
                so source_performance_view works without full lineage join.
    """
    db = get_client()
    try:
        row = {
            "id":                idea.id,
            "cycle_id":          cycle_id,
            "title":             idea.title,
            "domain":            idea.domain,
            "problem_statement": idea.problem,
            "physical_limit":    idea.physical_limit,
            "proposed_direction": idea.proposed_direction or "",
            "company_context":   idea.company_context or "",
            "diamond_score":     0.0,
            # Persist all sub-scores so load_active_ideas can reconstruct them
            "physics_score":     idea.diamond_score_partial.physics_feasibility,
            "market_score":      idea.diamond_score_partial.market_pain,
            "novelty_score":     idea.diamond_score_partial.novelty,
            "scalability_score": idea.diamond_score_partial.scalability,
            "status":            idea.status,
            "kill_reason":       idea.kill_reason,
        }
        if embedding is not None:
            row["embedding"] = embedding
        # Denormalize venue for fast source performance queries
        if source_url:
            from db.lineage import _extract_venue
            row["source_venue"] = _extract_venue(source_url)
        # Persist extracted physics params so they survive across cycles
        if idea.power_params:
            row["power_params"] = idea.power_params.model_dump(exclude_none=True)
        if idea.thermal_params:
            row["thermal_params"] = idea.thermal_params.model_dump(exclude_none=True)
        if idea.data_movement_params:
            row["data_movement_params"] = idea.data_movement_params.model_dump(exclude_none=True)
        if idea.pdn_params:
            row["pdn_params"] = idea.pdn_params.model_dump(exclude_none=True)
        db.table("ideas").upsert(row).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] save_idea failed: {e}")
        return False


def kill_idea(idea_id: str, kill_reason: str, kill_category: str) -> bool:
    db = get_client()
    try:
        if len(idea_id) < 36:
            # UUID column can't use LIKE — cast to text via ilike on string representation
            # Fetch active ideas only to find match
            rows = db.table("ideas").select("id").in_("status", ["active", "physics_unverified"]).execute()
            matches = [r["id"] for r in (rows.data or []) if r["id"].startswith(idea_id)]
            if matches:
                idea_id = matches[0]
            else:
                logger.warning(f"[DB] kill_idea: short id '{idea_id}' not found in active ideas — skipping")
                return False
        db.table("ideas").update({
            "status":      "killed",
            "kill_reason": f"[{kill_category}] {kill_reason}",
        }).eq("id", idea_id).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] kill_idea failed: {e}")
        return False


def update_diamond_score(idea_id: str, score: float, status: str,
                         physics: float = None, market: float = None,
                         novelty: float = None, scalability: float = None) -> bool:
    db = get_client()
    try:
        row: dict = {"diamond_score": score, "status": status}
        if physics    is not None: row["physics_score"]     = physics
        if market     is not None: row["market_score"]      = market
        if novelty    is not None: row["novelty_score"]     = novelty
        if scalability is not None: row["scalability_score"] = scalability
        db.table("ideas").update(row).eq("id", idea_id).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] update_diamond_score failed: {e}")
        return False


def load_active_ideas(cycle_id: str = None) -> list[Idea]:
    """Load all active ideas from today's cycle (or all active) for inter-cycle loading."""
    db = get_client()
    try:
        query = db.table("ideas").select("*").in_("status", ["active", "physics_unverified"])
        if cycle_id:
            query = query.eq("cycle_id", cycle_id)
        res = query.order("created_at", desc=True).execute()
        ideas = []
        for row in res.data:
            try:
                from core.schemas import (
                    PowerParams, ThermalParams, DataMovementParams,
                    PDNParams, DiamondScorePartial
                )
                idea = Idea(
                    id=row["id"],
                    title=row.get("title", ""),
                    domain=row.get("domain", "hardware") if row.get("domain") in [
                        "thermal","power","data_movement","hardware","pdn","cross_domain",
                        "packaging","bandwidth","memory","interconnect","networking","software",
                        "compute_scheduling","hardware_utilization","compute_resource_management",
                        "distributed_systems"
                    ] else "hardware",   # FIX: unknown domains → "hardware" instead of crash+drop
                    problem=row.get("problem_statement", ""),
                    physical_limit=row.get("physical_limit", ""),
                    proposed_direction=row.get("proposed_direction", ""),
                    company_context=row.get("company_context"),
                    status=row.get("status", "active"),
                    kill_reason=row.get("kill_reason"),
                    cycle_id=row.get("cycle_id"),  # FIX: needed for novelty exclude logic
                    diamond_score_partial=DiamondScorePartial(
                        physics_feasibility=float(row.get("physics_score", 0)),
                        market_pain=float(row.get("market_score", 0)),
                        novelty=float(row.get("novelty_score", 0)),
                        scalability=float(row.get("scalability_score", 0)),
                    ),
                    power_params=PowerParams(**row["power_params"]) if row.get("power_params") else None,
                    thermal_params=ThermalParams(**row["thermal_params"]) if row.get("thermal_params") else None,
                    data_movement_params=DataMovementParams(**row["data_movement_params"]) if row.get("data_movement_params") else None,
                    pdn_params=PDNParams(**row["pdn_params"]) if row.get("pdn_params") else None,
                )
                ideas.append(idea)
            except Exception as e:
                logger.warning(f"[DB] Failed to deserialize idea {row.get('id','?')}: {e}")
                continue
        logger.info(f"[DB] Loaded {len(ideas)} active ideas")
        return ideas
    except Exception as e:
        logger.error(f"[DB] load_active_ideas failed: {e}")
        return []


def load_today_cycle_id(cycle_number: int) -> Optional[str]:
    """Get the cycle_id for today's run of a given cycle number."""
    db = get_client()
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        res = db.table("research_cycles") \
            .select("id") \
            .eq("date", today) \
            .eq("cycle_number", cycle_number) \
            .order("started_at", desc=True) \
            .limit(1) \
            .execute()
        if res.data:
            return res.data[0]["id"]
        return None
    except Exception as e:
        logger.error(f"[DB] load_today_cycle_id failed: {e}")
        return None


def save_finding(finding, cycle_id: str) -> bool:
    """Save a single Finding to the findings table."""
    db = get_client()
    try:
        db.table("findings").insert({
            "cycle_id":       cycle_id,
            "agent_name":     "research",
            "type":           finding.type,
            "domain":         finding.domain,
            "title":          finding.title,
            "description":    finding.description,
            "source_url":     finding.source_url,
            "source_type":    finding.source_type,
            "company_signal": finding.company_signal,
            "confidence":     finding.confidence,
            "numerical_params": finding.numerical_params,
        }).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] save_finding failed: {e}")
        return False


def load_today_findings(cycle_id: str = None) -> list[dict]:
    """Load findings from today for use in analysis cycles."""
    db = get_client()
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        query = db.table("findings").select("*")
        if cycle_id:
            query = query.eq("cycle_id", cycle_id)
        else:
            # Join via research_cycles to get today's findings
            today_cycles = db.table("research_cycles") \
                .select("id").eq("date", today).execute()
            cycle_ids = [c["id"] for c in today_cycles.data]
            if not cycle_ids:
                return []
            query = query.in_("cycle_id", cycle_ids)
        res = query.order("created_at").execute()
        return res.data
    except Exception as e:
        logger.error(f"[DB] load_today_findings failed: {e}")
        return []


def load_recent_findings(days_back: int = 7, limit: int = 50) -> list[dict]:
    """Load high-confidence findings from the past N days — feeds PhysicsLimitMapper memory."""
    db = get_client()
    try:
        from datetime import timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        res = (
            db.table("findings")
            .select("domain, title, description, confidence, company_signal, numerical_params, created_at")
            .gte("created_at", since)
            .gte("confidence", 0.6)
            .order("confidence", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"[DB] load_recent_findings failed: {e}")
        return []


def get_all_embeddings(exclude_cycle_id: str | None = None) -> list[dict]:
    """Return all idea embeddings for novelty detection, optionally excluding current cycle."""
    db = get_client()
    try:
        q = db.table("ideas").select("id, embedding").not_.is_("embedding", "null")
        if exclude_cycle_id:
            q = q.neq("cycle_id", exclude_cycle_id)
        res = q.execute()
        result = [{"id": r["id"], "embedding": r["embedding"]} for r in res.data]
        logger.info(f"[DB] Loaded {len(result)} archive embeddings" + (f" (excl. cycle {exclude_cycle_id[:8]})" if exclude_cycle_id else ""))
        return result
    except Exception as e:
        logger.error(f"[DB] get_all_embeddings failed: {e}")
        return []


# ── Agent output storage ──────────────────────────────────────────────────────

def save_agent_output(output: AgentOutput) -> bool:
    db = get_client()
    try:
        db.table("agent_outputs").insert(_sanitize({
            "cycle_id":   output.cycle_id,
            "agent_name": output.agent,
            "output":     output.model_dump(mode="json"),
            "llm_used":   output.llm_used,
            "tokens_used": output.tokens_used,
            "duration_ms": output.duration_ms,
            "status":     output.status,
        })).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] save_agent_output failed: {e}")
        return False


# ── Weekly cleanup & stats ────────────────────────────────────────────────────

def cleanup_weak_ideas(
    score_delete: float = 3.0,
    score_archive: float = 5.0,
    min_age_days: int = 7,
) -> dict:
    """
    Archive ideas scored 3-5, delete ideas scored < 3.
    Only touches ideas older than min_age_days (let new ideas mature first).
    Returns counts: {deleted, archived}.
    """
    from datetime import timedelta
    db = get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
    counts = {"deleted": 0, "archived": 0}

    try:
        # DELETE: score < score_delete, older than min_age_days, already killed/archived
        # We never delete diamonds or active ideas — only killed/archived ones below threshold
        res_del = db.table("ideas") \
            .delete() \
            .lt("diamond_score", score_delete) \
            .gt("diamond_score", 0) \
            .in_("status", ["killed", "archived"]) \
            .lt("created_at", cutoff) \
            .execute()
        counts["deleted"] = len(res_del.data) if res_del.data else 0
        logger.info(f"[DB] Cleanup: deleted {counts['deleted']} ideas (score < {score_delete})")
    except Exception as e:
        logger.error(f"[DB] cleanup delete failed: {e}")

    try:
        # ARCHIVE: score 3-5, status=active, older than min_age_days
        res_arch = db.table("ideas") \
            .update({"status": "archived"}) \
            .gte("diamond_score", score_delete) \
            .lt("diamond_score", score_archive) \
            .eq("status", "active") \
            .lt("created_at", cutoff) \
            .execute()
        counts["archived"] = len(res_arch.data) if res_arch.data else 0
        logger.info(f"[DB] Cleanup: archived {counts['archived']} ideas (score {score_delete}-{score_archive})")
    except Exception as e:
        logger.error(f"[DB] cleanup archive failed: {e}")

    return counts


def save_daily_summary(cycle_id: str, metadata: dict, date_str: str, counts: dict) -> bool:
    """Save daily summary to Supabase — called at end of Cycle 4."""
    db = get_client()
    try:
        # FIX: on_conflict="date" ensures upsert merges on the unique date key
        # Without this, Supabase falls back to INSERT and raises 23505 on cycle 4
        db.table("daily_summaries").upsert(_sanitize({
            "cycle_id":          cycle_id,
            "date":              date_str,
            "executive_summary": metadata.get("executive_summary", ""),
            "diamonds":          metadata.get("diamonds", []),
            "patterns":          metadata.get("cross_domain_patterns", []),
            "next_cycle_plan":   metadata.get("next_cycle_plan", {}),
            "ideas_generated":   counts.get("generated", 0),
            "ideas_killed":      counts.get("killed", 0),
            "diamonds_found":    counts.get("diamonds", 0),
        }), on_conflict="date").execute()
        return True
    except Exception as e:
        logger.error(f"[DB] save_daily_summary failed: {e}")
        return False


def get_daily_summary(date_str: str) -> Optional[dict]:
    """Fetch a specific day's summary."""
    db = get_client()
    try:
        res = db.table("daily_summaries").select("*").eq("date", date_str).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"[DB] get_daily_summary failed: {e}")
        return None


# ── Search Strategy (Feedback Loop) ─────────────────────────────────────────

def save_search_strategy(strategy: dict) -> bool:
    """
    Save search strategy to daily_summaries.next_cycle_plan.
    Updates existing row for today if it exists, inserts otherwise.
    """
    db = get_client()
    from datetime import datetime, timezone
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        # FIX: use upsert with on_conflict instead of update+fallback-insert
        # The old pattern caused 23505 duplicate key errors on rapid cycle runs
        db.table("daily_summaries").upsert(_sanitize({
            "date":            date_str,
            "next_cycle_plan": strategy,
        }), on_conflict="date").execute()
        return True
    except Exception as e:
        logger.error(f"[DB] save_search_strategy failed: {e}")
        return False


def load_search_strategy() -> dict:
    """
    Load the most recent search strategy from daily_summaries.
    Merges up to 7 days of history so memory accumulates across days.
    Falls back to seed.json on day 1.
    """
    db = get_client()
    merged: dict = {}
    try:
        res = (
            db.table("daily_summaries")
            .select("next_cycle_plan, date")
            .order("date", desc=True)
            .limit(7)
            .execute()
        )
        for row in (res.data or []):
            plan = row.get("next_cycle_plan") or {}
            if not plan:
                continue
            if not merged:
                # Most recent day wins for scalars
                merged = dict(plan)
            else:
                # Accumulate lists from older days (titles, keywords, patterns)
                for key in ("killed_idea_titles", "kill_patterns", "new_keywords"):
                    existing = merged.get(key, [])
                    older    = plan.get(key, [])
                    combined = list(dict.fromkeys(existing + older))  # dedup, preserve order
                    merged[key] = combined[:30]  # cap to avoid prompt bloat
        if merged:
            logger.info(
                f"[DB] load_search_strategy: loaded — "
                f"{len(merged.get('killed_idea_titles', []))} killed titles in memory, "
                f"{len(merged.get('new_keywords', []))} keywords"
            )
            return merged
    except Exception as e:
        logger.error(f"[DB] load_search_strategy failed: {e}")

    # ── Day-1 / cold-start fallback ──────────────────────────────────────────
    try:
        import json
        from pathlib import Path
        seed_path = Path("config/seed.json")
        if seed_path.exists():
            with seed_path.open() as f:
                seed = json.load(f)
            logger.info("[DB] load_search_strategy: no DB history — using seed.json defaults")
            return {
                "priority_domains":  seed.get("domains", []),
                "new_keywords":      seed.get("seed_keywords", [])[:10],
                "target_companies":  seed.get("target_companies", []),
                "kill_patterns":     [],
                "top_domains":       [],
                "weak_domains":      [],
                "killed_idea_titles": [],
            }
    except Exception as e:
        logger.warning(f"[DB] seed.json fallback failed: {e}")

    return {}


def load_cycle_memory(days_back: int = 14) -> dict:
    """
    Deep memory loader — returns structured summary of what the engine has
    learned over the past N days. Used by Cycle 1 harvest context so agents
    avoid repeating patterns and focus on genuinely new territory.

    Returns:
        {
          "killed_idea_titles":    [...],   # up to 40 — never generate these again
          "active_idea_titles":    [...],   # up to 20 — context for novelty
          "diamond_titles":        [...],   # what worked — inform direction
          "top_domains":           [...],   # domains with highest survival rate
          "weak_domains":          [...],   # domains that keep getting killed
          "kill_patterns":         [...],   # domain-level kill reasons
          "recent_finding_titles": [...],   # last 30 finding titles
          "evolved_keywords":      [...],   # keywords derived from surviving ideas
        }
    """
    db_client = get_client()
    result = {
        "killed_idea_titles":    [],
        "active_idea_titles":    [],
        "diamond_titles":        [],
        "top_domains":           [],
        "weak_domains":          [],
        "kill_patterns":         [],
        "recent_finding_titles": [],
        "evolved_keywords":      [],
    }
    try:
        from datetime import datetime, timezone, timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

        # All ideas from the last N days
        ideas_res = (
            db_client.table("ideas")
            .select("title, domain, status, kill_reason")
            .gte("created_at", since)
            .execute()
        )
        for idea in (ideas_res.data or []):
            title  = (idea.get("title") or "")[:80]
            status = idea.get("status", "")
            domain = idea.get("domain", "")
            if status == "killed":
                result["killed_idea_titles"].append(title)
                if domain:
                    result["kill_patterns"].append(domain)
            elif status == "diamond":
                result["diamond_titles"].append(title)
                # Extract keywords from diamond titles
                words = title.lower().split()
                for i in range(len(words) - 1):
                    phrase = " ".join(words[i:i+2])
                    if len(phrase) > 6:
                        result["evolved_keywords"].append(phrase)
            elif status in ("active", "archived"):
                result["active_idea_titles"].append(title)

        # Dedup and cap lists
        result["killed_idea_titles"]  = list(dict.fromkeys(result["killed_idea_titles"]))[:40]
        result["active_idea_titles"]  = list(dict.fromkeys(result["active_idea_titles"]))[:20]
        result["diamond_titles"]      = list(dict.fromkeys(result["diamond_titles"]))[:10]
        result["kill_patterns"]       = list(dict.fromkeys(result["kill_patterns"]))
        result["evolved_keywords"]    = list(dict.fromkeys(result["evolved_keywords"]))[:20]

        # Domain success rates
        rates = get_domain_success_rates(days_back=days_back)
        if rates:
            sorted_domains = sorted(
                [(d, r) for d, r in rates.items() if r is not None],
                key=lambda x: x[1], reverse=True
            )
            result["top_domains"]  = [d for d, _ in sorted_domains[:4]]
            result["weak_domains"] = [d for d, r in sorted_domains if r < 0.15]

        # Recent findings
        findings_res = (
            db_client.table("findings")
            .select("title")
            .gte("created_at", since)
            .order("confidence", desc=True)
            .limit(30)
            .execute()
        )
        result["recent_finding_titles"] = [
            (r.get("title") or "")[:80]
            for r in (findings_res.data or [])
        ]

        logger.info(
            f"[DB] load_cycle_memory: "
            f"{len(result['killed_idea_titles'])} killed, "
            f"{len(result['diamond_titles'])} diamonds, "
            f"{len(result['active_idea_titles'])} active, "
            f"{len(result['evolved_keywords'])} keywords"
        )

    except Exception as e:
        logger.error(f"[DB] load_cycle_memory failed: {e}")

    return result


def get_domain_success_rates(days_back: int = 14) -> dict:
    """
    Compute per-domain idea success rate over the last N days.
    Used by strategy feedback loop.
    Returns: {"thermal": 0.4, "power": 0.1, ...}
    """
    db = get_client()
    try:
        from datetime import datetime, timezone, timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        res = db.table("ideas").select("domain, status").gte("created_at", since).execute()
        rows = res.data or []

        from collections import defaultdict
        total   = defaultdict(int)
        survived = defaultdict(int)
        for row in rows:
            d = row.get("domain", "unknown")
            total[d] += 1
            if row.get("status") in ("active", "diamond", "archived"):
                survived[d] += 1

        rates = {}
        for d, t in total.items():
            rates[d] = round(survived[d] / t, 2) if t >= 3 else None  # need ≥3 ideas to be meaningful
        return rates
    except Exception as e:
        logger.error(f"[DB] get_domain_success_rates failed: {e}")
        return {}


def load_killed_ideas_sample(limit: int = 20) -> list[dict]:
    """Load recent killed ideas for strategy analysis."""
    db = get_client()
    try:
        res = (
            db.table("ideas")
            .select("domain, kill_reason, title")
            .eq("status", "killed")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"[DB] load_killed_ideas_sample failed: {e}")
        return []


# ── v5: Extended kill sample with full data for HypothesisGenerator ───────────

def load_killed_ideas_full(limit: int = 60) -> list[dict]:
    """Load recent killed ideas with full data for HypothesisGenerator."""
    db = get_client()
    try:
        res = (
            db.table("ideas")
            .select("id, domain, kill_reason, title, physics_score, market_score")
            .eq("status", "killed")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"[DB] load_killed_ideas_full failed: {e}")
        return []


def save_hypotheses(hypotheses: list[dict], cycle_id: str) -> bool:
    """Persist hypotheses from HypothesisGenerator to DB."""
    db = get_client()
    try:
        rows = []
        for h in hypotheses:
            rows.append({
                "cycle_id":          cycle_id,
                "hypothesis_id":     h.get("id", "H?"),
                "title":             h.get("title", "")[:500],
                "assumed_constraint": h.get("assumed_constraint", "")[:1000],
                "challenge":         h.get("challenge", "")[:1000],
                "physical_basis":    h.get("physical_basis", "")[:500],
                "unlocks":           h.get("unlocks", []),
                "unlocked_idea_count": int(h.get("unlocked_idea_count", 0)),
                "testability":       h.get("testability", "")[:500],
                "confidence":        float(h.get("confidence", 0.5)),
                "priority":          h.get("priority", "medium"),
                "status":            "open",
            })
        if rows:
            db.table("hypotheses").insert(rows).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] save_hypotheses failed: {e}")
        return False


def load_open_hypotheses(limit: int = 10) -> list[dict]:
    """Load open hypotheses — injected into Cycle 1 researcher context."""
    db = get_client()
    try:
        res = (
            db.table("hypotheses")
            .select("hypothesis_id, title, assumed_constraint, challenge, testability, priority")
            .eq("status", "open")
            .order("confidence", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.warning(f"[DB] load_open_hypotheses failed (non-blocking): {e}")
        return []


def save_weekly_report(report: str, metadata: dict, week_start: str) -> bool:
    """Save weekly report to weekly_reports table."""
    db = get_client()
    try:
        db.table("weekly_reports").upsert({
            "week_start":       week_start,
            "report_markdown":  report,
            "metadata":         metadata,
            "quality":          metadata.get("week_quality", "average"),
        }).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] save_weekly_report failed: {e}")
        return False


def get_week_stats(days_back: int = 7) -> dict:
    """Aggregate stats for WeeklyAnalyst."""
    db = get_client()
    try:
        from datetime import datetime, timezone, timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

        ideas_res = db.table("ideas").select("status, diamond_score, domain, title, physics_score, market_score").gte("created_at", since).execute()
        rows = ideas_res.data or []

        stats = {
            "generated":      len(rows),
            "killed_physics": sum(1 for r in rows if r.get("status") == "killed" and "physics" in (r.get("kill_reason") or "").lower()),
            "killed_agents":  sum(1 for r in rows if r.get("status") == "killed" and "physics" not in (r.get("kill_reason") or "").lower()),
            "surviving":      sum(1 for r in rows if r.get("status") in ("active", "diamond")),
            "diamonds":       sum(1 for r in rows if r.get("status") == "diamond"),
            "avg_score":      round(sum(r.get("diamond_score", 0) for r in rows if r.get("diamond_score", 0) > 0) /
                              max(sum(1 for r in rows if r.get("diamond_score", 0) > 0), 1), 2),
        }

        top_ideas = sorted(
            [r for r in rows if r.get("diamond_score", 0) > 0],
            key=lambda x: x.get("diamond_score", 0), reverse=True
        )[:15]

        # Dead end domains: high volume, low avg score
        from collections import defaultdict
        domain_scores: dict = defaultdict(list)
        for r in rows:
            if r.get("diamond_score", 0) > 0:
                domain_scores[r.get("domain", "?")].append(r["diamond_score"])
        dead_ends = [
            {"domain": d, "count": len(scores), "avg_score": round(sum(scores)/len(scores), 2)}
            for d, scores in domain_scores.items()
            if len(scores) >= 3 and sum(scores)/len(scores) < 4.0
        ]

        return {"stats": stats, "top_ideas": top_ideas, "dead_end_domains": dead_ends}
    except Exception as e:
        logger.error(f"[DB] get_week_stats failed: {e}")
        return {}
        return False
