"""
lineage.py — Idea lineage tracking.

Every Idea must know:
  1. Which source (paper URL / patent URL / job posting) generated it
  2. Which findings contributed to it
  3. Which agent created it

After 2+ weeks: we can rank sources by "diamond yield" —
sources that produced diamonds get higher weight in future searches.

This is the single most impactful learning mechanism:
  If arXiv paper X produced 3 diamonds → prioritize that author / venue / keyword cluster.
  If job postings from Google consistently produce better signals → weight them higher.
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def save_idea_lineage(
    idea_id: str,
    source_urls: list[str],
    source_types: list[str],
    finding_ids: list[str],
    agent_name: str,
    db_client,
) -> bool:
    """
    Save the lineage for an idea — which sources and findings created it.
    Called immediately after save_idea().
    """
    try:
        rows = []
        for i, url in enumerate(source_urls):
            if url:
                rows.append({
                    "idea_id": idea_id,
                    "source_url": url[:500],
                    "source_type": source_types[i] if i < len(source_types) else "unknown",
                    "agent_name": agent_name,
                })
        if rows:
            db_client.table("idea_lineage").insert(rows).execute()
            rows = []

        # Save finding_ids separately — only if they exist in findings table
        if finding_ids:
            existing = db_client.table("findings").select("id").in_("id", finding_ids).execute()
            valid_ids = {r["id"] for r in (existing.data or [])}
            for fid in finding_ids:
                if fid in valid_ids:
                    rows.append({
                        "idea_id": idea_id,
                        "finding_id": fid,
                        "agent_name": agent_name,
                    })
        if rows:
            db_client.table("idea_lineage").insert(rows).execute()
        return True
    except Exception as e:
        logger.warning(f"[Lineage] save failed (non-blocking): {e}")
        return False


def get_source_diamond_yield(db_client, days_back: int = 30) -> dict[str, float]:
    """
    Compute diamond yield per source URL.
    Returns: {"https://arxiv.org/...": 0.8, ...}
    Used by: Cycle 1 to weight which sources to prioritize.
    """
    try:
        from datetime import datetime, timezone, timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

        res = db_client.table("idea_lineage") \
            .select("source_url, idea_id") \
            .gte("created_at", since) \
            .not_.is_("source_url", "null") \
            .execute()

        idea_ids = list({r["idea_id"] for r in (res.data or [])})
        if not idea_ids:
            return {}

        # Get status of those ideas
        status_res = db_client.table("ideas") \
            .select("id, status") \
            .in_("id", idea_ids[:200]) \
            .execute()

        diamond_set = {r["id"] for r in (status_res.data or []) if r.get("status") == "diamond"}

        # Compute yield per source
        source_totals: dict[str, int] = {}
        source_diamonds: dict[str, int] = {}
        for row in (res.data or []):
            url = row.get("source_url", "")
            if not url:
                continue
            # Use domain/venue, not full URL
            venue = _extract_venue(url)
            source_totals[venue] = source_totals.get(venue, 0) + 1
            if row["idea_id"] in diamond_set:
                source_diamonds[venue] = source_diamonds.get(venue, 0) + 1

        return {
            venue: round(source_diamonds.get(venue, 0) / total, 3)
            for venue, total in source_totals.items()
            if total >= 2  # need at least 2 ideas to compute meaningful yield
        }
    except Exception as e:
        logger.warning(f"[Lineage] get_source_diamond_yield failed: {e}")
        return {}


def get_top_yielding_sources(db_client, top_n: int = 10, days_back: int = 30) -> list[dict]:
    """
    Return top N sources by diamond yield.
    Used to inject high-yield sources into next cycle's search context.
    """
    yields = get_source_diamond_yield(db_client, days_back)
    sorted_sources = sorted(yields.items(), key=lambda x: x[1], reverse=True)
    return [{"venue": v, "yield": y} for v, y in sorted_sources[:top_n]]


def _extract_venue(url: str) -> str:
    """Extract venue/domain from URL for grouping."""
    url = url.lower()
    if "arxiv.org" in url:
        return "arxiv"
    if "patents.google.com" in url or "patents.justia.com" in url:
        return "google_patents"
    if "semanticscholar.org" in url:
        return "semantic_scholar"
    if "huggingface.co" in url:
        return "huggingface"
    if "github.com" in url:
        return "github"
    if "linkedin.com" in url or "jobs" in url:
        return "job_postings"
    if "opencompute.org" in url or "opencompute" in url:
        return "opencompute"
    if "ieee.org" in url or "ieeexplore" in url:
        return "ieee"
    if "nature.com" in url:
        return "nature"
    if "proceedings.mlsys" in url or "mlsys" in url:
        return "mlsys"
    return "other"