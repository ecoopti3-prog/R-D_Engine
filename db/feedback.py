"""
feedback.py — Human-in-the-loop feedback system.

THE MISSING SIGNAL:
The system currently learns only from statistical kill/survive rates.
That's weak signal. The strongest signal is: a human expert looked at an idea and said
"this is genuinely interesting" or "this is obviously wrong."

This module:
1. Provides DB functions for saving/loading human feedback
2. Integrates feedback into ChiefScientist scoring (human rating overrides/boosts)
3. Tracks which feedback-rated ideas became diamonds vs. failed (meta-learning)

USAGE:
  From dashboard (index.html): POST to Supabase directly via anon key
  In Cycle 4: load_feedback_signals() injects ratings into ChiefScientist context

FEEDBACK SCALE:
  5 = "This is a real insight, I want to investigate this"
  4 = "Interesting direction, worth keeping"
  3 = "Plausible but I'm not excited"
  2 = "Probably wrong but I'm not sure why"
  1 = "This is clearly wrong — kill it"
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Boost/penalty to diamond_score from human feedback
FEEDBACK_WEIGHT = 0.15  # human rating contributes 15% of final score
FEEDBACK_SCALE = {5: 10.0, 4: 8.0, 3: 5.0, 2: 2.0, 1: 0.0}  # rating → score_contribution


def save_feedback(
    idea_id: str,
    rating: int,
    reason: str,
    db_client,
    rater: str = "human",
) -> bool:
    """Save a human feedback rating for an idea."""
    if rating not in (1, 2, 3, 4, 5):
        logger.error(f"[Feedback] Invalid rating {rating} — must be 1-5")
        return False
    try:
        db_client.table("idea_feedback").upsert({
            "idea_id": idea_id,
            "rating": rating,
            "reason": reason[:500] if reason else "",
            "rater": rater,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        logger.info(f"[Feedback] Saved rating={rating} for idea {idea_id[:8]}")
        return True
    except Exception as e:
        logger.error(f"[Feedback] save_feedback failed: {e}")
        return False


def load_feedback_signals(db_client, limit: int = 50) -> list[dict]:
    """
    Load recent feedback signals.
    Returns list of {idea_id, rating, reason, feedback_score}.
    Used by ChiefScientist to adjust final diamond scores.
    """
    try:
        res = db_client.table("idea_feedback") \
            .select("idea_id, rating, reason, created_at") \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()

        signals = []
        for row in (res.data or []):
            rating = row.get("rating", 3)
            signals.append({
                "idea_id": row["idea_id"],
                "rating": rating,
                "reason": row.get("reason", ""),
                "feedback_score": FEEDBACK_SCALE.get(rating, 5.0),
            })
        return signals
    except Exception as e:
        logger.warning(f"[Feedback] load_feedback_signals failed: {e}")
        return []


def apply_feedback_to_scores(
    scored_ideas: list[dict],
    feedback_signals: list[dict],
) -> list[dict]:
    """
    Adjust diamond scores based on human feedback.
    A rating=5 adds 15% weight toward 10.0. A rating=1 forces re-evaluation.
    """
    if not feedback_signals:
        return scored_ideas

    feedback_map = {s["idea_id"]: s for s in feedback_signals}

    for idea in scored_ideas:
        idea_id = idea.get("idea_id", "")
        fb = feedback_map.get(idea_id)
        if not fb:
            continue

        original_score = float(idea.get("diamond_score", 0))
        fb_score = fb["feedback_score"]

        # Blend: original_score * (1 - weight) + feedback_score * weight
        adjusted = original_score * (1 - FEEDBACK_WEIGHT) + fb_score * FEEDBACK_WEIGHT
        adjusted = round(min(10.0, max(0.0, adjusted)), 2)

        idea["diamond_score"] = adjusted
        idea["feedback_adjusted"] = True
        idea["human_rating"] = fb["rating"]
        idea["reasoning"] = (
            f"{idea.get('reasoning', '')} "
            f"[Human rating: {fb['rating']}/5 → score adjusted {original_score:.1f}→{adjusted:.1f}]"
        ).strip()

        # Force kill if rating=1
        if fb["rating"] == 1:
            idea["status"] = "killed"
            idea["reasoning"] += f" [HUMAN KILL: {fb.get('reason', 'rated 1/5')}]"

        logger.info(f"[Feedback] idea {idea_id[:8]} — rating={fb['rating']} adjusted score {original_score:.1f}→{adjusted:.1f}")

    return scored_ideas


def get_feedback_accuracy(db_client, days_back: int = 30) -> dict:
    """
    Meta-learning: how accurate is human feedback?
    Compare human ratings vs eventual diamond status.
    Used to calibrate FEEDBACK_WEIGHT over time.
    """
    try:
        from datetime import timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

        feedback_res = db_client.table("idea_feedback") \
            .select("idea_id, rating") \
            .gte("created_at", since) \
            .execute()

        if not feedback_res.data:
            return {}

        idea_ids = [r["idea_id"] for r in feedback_res.data]
        ideas_res = db_client.table("ideas") \
            .select("id, status") \
            .in_("id", idea_ids) \
            .execute()

        status_map = {r["id"]: r["status"] for r in (ideas_res.data or [])}

        stats = {"correct": 0, "incorrect": 0, "total": 0}
        for row in feedback_res.data:
            idea_id = row["idea_id"]
            rating = row["rating"]
            status = status_map.get(idea_id, "unknown")
            if status == "unknown":
                continue
            stats["total"] += 1
            # High rating (4-5) → should be active/diamond
            # Low rating (1-2) → should be killed
            if rating >= 4 and status in ("active", "diamond"):
                stats["correct"] += 1
            elif rating <= 2 and status == "killed":
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1

        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else None
        return {"accuracy": accuracy, **stats}
    except Exception as e:
        logger.warning(f"[Feedback] get_feedback_accuracy failed: {e}")
        return {}
