"""
detector.py — Novelty Detection via pgvector cosine similarity.
No LLM. Pure math. sentence-transformers runs locally at $0.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Optional
from core.schemas import Idea, NoveltyResult
from config.settings import (
    NOVELTY_AUTO_KILL_THRESHOLD, NOVELTY_SIMILAR_FLAG_THRESHOLD,
    EMBEDDING_MODEL, EMBEDDING_DIM
)

logger = logging.getLogger(__name__)

_model = None

def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[Novelty] Loading embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except ImportError:
            logger.error("[Novelty] sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _model


def embed_idea(idea: Idea) -> list[float]:
    """
    Build embedding input from idea fields and return 384-dim vector.
    Uses: title + problem + physical_limit + domain
    """
    text = f"{idea.title}. {idea.problem}. {idea.physical_limit}. {idea.domain}"
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two normalized vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b))  # Already normalized → dot product = cosine similarity


def check_novelty(idea: Idea, archive_vectors: list[dict],
                  precomputed_embedding: list[float] = None) -> NoveltyResult:
    """
    Compare idea against all archived idea vectors.
    archive_vectors: list of {"id": str, "embedding": list[float]}
    precomputed_embedding: if provided, skip re-embedding the idea (saves one model call)

    Returns NoveltyResult with action: pass | flag | kill
    """
    if not archive_vectors:
        logger.info(f"[Novelty] Empty archive — Cold Start, novelty_score=10 for {idea.id[:8]}")
        return NoveltyResult(
            idea_id=idea.id,
            cosine_similarity=0.0,
            novelty_score=10.0,
            similar_to=None,
            action="pass",
        )

    try:
        new_vec = precomputed_embedding if precomputed_embedding is not None else embed_idea(idea)
    except Exception as e:
        logger.error(f"[Novelty] Embedding failed: {e}")
        return NoveltyResult(
            idea_id=idea.id,
            cosine_similarity=0.0,
            novelty_score=5.0,
            similar_to=None,
            action="pass",
        )

    max_sim = 0.0
    most_similar_id = None

    for archived in archive_vectors:
        emb = archived["embedding"]
        if isinstance(emb, str):
            import json as _json
            emb = _json.loads(emb)
        sim = cosine_similarity(new_vec, emb)
        if sim > max_sim:
            max_sim = sim
            most_similar_id = archived["id"]

    # Map cosine similarity → novelty score
    if max_sim > NOVELTY_AUTO_KILL_THRESHOLD:
        novelty_score = 0.0
        action = "kill"
    elif max_sim > NOVELTY_SIMILAR_FLAG_THRESHOLD:
        novelty_score = 3.0
        action = "flag"
    elif max_sim > 0.65:
        novelty_score = 6.0
        action = "pass"
    elif max_sim > 0.50:
        novelty_score = 8.0
        action = "pass"
    else:
        novelty_score = 10.0
        action = "pass"

    logger.info(
        f"[Novelty] idea={idea.id[:8]} | sim={max_sim:.3f} | "
        f"score={novelty_score} | action={action}"
    )

    return NoveltyResult(
        idea_id=idea.id,
        cosine_similarity=max_sim,
        novelty_score=novelty_score,
        similar_to=most_similar_id if action != "pass" else None,
        action=action,
    )


# ── v7: Parameter-based novelty check ────────────────────────────────────────
# Detects ideas that share the SAME failure parameter as killed ideas,
# even if textually different. This catches cases cosine similarity misses.

def check_param_novelty(idea: Idea, killed_sim_patterns: list[dict]) -> dict:
    """
    v7: Check if this idea is likely to fail on the same parameter as many killed ideas.

    killed_sim_patterns: from load_sim_kill_patterns() — list of
      {"domain": "thermal", "failure_count": 23, "avg_r_theta_actual": 0.21,
       "avg_r_theta_critical": 0.17, "avg_improvement_needed_pct": 19.0}

    Returns:
      {"flagged": bool, "reason": str, "matching_pattern": dict|None}
    """
    if not killed_sim_patterns:
        return {"flagged": False, "reason": "no sim patterns available", "matching_pattern": None}

    tp = idea.thermal_params
    pp = idea.power_params

    for pattern in killed_sim_patterns:
        domain = pattern.get("domain")
        count  = pattern.get("failure_count", 0)

        if count < 5:  # not enough data to flag
            continue

        if domain == "thermal" and tp:
            r_theta = tp.thermal_resistance_c_per_w
            r_theta_crit = pattern.get("avg_r_theta_critical")
            if r_theta and r_theta_crit and r_theta > r_theta_crit * 0.95:
                return {
                    "flagged": True,
                    "reason": (
                        f"R_theta={r_theta:.3f} C/W is near the measured critical limit "
                        f"{r_theta_crit:.3f} C/W that killed {count} similar ideas. "
                        f"Needs {pattern.get('avg_improvement_needed_pct','?'):.0f}% improvement to pass."
                    ),
                    "matching_pattern": pattern,
                }

    return {"flagged": False, "reason": "no matching failure pattern", "matching_pattern": None}
