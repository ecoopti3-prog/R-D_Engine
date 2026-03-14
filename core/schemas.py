"""
schemas.py — Pydantic v2 strict schemas for all agent I/O.
Every agent output is validated here. If validation fails → retry or skip.
"""
from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
import uuid


# ── Enums ─────────────────────────────────────────────────────────────────────

AgentStatus   = Literal["done", "failed", "skipped"]
IdeaStatus    = Literal["active", "killed", "archived", "diamond", "physics_unverified"]
KillCategory  = str  # free-form — LLM may return any value
FindingType   = Literal["bottleneck", "limit", "trend", "gap"]
Domain        = Literal["thermal", "power", "data_movement", "hardware", "pdn", "cross_domain", "packaging", "bandwidth", "memory", "interconnect", "compute"]
RelationType  = Literal["related", "contradicts", "extends", "duplicate"]


# ── Finding ───────────────────────────────────────────────────────────────────

class Finding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: FindingType
    domain: Domain
    title: str = Field(max_length=500)
    description: str
    source_url: Optional[str] = None
    source_type: str = "other"  # free-form — LLM may return any value
    company_signal: Optional[str] = None   # e.g. "NVIDIA", "TSMC" if from job posting
    confidence: float = Field(ge=0.0, le=1.0)
    numerical_params: dict = Field(default_factory=dict)   # raw extracted numbers


# ── Numerical Parameters (strict extraction) ──────────────────────────────────

class PowerParams(BaseModel):
    watt:              Optional[float] = None
    tdp_watt:          Optional[float] = None
    power_density_w_cm2: Optional[float] = None
    energy_per_op_pj:  Optional[float] = None
    voltage_v:         Optional[float] = None
    current_a:         Optional[float] = None
    efficiency_pct:    Optional[float] = None
    source_ref:        Optional[str]  = None

class ThermalParams(BaseModel):
    t_junction_c:      Optional[float] = None
    heat_flux_w_cm2:   Optional[float] = None
    thermal_resistance_c_per_w: Optional[float] = None
    delta_t_c:         Optional[float] = None
    t_ambient_c:       Optional[float] = None
    cop_claimed:       Optional[float] = None
    material:          Optional[str]  = None
    source_ref:        Optional[str]  = None

class DataMovementParams(BaseModel):
    bandwidth_gb_s:    Optional[float] = None
    latency_ns:        Optional[float] = None
    memory_capacity_gb: Optional[float] = None
    interconnect_speed_gb_s: Optional[float] = None
    compute_tflops:    Optional[float] = None
    source_ref:        Optional[str]  = None

class PDNParams(BaseModel):
    ir_drop_mv:        Optional[float] = None
    vdd_v:             Optional[float] = None
    pdn_impedance_mohm: Optional[float] = None
    frequency_ghz:     Optional[float] = None
    bump_density_per_mm2: Optional[float] = None
    current_a:         Optional[float] = None
    di_dt_a_per_ns:    Optional[float] = None
    decap_nf:          Optional[float] = None
    n_chiplets:        Optional[int]  = None
    source_ref:        Optional[str]  = None


# ── Diamond Score (partial — each agent fills what it knows) ──────────────────

class DiamondScorePartial(BaseModel):
    physics_feasibility: float = Field(default=0.0, ge=0.0, le=10.0)
    market_pain:         float = Field(default=0.0, ge=0.0, le=10.0)
    novelty:             float = Field(default=0.0, ge=0.0, le=10.0)
    scalability:         float = Field(default=0.0, ge=0.0, le=10.0)


# ── Idea ──────────────────────────────────────────────────────────────────────

class SimRevisionTarget(BaseModel):
    """Quantified target from Sim Engine inversion — what must change for this idea to pass."""
    parameter: str                      # e.g. "R_theta_c_per_w"
    current_value: Optional[float] = None
    required_value: Optional[float] = None
    delta_pct: Optional[float] = None   # how much improvement needed (%)
    domain: Optional[str] = None        # thermal / pdn / data_movement
    description: str = ""               # human-readable: "R_theta must drop from 0.20 to 0.17 C/W"


class SimFeedback(BaseModel):
    """Structured sim results injected back into RD Engine for HypothesisGenerator + PhysicsLimitMapper."""
    sim_score: float = Field(default=0.0, ge=0.0, le=10.0)
    overall_status: str = "skipped"     # pass / fail / marginal / critical
    near_miss: bool = False             # sim_score 4-6 AND fixable with single param change
    revision_targets: List[SimRevisionTarget] = Field(default_factory=list)
    # Key numerical results (not text — actual numbers for knowledge graph)
    r_theta_actual: Optional[float] = None
    r_theta_critical: Optional[float] = None
    t_op_c: Optional[float] = None
    margin_to_runaway_pct: Optional[float] = None
    yield_pct: Optional[float] = None
    min_mttf_years: Optional[float] = None
    top_failure_domain: Optional[str] = None   # which domain caused the kill
    timestamp: Optional[str] = None


class Idea(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    domain: Domain
    problem: str
    physical_limit: str
    proposed_direction: Optional[str] = None
    diamond_score_partial: DiamondScorePartial = Field(default_factory=DiamondScorePartial)
    power_params:         Optional[PowerParams]        = None
    thermal_params:       Optional[ThermalParams]      = None
    data_movement_params: Optional[DataMovementParams] = None
    pdn_params:           Optional[PDNParams]          = None
    status: IdeaStatus = "active"
    kill_reason: Optional[str] = None
    physics_kill_detail: Optional[str] = None
    company_context: Optional[str] = None   # e.g. "NVIDIA Blackwell bottleneck"
    # ── Sim Engine feedback (v7: closed loop) ─────────────────────────────────
    sim_feedback: Optional[SimFeedback] = None   # populated after Sim Engine runs


# ── Kill ──────────────────────────────────────────────────────────────────────

class Kill(BaseModel):
    idea_id: str
    killed_by: str
    reason: str
    kill_category: KillCategory
    evidence_url: Optional[str] = None


# ── Agent Output (universal wrapper) ─────────────────────────────────────────

class AgentOutput(BaseModel):
    agent: str
    cycle_id: str
    timestamp: str
    status: AgentStatus
    findings: List[Finding]  = Field(default_factory=list)
    ideas:    List[Idea]     = Field(default_factory=list)
    kills:    List[Kill]     = Field(default_factory=list)
    open_questions: List[str]= Field(default_factory=list)
    metadata: dict           = Field(default_factory=dict)
    llm_used: Optional[str]  = None
    tokens_used: Optional[int] = None
    duration_ms: Optional[int] = None


# ── Physics Gate result ───────────────────────────────────────────────────────

class PhysicsVerdict(BaseModel):
    idea_id: str
    passed: bool
    score: float = Field(ge=0.0, le=10.0)
    checks_run: List[str]
    kill_reason: Optional[str] = None
    details: dict = Field(default_factory=dict)


# ── Market result ─────────────────────────────────────────────────────────────

class MarketVerdict(BaseModel):
    idea_id: str
    market_pain_score: float = Field(ge=0.0, le=10.0)
    market_size_usd_bn: Optional[float] = None
    primary_buyer: Optional[str] = None
    roi_at_10pct: Optional[str] = None
    roi_at_50pct: Optional[str] = None


# ── Novelty result ────────────────────────────────────────────────────────────

class NoveltyResult(BaseModel):
    idea_id: str
    cosine_similarity: float
    novelty_score: float = Field(ge=0.0, le=10.0)
    similar_to: Optional[str] = None     # id of similar existing idea
    action: Literal["pass", "flag", "kill"]


# ── Cycle state ───────────────────────────────────────────────────────────────

class CycleState(BaseModel):
    cycle_id: str
    date: str
    cycle_number: int = Field(ge=1, le=4)
    status: Literal["running", "done", "failed"]
    last_agent_completed: Optional[str] = None
    ideas_generated: int = 0
    ideas_killed: int = 0
    diamonds_found: int = 0