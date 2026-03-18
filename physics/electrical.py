"""
electrical.py — Electrical and PDN physics validators.
No LLM. Deterministic.

Constants:
  Landauer limit: kT*ln(2) ≈ 0.017 aJ = 1.7e-5 pJ at 300K
  CMOS breakdown: ~1V per nm (simplified; actual depends on oxide)
  Electromigration (Black's equation): J_max ~1e6 A/cm2 for Cu
  PDN IR drop budget: typically 2-5% of VDD
"""
from __future__ import annotations
import math
from typing import Tuple

# ── Physical constants ────────────────────────────────────────────────────────
K_BOLTZMANN    = 1.380649e-23   # J/K
T_ROOM_K       = 300.0
LANDAUER_J     = K_BOLTZMANN * T_ROOM_K * math.log(2)   # ~2.8e-21 J
LANDAUER_PJ    = LANDAUER_J * 1e12                       # ~2.8e-9 pJ = 0.0000028 pJ

# ── CMOS limits ───────────────────────────────────────────────────────────────
CMOS_BREAKDOWN_V_PER_NM = 0.8    # simplified; real devices use oxide thickness

# ── PDN ───────────────────────────────────────────────────────────────────────
IR_DROP_BUDGET_PCT  = 5.0        # 5% of VDD is standard budget
ELECTROMIGRATION_J_MAX_A_CM2 = 1e6  # Cu interconnect practical limit


def check_energy_per_op(pj_op: float) -> Tuple[bool, str]:
    """
    Validate energy per operation against Landauer limit.
    Nothing can compute below Landauer limit at room temperature.
    """
    if pj_op <= 0:
        return False, f"Energy per op must be positive, got {pj_op} pJ"
    if pj_op < LANDAUER_PJ:
        return (
            False,
            f"Energy per op {pj_op:.2e} pJ is below Landauer limit "
            f"{LANDAUER_PJ:.2e} pJ at {T_ROOM_K}K — physically impossible"
        )
    ratio = pj_op / LANDAUER_PJ
    return True, f"Energy per op {pj_op:.4f} pJ is {ratio:.0e}× above Landauer limit — OK"


def check_power_density(w_cm2: float) -> Tuple[bool, str]:
    """
    Validate power density against silicon breakdown limits.
    Current state-of-art: ~100 W/cm2 for logic (Hotstage limit)
    Theoretical CMOS breakdown: ~1000 W/cm2 before electrical failure
    """
    SILICON_PRACTICAL_LIMIT   = 100.0    # W/cm2 — cooling-limited in practice
    SILICON_ELECTRICAL_LIMIT  = 1000.0   # W/cm2 — oxide breakdown
    if w_cm2 > SILICON_ELECTRICAL_LIMIT:
        return (
            False,
            f"Power density {w_cm2} W/cm2 exceeds silicon electrical breakdown limit "
            f"{SILICON_ELECTRICAL_LIMIT} W/cm2"
        )
    if w_cm2 > SILICON_PRACTICAL_LIMIT:
        return (
            True,
            f"Power density {w_cm2} W/cm2 above practical cooling limit "
            f"{SILICON_PRACTICAL_LIMIT} W/cm2 — requires advanced cooling (flag for thermal check)"
        )
    return True, f"Power density {w_cm2} W/cm2 within practical limits — OK"


def check_voltage_scaling(voltage_v: float, node_nm: float) -> Tuple[bool, str]:
    """
    Validate voltage against CMOS reliability for given process node.
    Simplified: V_max ≈ 0.8V/nm × (oxide_thickness ≈ node_nm/30)
    """
    # Real 5nm: oxide ~1.3nm (high-k), 3nm: ~1.0nm
    # Formula: node_nm/10 is closer to reality than node_nm/30
    # Floor at 1.0nm — below this no commercial transistor operates
    oxide_thickness_nm = max(1.0, node_nm / 10.0)
    v_breakdown = CMOS_BREAKDOWN_V_PER_NM * oxide_thickness_nm
    if voltage_v > v_breakdown:
        return (
            False,
            f"Voltage {voltage_v}V exceeds estimated breakdown {v_breakdown:.2f}V "
            f"for {node_nm}nm node (oxide ~{oxide_thickness_nm:.1f}nm)"
        )
    return True, f"Voltage {voltage_v}V within breakdown limit {v_breakdown:.2f}V for {node_nm}nm — OK"


# ── PDN Validators ────────────────────────────────────────────────────────────

def check_ir_drop(current_a: float, resistance_mohm: float, vdd_v: float) -> Tuple[bool, str]:
    """
    Validate IR drop against budget (5% VDD).
    V_drop = I × R
    """
    r_ohm = resistance_mohm / 1000.0
    v_drop = current_a * r_ohm
    drop_pct = (v_drop / vdd_v) * 100 if vdd_v > 0 else 999
    budget_v = vdd_v * (IR_DROP_BUDGET_PCT / 100)
    if v_drop > budget_v:
        return (
            False,
            f"IR drop {v_drop*1000:.1f}mV ({drop_pct:.1f}% of VDD {vdd_v}V) "
            f"exceeds {IR_DROP_BUDGET_PCT}% PDN budget ({budget_v*1000:.1f}mV)"
        )
    return True, f"IR drop {v_drop*1000:.1f}mV ({drop_pct:.1f}% of VDD) within budget — OK"


def check_pdn_impedance(z_mohm: float, freq_ghz: float) -> Tuple[bool, str]:
    """
    Validate PDN impedance profile.
    Target impedance: Z_target = VDD_ripple% × VDD / I_max
    Simplified check: at high freq, Z > 10 mohm causes timing violations.
    """
    Z_LIMIT_HIGH_FREQ_MOHM = 10.0   # practical limit above 1GHz
    if freq_ghz > 1.0 and z_mohm > Z_LIMIT_HIGH_FREQ_MOHM:
        return (
            False,
            f"PDN impedance {z_mohm}mohm at {freq_ghz}GHz exceeds high-freq limit "
            f"{Z_LIMIT_HIGH_FREQ_MOHM}mohm — will cause voltage noise and timing violations"
        )
    return True, f"PDN impedance {z_mohm}mohm at {freq_ghz}GHz within limit — OK"


def check_power_bump_density(bumps_per_mm2: float, current_per_bump_ma: float) -> Tuple[bool, str]:
    """
    Validate power bump density against electromigration limit.
    Bump diameter ~80μm for standard flip-chip → area ~5e-5 cm2
    J = I / A
    """
    BUMP_AREA_CM2 = 5e-5   # ~80μm diameter bump
    current_a     = current_per_bump_ma / 1000.0
    j_a_cm2       = current_a / BUMP_AREA_CM2
    if j_a_cm2 > ELECTROMIGRATION_J_MAX_A_CM2:
        return (
            False,
            f"Current density {j_a_cm2:.2e} A/cm2 per bump exceeds "
            f"electromigration limit {ELECTROMIGRATION_J_MAX_A_CM2:.0e} A/cm2"
        )
    return True, f"Current density {j_a_cm2:.2e} A/cm2 per bump within electromigration limit — OK"


def check_decap_sufficiency(c_nf: float, di_dt_a_per_ns: float, vdd_v: float, allowed_droop_pct: float = 5.0) -> Tuple[bool, str]:
    """
    Validate decoupling capacitance against di/dt transients.
    V_droop = L * di/dt  (simplified: use C to limit droop)
    V = Q/C → ΔV = ΔI × Δt / C
    Δt ≈ 1ns for modern switching transients.
    """
    delta_t_s  = 1e-9   # 1ns transient
    c_farads   = c_nf * 1e-9
    di_a       = di_dt_a_per_ns * 1e9 * delta_t_s   # ≈ di_dt_a_per_ns amps
    v_droop    = di_a / c_farads * delta_t_s if c_farads > 0 else 999
    allowed_v  = vdd_v * (allowed_droop_pct / 100)
    if v_droop > allowed_v:
        return (
            False,
            f"Decap {c_nf}nF insufficient for di/dt {di_dt_a_per_ns}A/ns — "
            f"voltage droop {v_droop*1000:.1f}mV exceeds {allowed_droop_pct}% budget "
            f"({allowed_v*1000:.1f}mV at VDD={vdd_v}V)"
        )
    return True, f"Decap {c_nf}nF sufficient — droop {v_droop*1000:.1f}mV within budget — OK"
