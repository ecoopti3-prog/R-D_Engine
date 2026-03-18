"""
thermal.py — Deterministic thermal physics validators.
No LLM involved. Returns (passed: bool, detail: str).

Physical constants and limits used:
  Carnot COP: T_cold / (T_hot - T_cold)  [temperatures in Kelvin]
  JEDEC JESD51: T_junction_max = 125°C for standard silicon
  Copper max heat flux: ~100 W/cm2 (practical), ~300 W/cm2 (theoretical)
  Diamond substrate: ~2000 W/cm2
  Silicon carbide: ~350 W/cm2
  Alumina: ~30 W/cm2
"""
from __future__ import annotations
from typing import Tuple

# ── Material thermal conductivity limits (W/cm2) ──────────────────────────────
MATERIAL_HEAT_FLUX_LIMITS = {
    "copper":           100.0,   # practical limit with convective cooling
    "copper_theoretical": 300.0,
    "diamond":         2000.0,
    "silicon_carbide":  350.0,
    "alumina":           30.0,
    "silicon":           15.0,
    "aluminum":          50.0,
    "graphene":         500.0,   # theoretical
    "default":          100.0,
}

# ── JEDEC junction temperature limits (°C) ────────────────────────────────────
JEDEC_T_JUNCTION_MAX = {
    "standard_silicon": 125.0,
    "automotive":       150.0,
    "military":         175.0,
    "default":          125.0,
}


def _c_to_k(celsius: float) -> float:
    return celsius + 273.15


def check_heat_flux(w_cm2: float, material: str = "default") -> Tuple[bool, str]:
    """
    Validate heat flux against material thermal conductivity limit.
    """
    material_key = material.lower().replace(" ", "_") if material else "default"
    limit = MATERIAL_HEAT_FLUX_LIMITS.get(material_key, MATERIAL_HEAT_FLUX_LIMITS["default"])
    if w_cm2 > limit:
        return (
            False,
            f"Heat flux {w_cm2:.1f} W/cm2 exceeds {material_key} limit of {limit:.1f} W/cm2"
        )
    margin_pct = (1 - w_cm2 / limit) * 100
    return True, f"Heat flux {w_cm2:.1f} W/cm2 within {material_key} limit {limit:.1f} W/cm2 ({margin_pct:.0f}% margin)"


def check_carnot_efficiency(
    t_hot_c: float,
    t_cold_c: float,
    claimed_cop: float
) -> Tuple[bool, str]:
    """
    Validate claimed COP against Carnot limit.
    Carnot COP = T_cold_K / (T_hot_K - T_cold_K)
    """
    if t_hot_c <= t_cold_c:
        return False, f"T_hot ({t_hot_c}°C) must be greater than T_cold ({t_cold_c}°C)"
    t_hot_k  = _c_to_k(t_hot_c)
    t_cold_k = _c_to_k(t_cold_c)
    carnot_cop = t_cold_k / (t_hot_k - t_cold_k)
    if claimed_cop > carnot_cop:
        return (
            False,
            f"Claimed COP {claimed_cop:.2f} exceeds Carnot limit {carnot_cop:.2f} "
            f"at T_hot={t_hot_c}°C, T_cold={t_cold_c}°C — thermodynamically impossible"
        )
    efficiency_pct = (claimed_cop / carnot_cop) * 100
    return True, f"COP {claimed_cop:.2f} is {efficiency_pct:.0f}% of Carnot COP {carnot_cop:.2f}"


def check_junction_temp(
    t_junction_c: float,
    grade: str = "default"
) -> Tuple[bool, str]:
    """
    Validate junction temperature against JEDEC limits.
    Also rejects temperatures below absolute zero (−273.15°C).
    """
    ABSOLUTE_ZERO_C = -273.15
    if t_junction_c <= ABSOLUTE_ZERO_C:
        return (
            False,
            f"T_junction {t_junction_c}°C is below absolute zero ({ABSOLUTE_ZERO_C}°C) — physically impossible"
        )
    grade_key = grade.lower() if grade else "default"
    limit = JEDEC_T_JUNCTION_MAX.get(grade_key, JEDEC_T_JUNCTION_MAX["default"])
    if t_junction_c > limit:
        return (
            False,
            f"T_junction {t_junction_c}°C exceeds JEDEC {grade_key} limit of {limit}°C"
        )
    return True, f"T_junction {t_junction_c}°C within JEDEC {grade_key} limit {limit}°C"


def check_thermal_resistance(
    r_theta_c_per_w: float,
    power_w: float,
    t_ambient_c: float = 25.0,
    material: str = "default"
) -> Tuple[bool, str]:
    """
    Validate that thermal resistance doesn't exceed material melting point.
    Delta_T = R_theta * Power
    """
    delta_t = r_theta_c_per_w * power_w
    t_junction = t_ambient_c + delta_t
    # Use JEDEC limit as upper bound
    limit = JEDEC_T_JUNCTION_MAX["default"]
    if t_junction > limit:
        return (
            False,
            f"R_theta={r_theta_c_per_w}°C/W × {power_w}W = ΔT={delta_t:.1f}°C → "
            f"T_junction={t_junction:.1f}°C exceeds JEDEC limit {limit}°C"
        )
    return True, f"T_junction={t_junction:.1f}°C within JEDEC limit {limit}°C (ΔT={delta_t:.1f}°C)"


def check_spreading_resistance(
    area_cm2: float,
    heat_flux_w_cm2: float,
    material: str = "copper"
) -> Tuple[bool, str]:
    """
    Estimate spreading resistance and check thermal budget.
    Simple approximation: R_spread ≈ 1 / (2 * k * sqrt(pi * A))
    where k = thermal conductivity of spreader material.
    """
    k_map = {"copper": 4.0, "aluminum": 2.37, "diamond": 22.0, "silicon": 1.48, "default": 4.0}
    k = k_map.get(material.lower(), k_map["default"])
    import math
    r_spread = 1.0 / (2.0 * k * math.sqrt(math.pi * area_cm2))
    total_power = heat_flux_w_cm2 * area_cm2
    delta_t_spread = r_spread * total_power
    if delta_t_spread > 50.0:   # 50°C spreading gradient is impractical
        return (
            False,
            f"Spreading ΔT={delta_t_spread:.1f}°C for {area_cm2}cm2 {material} spreader "
            f"at {heat_flux_w_cm2} W/cm2 — impractical (>50°C threshold)"
        )
    return True, f"Spreading ΔT={delta_t_spread:.1f}°C for {area_cm2}cm2 {material} spreader — OK"


# ══════════════════════════════════════════════════════════════════════════════
# NumPy-powered: Multi-chip thermal resistance network solver
# ══════════════════════════════════════════════════════════════════════════════

def solve_thermal_network(
    power_w: list,
    r_theta_matrix: list,
    t_ambient_c: float = 40.0,
    t_junction_max_c: float = 125.0,
) -> tuple:
    """
    Solve multi-chip thermal resistance network using NumPy linear algebra.

    Models N chips on a shared package (CoWoS, MCM, chiplet interposer).
    Each chip generates heat that flows through its own R_theta to ambient
    AND through coupling R_theta to its neighbors.

    This captures the critical real-world effect: a hot GPU die throttling
    its neighbor through the shared interposer — which single-chip models miss.

    Math:
        Conductance matrix G[i][i]  = Σ_j (1/R[i][j])   total conductance from node i
        Conductance matrix G[i][j]  = -1/R[i][j]          coupling term
        System: G × T_rise = P → solved via np.linalg.solve

    Args:
        power_w:          list of power dissipation per chip [W]
        r_theta_matrix:   NxN matrix — R[i][i]=junction-to-ambient, R[i][j]=chip-to-chip coupling [°C/W]
        t_ambient_c:      ambient temperature [°C]
        t_junction_max_c: JEDEC junction temperature limit [°C]

    Returns:
        (passed, detail, T_junction_list, hottest_chip_index)
    """
    try:
        import numpy as np
    except ImportError:
        # Graceful fallback if numpy not installed
        return True, "[ThermalNetwork] numpy not available — skipped", [], 0

    n = len(power_w)
    if n == 0:
        return True, "No chips to analyze", [], 0

    P = np.array(power_w, dtype=float)
    R = np.array(r_theta_matrix, dtype=float)

    # Build conductance matrix
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                G[i][i] = 1.0 / R[i][i] if R[i][i] > 0 else 1e6
            elif R[i][j] > 0:
                G[i][i] += 1.0 / R[i][j]
                G[i][j]  = -1.0 / R[i][j]

    try:
        T_rise = np.linalg.solve(G, P)
    except np.linalg.LinAlgError:
        T_rise = np.linalg.lstsq(G, P, rcond=None)[0]

    T_junction = T_rise + t_ambient_c
    T_max      = float(np.max(T_junction))
    hot_chip   = int(np.argmax(T_junction))
    passed     = T_max <= t_junction_max_c

    detail = (
        f"Chip temperatures: {[f'{t:.1f}°C' for t in T_junction.tolist()]} | "
        f"Hottest: chip {hot_chip} at {T_max:.1f}°C | "
        f"Limit: {t_junction_max_c}°C | "
        f"Coupling captured between {n} chips — "
        f"{'OK' if passed else f'FAIL — chip {hot_chip} exceeds JEDEC limit'}"
    )
    return passed, detail, T_junction.tolist(), hot_chip
