"""
fluid_dynamics.py — Deterministic fluid & liquid cooling physics validators.
No LLM. Returns (passed: bool, detail: str).

Physical constants:
  Water density at 20°C: 998 kg/m³
  Water dynamic viscosity at 20°C: 1.002e-3 Pa·s
  Water vapor pressure at 20°C: 2338 Pa
  Specific heat water: 4182 J/(kg·K)
  Turbulent flow: Re > 4000
  Laminar flow: Re < 2300
"""
from __future__ import annotations
import math
from typing import Tuple

# ── Fluid properties (water at 20°C) ─────────────────────────────────────────
WATER_DENSITY_KG_M3     = 998.0
WATER_VISCOSITY_PA_S    = 1.002e-3
WATER_VAPOR_PRESSURE_PA = 2338.0
WATER_CP_J_KG_K         = 4182.0

# Coolant additives change these — typical 30% glycol mix
GLYCOL_MIX_DENSITY      = 1040.0
GLYCOL_MIX_VISCOSITY    = 2.0e-3   # higher viscosity → worse flow
GLYCOL_MIX_CP           = 3800.0

# ── Corrosion / galvanic series thresholds ────────────────────────────────────
# EMF potential difference > 0.25V between metals → significant galvanic corrosion
GALVANIC_EMF_LIMIT_V    = 0.25

GALVANIC_POTENTIAL_V = {
    "copper":    0.34,
    "aluminum": -1.66,
    "steel":    -0.44,
    "stainless": -0.13,
    "nickel":    -0.25,
    "zinc":      -0.76,
}


def check_reynolds_number(
    velocity_m_s: float,
    hydraulic_diameter_m: float,
    fluid: str = "water",
) -> Tuple[bool, str]:
    """
    Compute Reynolds number and classify flow regime.
    Turbulent flow (Re > 4000) is desirable for heat transfer.
    Very high Re (> 50000) → noise/vibration concern.
    """
    density   = WATER_DENSITY_KG_M3 if fluid == "water" else GLYCOL_MIX_DENSITY
    viscosity = WATER_VISCOSITY_PA_S if fluid == "water" else GLYCOL_MIX_VISCOSITY

    if velocity_m_s <= 0 or hydraulic_diameter_m <= 0:
        return False, "Velocity and diameter must be positive"

    Re     = (density * velocity_m_s * hydraulic_diameter_m) / viscosity
    passed = Re > 4000

    if Re < 2300:
        regime = "LAMINAR — poor heat transfer"
    elif Re < 4000:
        regime = "TRANSITIONAL — unstable"
    elif Re < 50000:
        regime = "TURBULENT — good heat transfer"
    else:
        regime = "HIGHLY TURBULENT — noise/erosion risk"

    detail = (
        f"Re = {Re:.0f} | Velocity: {velocity_m_s:.3f} m/s | "
        f"D_h: {hydraulic_diameter_m*1000:.2f} mm | "
        f"Fluid: {fluid} | Regime: {regime}"
    )
    return passed, detail


def check_cavitation(
    static_pressure_pa: float,
    fluid_velocity_m_s: float,
    fluid: str = "water",
    npsh_required_m: float = 2.0,
) -> Tuple[bool, str]:
    """
    Cavitation check: occurs when local pressure drops below vapor pressure.
    Uses Bernoulli: P_local = P_static - 0.5 × ρ × v²
    NPSH available must exceed NPSH required.
    """
    density       = WATER_DENSITY_KG_M3 if fluid == "water" else GLYCOL_MIX_DENSITY
    vapor_p       = WATER_VAPOR_PRESSURE_PA  # simplified — use water vapor pressure

    g             = 9.81
    dynamic_p     = 0.5 * density * fluid_velocity_m_s**2
    local_p       = static_pressure_pa - dynamic_p

    npsh_avail_m  = (static_pressure_pa - vapor_p) / (density * g)
    passed        = npsh_avail_m >= npsh_required_m and local_p > vapor_p

    detail = (
        f"Local pressure: {local_p:.0f} Pa | "
        f"Vapor pressure: {vapor_p:.0f} Pa | "
        f"NPSH available: {npsh_avail_m:.2f} m | "
        f"NPSH required: {npsh_required_m:.2f} m — "
        f"{'OK' if passed else 'FAIL — cavitation risk: pump damage / flow instability'}"
    )
    return passed, detail


def check_pressure_drop(
    flow_rate_m3_s: float,
    pipe_length_m: float,
    pipe_diameter_m: float,
    fluid: str = "water",
    max_drop_kpa: float = 50.0,
) -> Tuple[bool, str]:
    """
    Darcy-Weisbach pressure drop.
    Uses Blasius friction factor for turbulent flow: f = 0.316 × Re^(-0.25)
    """
    density   = WATER_DENSITY_KG_M3 if fluid == "water" else GLYCOL_MIX_DENSITY
    viscosity = WATER_VISCOSITY_PA_S if fluid == "water" else GLYCOL_MIX_VISCOSITY

    area      = math.pi * (pipe_diameter_m / 2)**2
    velocity  = flow_rate_m3_s / area
    Re        = density * velocity * pipe_diameter_m / viscosity

    if Re < 1:
        return False, "Flow rate too low to compute meaningfully"

    # Friction factor
    if Re < 2300:
        f = 64 / Re   # Hagen-Poiseuille (laminar)
    else:
        f = 0.316 * Re**(-0.25)  # Blasius (turbulent, smooth pipe)

    delta_p_pa  = f * (pipe_length_m / pipe_diameter_m) * 0.5 * density * velocity**2
    delta_p_kpa = delta_p_pa / 1000
    passed      = delta_p_kpa <= max_drop_kpa

    detail = (
        f"ΔP: {delta_p_kpa:.2f} kPa | Limit: {max_drop_kpa:.1f} kPa | "
        f"Re: {Re:.0f} | f: {f:.4f} | Velocity: {velocity:.3f} m/s — "
        f"{'OK' if passed else 'FAIL — pressure drop exceeds pump capacity'}"
    )
    return passed, detail


def check_cooling_capacity(
    flow_rate_m3_s: float,
    delta_t_k: float,
    required_power_w: float,
    fluid: str = "water",
) -> Tuple[bool, str]:
    """
    Check if coolant flow can remove the required heat load.
    Q = ṁ × Cp × ΔT = ρ × flow_rate × Cp × ΔT
    """
    density = WATER_DENSITY_KG_M3 if fluid == "water" else GLYCOL_MIX_DENSITY
    cp      = WATER_CP_J_KG_K if fluid == "water" else GLYCOL_MIX_CP

    mass_flow    = density * flow_rate_m3_s
    capacity_w   = mass_flow * cp * delta_t_k
    passed       = capacity_w >= required_power_w
    margin_pct   = (capacity_w - required_power_w) / required_power_w * 100

    detail = (
        f"Cooling capacity: {capacity_w:.1f} W | Required: {required_power_w:.1f} W | "
        f"Margin: {margin_pct:+.1f}% | "
        f"Flow: {flow_rate_m3_s*1000:.3f} L/s | ΔT: {delta_t_k:.1f} K — "
        f"{'OK' if passed else 'FAIL — insufficient cooling capacity'}"
    )
    return passed, detail


def check_galvanic_corrosion(
    metal_a: str,
    metal_b: str,
) -> Tuple[bool, str]:
    """
    Check galvanic corrosion risk between two metals in coolant loop.
    EMF difference > 0.25V → significant corrosion of anodic metal.
    """
    pot_a = GALVANIC_POTENTIAL_V.get(metal_a.lower())
    pot_b = GALVANIC_POTENTIAL_V.get(metal_b.lower())

    if pot_a is None or pot_b is None:
        unknown = metal_a if pot_a is None else metal_b
        return True, f"Warning: galvanic potential unknown for '{unknown}' — verify manually"

    diff   = abs(pot_a - pot_b)
    passed = diff <= GALVANIC_EMF_LIMIT_V
    anode  = metal_a if pot_a < pot_b else metal_b

    detail = (
        f"EMF difference: {diff:.3f} V | Limit: {GALVANIC_EMF_LIMIT_V} V | "
        f"Anode (corrodes): {anode} — "
        f"{'OK' if passed else f'FAIL — galvanic corrosion risk, {anode} will degrade'}"
    )
    return passed, detail
