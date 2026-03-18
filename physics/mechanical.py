"""
mechanical.py — Deterministic mechanical & robotics physics validators.
No LLM. Returns (passed: bool, detail: str).

Physical constants and limits:
  Young's modulus steel: ~200 GPa
  Yield strength (structural steel): ~250 MPa
  Fatigue limit steel (S-N, 10^7 cycles): ~0.5 × UTS
  Dynamic friction coefficient (steel/steel): 0.1–0.2
  Vibration: resonance near natural frequency → failure risk
"""
from __future__ import annotations
import math
from typing import Tuple

# ── Material properties ───────────────────────────────────────────────────────
YOUNG_MODULUS_GPA = {
    "steel":          200.0,
    "aluminum":        69.0,
    "titanium":       116.0,
    "carbon_fiber":   150.0,  # typical CFRP axial
    "nylon":            3.0,
    "default":        200.0,
}

YIELD_STRENGTH_MPA = {
    "structural_steel": 250.0,
    "high_strength_steel": 690.0,
    "aluminum_6061":    276.0,
    "titanium_6al4v":   880.0,
    "default":          250.0,
}

# Fatigue limit as fraction of UTS (Wöhler, 10^7 cycles, fully reversed)
FATIGUE_RATIO = {
    "steel":     0.50,
    "aluminum":  0.35,   # aluminum has no true endurance limit
    "titanium":  0.45,
    "default":   0.45,
}

# Dynamic friction coefficients (kinetic)
FRICTION_COEFF = {
    "steel_steel_dry":      0.15,
    "steel_steel_lubricated": 0.05,
    "nylon_steel":          0.25,
    "default":              0.15,
}

SAFETY_FACTOR_MIN = 2.0   # minimum accepted safety factor for structural loads


def check_stress_vs_yield(
    applied_stress_mpa: float,
    material: str = "default",
    safety_factor: float = SAFETY_FACTOR_MIN,
) -> Tuple[bool, str]:
    """
    Check if applied stress is safe relative to yield strength.
    Fails if applied_stress × safety_factor > yield_strength.
    """
    if applied_stress_mpa <= 0:
        return False, f"Applied stress must be positive, got {applied_stress_mpa} MPa"

    yield_s = YIELD_STRENGTH_MPA.get(material, YIELD_STRENGTH_MPA["default"])
    allowable = yield_s / safety_factor
    passed = applied_stress_mpa <= allowable
    detail = (
        f"Applied: {applied_stress_mpa:.1f} MPa | "
        f"Yield ({material}): {yield_s:.1f} MPa | "
        f"Allowable (SF={safety_factor}): {allowable:.1f} MPa — "
        f"{'OK' if passed else 'FAIL — yielding expected'}"
    )
    return passed, detail


def check_fatigue_life(
    stress_amplitude_mpa: float,
    uts_mpa: float,
    material: str = "default",
    target_cycles: float = 1e7,
) -> Tuple[bool, str]:
    """
    Estimate whether stress amplitude survives target_cycles using
    simplified Basquin/Wöhler approach.
    Fatigue limit = FATIGUE_RATIO × UTS at 10^7 cycles.
    """
    if stress_amplitude_mpa <= 0 or uts_mpa <= 0:
        return False, "Stress amplitude and UTS must be positive"

    ratio      = FATIGUE_RATIO.get(material, FATIGUE_RATIO["default"])
    endurance  = ratio * uts_mpa
    passed     = stress_amplitude_mpa <= endurance
    utilization = stress_amplitude_mpa / endurance

    detail = (
        f"Stress amplitude: {stress_amplitude_mpa:.1f} MPa | "
        f"Fatigue limit ({material}, 10^7 cycles): {endurance:.1f} MPa "
        f"({ratio*100:.0f}% × UTS {uts_mpa:.0f} MPa) | "
        f"Utilization: {utilization*100:.1f}% — "
        f"{'OK' if passed else 'FAIL — fatigue failure expected before 10^7 cycles'}"
    )
    return passed, detail


def check_vibration_resonance(
    excitation_freq_hz: float,
    natural_freq_hz: float,
    resonance_band_pct: float = 10.0,
) -> Tuple[bool, str]:
    """
    Warn if excitation frequency is within ±resonance_band_pct% of natural frequency.
    Near resonance → amplitude amplification → accelerated fatigue.
    """
    if natural_freq_hz <= 0:
        return False, "Natural frequency must be positive"

    lower = natural_freq_hz * (1 - resonance_band_pct / 100)
    upper = natural_freq_hz * (1 + resonance_band_pct / 100)
    in_band = lower <= excitation_freq_hz <= upper

    detail = (
        f"Excitation: {excitation_freq_hz:.2f} Hz | "
        f"Natural freq: {natural_freq_hz:.2f} Hz | "
        f"Resonance band: {lower:.2f}–{upper:.2f} Hz — "
        f"{'FAIL — resonance risk, redesign or add damping' if in_band else 'OK'}"
    )
    return not in_band, detail


def check_deflection(
    force_n: float,
    length_m: float,
    cross_section_m2: float,
    material: str = "default",
    max_deflection_m: float = 0.001,
) -> Tuple[bool, str]:
    """
    Simple cantilever deflection: δ = F·L³ / (3·E·I)
    Approximates I = cross_section² / (4π) for circular section.
    """
    if force_n <= 0 or length_m <= 0 or cross_section_m2 <= 0:
        return False, "Force, length, and cross-section must be positive"

    E_gpa  = YOUNG_MODULUS_GPA.get(material, YOUNG_MODULUS_GPA["default"])
    E_pa   = E_gpa * 1e9
    # Approximate moment of inertia for circular cross-section
    radius = math.sqrt(cross_section_m2 / math.pi)
    I      = math.pi * radius**4 / 4

    deflection = (force_n * length_m**3) / (3 * E_pa * I)
    passed     = deflection <= max_deflection_m

    detail = (
        f"Deflection: {deflection*1000:.3f} mm | "
        f"Limit: {max_deflection_m*1000:.3f} mm | "
        f"E ({material}): {E_gpa} GPa — "
        f"{'OK' if passed else 'FAIL — deflection exceeds limit'}"
    )
    return passed, detail
