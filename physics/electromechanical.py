"""
electromechanical.py — Wiring harness & motor/actuator physics validators.
No LLM. Returns (passed: bool, detail: str).

Physical constants:
  Copper resistivity at 20°C: 1.72e-8 Ω·m
  Temperature coefficient of Cu: 0.00393 /°C
  PVC insulation max temp: 70°C continuous, 160°C peak
  XLPE insulation max temp: 90°C continuous
  Arc flash threshold: ~100 A through thin conductor → ionization
  NEC ampacity: conservative, based on 30°C ambient
"""
from __future__ import annotations
import math
from typing import Tuple

# ── Copper wire properties ────────────────────────────────────────────────────
CU_RESISTIVITY_OHM_M    = 1.72e-8
CU_TEMP_COEFF           = 0.00393   # per °C, referenced to 20°C

# ── Insulation thermal limits (°C) ───────────────────────────────────────────
INSULATION_TEMP_LIMIT = {
    "pvc":          70.0,
    "xlpe":         90.0,
    "ptfe":        200.0,
    "silicone":    180.0,
    "default":      70.0,
}

# ── Contact resistance limits (mΩ) ───────────────────────────────────────────
CONTACT_RESISTANCE_LIMIT_MOHM = {
    "new_connector":          5.0,
    "aged_connector":        20.0,
    "warning_threshold":     15.0,
    "failure_threshold":     50.0,
}

# ── Motor constants ───────────────────────────────────────────────────────────
# Typical servo motor efficiency
MOTOR_EFFICIENCY_TYPICAL = 0.92

# Bearing fatigue life constant (Lundberg-Palmgren, simplified)
# L10 life in hours = (C/P)^3 × 16667 / n_rpm
# C = dynamic load rating, P = equivalent dynamic load


def check_joule_heating(
    current_a: float,
    resistance_ohm: float,
    ambient_temp_c: float = 40.0,
    insulation: str = "default",
    thermal_resistance_c_per_w: float = 10.0,
) -> Tuple[bool, str]:
    """
    P = I²R, then estimate wire temperature rise via thermal resistance.
    Compare against insulation temperature limit.
    """
    if current_a <= 0 or resistance_ohm <= 0:
        return False, "Current and resistance must be positive"

    power_w    = current_a**2 * resistance_ohm
    temp_rise  = power_w * thermal_resistance_c_per_w
    wire_temp  = ambient_temp_c + temp_rise
    limit      = INSULATION_TEMP_LIMIT.get(insulation, INSULATION_TEMP_LIMIT["default"])
    passed     = wire_temp <= limit

    detail = (
        f"I²R loss: {power_w:.2f} W | "
        f"Wire temp: {wire_temp:.1f}°C | "
        f"Insulation limit ({insulation}): {limit:.0f}°C | "
        f"Margin: {limit - wire_temp:+.1f}°C — "
        f"{'OK' if passed else 'FAIL — insulation degradation / fire risk'}"
    )
    return passed, detail


def check_voltage_drop(
    current_a: float,
    wire_length_m: float,
    wire_cross_section_mm2: float,
    voltage_v: float,
    max_drop_pct: float = 3.0,
    temp_c: float = 60.0,
) -> Tuple[bool, str]:
    """
    Voltage drop in cable: V = 2 × I × R (round trip).
    Resistance adjusted for temperature.
    """
    if current_a <= 0 or wire_length_m <= 0 or wire_cross_section_mm2 <= 0:
        return False, "All inputs must be positive"

    # Adjust resistivity for temperature
    rho_t        = CU_RESISTIVITY_OHM_M * (1 + CU_TEMP_COEFF * (temp_c - 20))
    area_m2      = wire_cross_section_mm2 * 1e-6
    r_one_way    = rho_t * wire_length_m / area_m2
    v_drop       = 2 * current_a * r_one_way
    drop_pct     = (v_drop / voltage_v) * 100
    passed       = drop_pct <= max_drop_pct

    detail = (
        f"Voltage drop: {v_drop:.3f} V ({drop_pct:.2f}%) | "
        f"Limit: {max_drop_pct:.1f}% | "
        f"R (one-way): {r_one_way*1000:.3f} mΩ | "
        f"Wire: {wire_cross_section_mm2} mm², {wire_length_m} m, {temp_c}°C — "
        f"{'OK' if passed else 'FAIL — excessive voltage drop, upsize conductor'}"
    )
    return passed, detail


def check_contact_resistance(
    contact_resistance_mohm: float,
    current_a: float,
    ambient_temp_c: float = 40.0,
    insulation: str = "default",
) -> Tuple[bool, str]:
    """
    Check connector contact resistance for overheating risk.
    Power in contact = I² × R_contact; temperature rise estimated.
    """
    if contact_resistance_mohm < 0 or current_a <= 0:
        return False, "Contact resistance and current must be positive"

    r_ohm      = contact_resistance_mohm * 1e-3
    power_w    = current_a**2 * r_ohm
    # Connectors have ~5–15 °C/W thermal resistance
    temp_rise  = power_w * 8.0   # conservative estimate
    temp_c     = ambient_temp_c + temp_rise
    limit      = INSULATION_TEMP_LIMIT.get(insulation, INSULATION_TEMP_LIMIT["default"])
    threshold  = CONTACT_RESISTANCE_LIMIT_MOHM["warning_threshold"]
    fail_thr   = CONTACT_RESISTANCE_LIMIT_MOHM["failure_threshold"]

    if contact_resistance_mohm >= fail_thr:
        passed = False
        status = "FAIL — connector failure imminent"
    elif contact_resistance_mohm >= threshold:
        passed = False
        status = "WARNING — replace connector soon"
    elif temp_c > limit:
        passed = False
        status = "FAIL — thermal runaway in contact"
    else:
        passed = True
        status = "OK"

    detail = (
        f"Contact R: {contact_resistance_mohm:.1f} mΩ | "
        f"I²R at contact: {power_w:.3f} W | "
        f"Contact temp: {temp_c:.1f}°C | "
        f"Limit: {limit:.0f}°C — {status}"
    )
    return passed, detail


def check_motor_thermal_derating(
    ambient_temp_c: float,
    rated_power_w: float,
    rated_ambient_c: float = 40.0,
    derating_pct_per_degree: float = 1.0,
) -> Tuple[bool, str]:
    """
    Motor output must be derated at high ambient temps.
    Standard: ~1% per °C above rated ambient.
    """
    if ambient_temp_c <= rated_ambient_c:
        return True, f"Ambient {ambient_temp_c}°C ≤ rated {rated_ambient_c}°C — no derating needed"

    excess_c        = ambient_temp_c - rated_ambient_c
    derating_pct    = excess_c * derating_pct_per_degree
    derated_power_w = rated_power_w * (1 - derating_pct / 100)
    passed          = derating_pct < 25.0   # >25% derating → redesign needed

    detail = (
        f"Ambient: {ambient_temp_c}°C | Rated ambient: {rated_ambient_c}°C | "
        f"Derating: {derating_pct:.1f}% | "
        f"Available power: {derated_power_w:.1f} W (from rated {rated_power_w:.1f} W) — "
        f"{'OK' if passed else 'FAIL — excessive derating, select higher-rated motor'}"
    )
    return passed, detail


def check_bearing_fatigue_life(
    dynamic_load_rating_kn: float,
    equivalent_load_kn: float,
    rpm: float,
    target_hours: float = 20000.0,
) -> Tuple[bool, str]:
    """
    Lundberg-Palmgren L10 bearing life:
    L10 (millions of revs) = (C/P)^3
    L10 hours = L10_Mrev × 10^6 / (60 × rpm)
    """
    if equivalent_load_kn <= 0 or dynamic_load_rating_kn <= 0 or rpm <= 0:
        return False, "All inputs must be positive"

    ratio         = dynamic_load_rating_kn / equivalent_load_kn
    l10_mrev      = ratio**3
    l10_hours     = (l10_mrev * 1e6) / (60 * rpm)
    passed        = l10_hours >= target_hours

    detail = (
        f"L10 life: {l10_hours:.0f} h | Target: {target_hours:.0f} h | "
        f"C/P ratio: {ratio:.2f} | "
        f"C: {dynamic_load_rating_kn:.1f} kN | P: {equivalent_load_kn:.1f} kN | "
        f"Speed: {rpm:.0f} RPM — "
        f"{'OK' if passed else 'FAIL — bearing replacement required before target service life'}"
    )
    return passed, detail


def check_back_emf_limit(
    motor_kv_rpm_per_v: float,
    max_rpm: float,
    supply_voltage_v: float,
) -> Tuple[bool, str]:
    """
    Check that back-EMF at max RPM does not exceed supply voltage.
    If back-EMF ≥ supply voltage → motor stalls / controller shuts down.
    Back-EMF = max_rpm / Kv
    """
    if motor_kv_rpm_per_v <= 0 or max_rpm <= 0 or supply_voltage_v <= 0:
        return False, "All inputs must be positive"

    back_emf_v = max_rpm / motor_kv_rpm_per_v
    passed     = back_emf_v < supply_voltage_v * 0.95   # 5% headroom

    detail = (
        f"Back-EMF at {max_rpm:.0f} RPM: {back_emf_v:.2f} V | "
        f"Supply: {supply_voltage_v:.1f} V | "
        f"Headroom: {supply_voltage_v - back_emf_v:.2f} V — "
        f"{'OK' if passed else 'FAIL — motor cannot reach target RPM on this supply'}"
    )
    return passed, detail
