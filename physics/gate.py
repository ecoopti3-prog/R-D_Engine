"""
gate.py — Physics Gate orchestrator.
Runs all applicable validators against an Idea and returns PhysicsVerdict.
This is the hard wall. No LLM. Deterministic. Final.
"""
from __future__ import annotations
import logging
from core.schemas import Idea, PhysicsVerdict
from physics.thermal import (
    check_heat_flux, check_carnot_efficiency,
    check_junction_temp, check_thermal_resistance
)
from physics.electrical import (
    check_energy_per_op, check_power_density, check_voltage_scaling,
    check_ir_drop, check_pdn_impedance, check_power_bump_density, check_decap_sufficiency
)
from physics.data_movement import check_bandwidth_wall, check_memory_latency, check_interconnect
from physics.mechanical import (
    check_stress_vs_yield, check_fatigue_life, check_vibration_resonance
)
from physics.fluid_dynamics import (
    check_reynolds_number, check_cavitation, check_pressure_drop,
    check_cooling_capacity, check_galvanic_corrosion
)
from physics.electromechanical import (
    check_joule_heating, check_contact_resistance,
    check_motor_thermal_derating, check_bearing_fatigue_life
)

logger = logging.getLogger(__name__)


def run_physics_gate(idea: Idea) -> PhysicsVerdict:
    """
    Run all applicable physics checks for the idea's domain.
    Returns PhysicsVerdict with passed=False + kill_reason if any check fails.
    """
    checks_run = []
    details    = {}
    kill_reason = None

    def _run(name: str, result_fn, *args):
        nonlocal kill_reason
        passed, detail = result_fn(*args)
        checks_run.append(name)
        details[name] = detail
        if not passed and kill_reason is None:
            kill_reason = detail
        return passed

    all_passed = True

    # ── Thermal checks ────────────────────────────────────────────────────────
    if idea.thermal_params:
        tp = idea.thermal_params
        if tp.heat_flux_w_cm2 is not None:
            ok = _run("heat_flux", check_heat_flux, tp.heat_flux_w_cm2, tp.material or "default")
            if not ok: all_passed = False

        if tp.t_junction_c is not None:
            ok = _run("junction_temp", check_junction_temp, tp.t_junction_c)
            if not ok: all_passed = False

        if tp.cop_claimed is not None and tp.t_junction_c is not None and tp.t_ambient_c is not None:
            ok = _run("carnot", check_carnot_efficiency,
                      tp.t_junction_c, tp.t_ambient_c, tp.cop_claimed)
            if not ok: all_passed = False

        if tp.thermal_resistance_c_per_w is not None and tp.t_ambient_c is not None:
            total_power = (idea.power_params.watt if idea.power_params and idea.power_params.watt else 100.0)
            ok = _run("thermal_resistance", check_thermal_resistance,
                      tp.thermal_resistance_c_per_w, total_power, tp.t_ambient_c)
            if not ok: all_passed = False

    # ── Electrical checks ─────────────────────────────────────────────────────
    if idea.power_params:
        pp = idea.power_params
        if pp.energy_per_op_pj is not None:
            ok = _run("landauer", check_energy_per_op, pp.energy_per_op_pj)
            if not ok: all_passed = False

        if pp.power_density_w_cm2 is not None:
            ok = _run("power_density", check_power_density, pp.power_density_w_cm2)
            if not ok: all_passed = False

        if pp.voltage_v is not None:
            # Infer process node from voltage: modern AI chips are 3-5nm
            # H100/A100 = 4-5nm, MI300X = 5nm, next-gen = 3nm
            # Use 5nm as conservative default; 0.7V → likely 3nm, 0.85V+ → likely 5-7nm
            if pp.voltage_v <= 0.75:
                node_nm = 3.0
            elif pp.voltage_v <= 0.85:
                node_nm = 5.0
            else:
                node_nm = 7.0
            ok = _run("voltage_scaling", check_voltage_scaling, pp.voltage_v, node_nm)
            if not ok: all_passed = False

    # ── PDN checks ────────────────────────────────────────────────────────────
    if idea.pdn_params:
        pdn = idea.pdn_params
        if pdn.ir_drop_mv is not None and pdn.vdd_v is not None and pdn.current_a is not None:
            r_mohm = (pdn.ir_drop_mv / pdn.current_a) if pdn.current_a > 0 else 0
            ok = _run("pdn_ir_drop", check_ir_drop, pdn.current_a, r_mohm, pdn.vdd_v)
            if not ok: all_passed = False

        if pdn.pdn_impedance_mohm is not None and pdn.frequency_ghz is not None:
            ok = _run("pdn_impedance", check_pdn_impedance, pdn.pdn_impedance_mohm, pdn.frequency_ghz)
            if not ok: all_passed = False

        if pdn.bump_density_per_mm2 is not None and pdn.current_a is not None and pdn.bump_density_per_mm2 > 0:
            current_per_bump_ma = (pdn.current_a / pdn.bump_density_per_mm2) * 1000
            ok = _run("electromigration", check_power_bump_density,
                      pdn.bump_density_per_mm2, current_per_bump_ma)
            if not ok: all_passed = False

        if pdn.decap_nf is not None and pdn.di_dt_a_per_ns is not None and pdn.vdd_v is not None:
            ok = _run("decap", check_decap_sufficiency,
                      pdn.decap_nf, pdn.di_dt_a_per_ns, pdn.vdd_v)
            if not ok: all_passed = False

    # ── Data movement checks ──────────────────────────────────────────────────
    if idea.data_movement_params:
        dm = idea.data_movement_params
        if dm.bandwidth_gb_s is not None and dm.compute_tflops is not None:
            ok = _run("bandwidth_wall", check_bandwidth_wall, dm.bandwidth_gb_s, dm.compute_tflops)
            if not ok: all_passed = False

        if dm.latency_ns is not None:
            ok = _run("memory_latency", check_memory_latency, dm.latency_ns)
            if not ok: all_passed = False

    # ── Robotics / Mechanical checks ─────────────────────────────────────────
    if idea.domain in ("robotics_mechanical", "actuators_motors", "wiring_harness",
                        "fluid_dynamics", "liquid_cooling"):
        np_dict = idea.numerical_params if hasattr(idea, "numerical_params") and idea.numerical_params else {}

        # Mechanical stress check
        stress = np_dict.get("stress_mpa") or np_dict.get("applied_stress_mpa")
        if stress is not None:
            ok = _run("stress_vs_yield", check_stress_vs_yield, float(stress))
            if not ok: all_passed = False

        # Fatigue life check
        stress_amp = np_dict.get("stress_amplitude_mpa")
        uts        = np_dict.get("uts_mpa")
        if stress_amp is not None and uts is not None:
            ok = _run("fatigue_life", check_fatigue_life, float(stress_amp), float(uts))
            if not ok: all_passed = False

        # Vibration resonance check
        excitation = np_dict.get("excitation_freq_hz")
        natural    = np_dict.get("natural_freq_hz")
        if excitation is not None and natural is not None:
            ok = _run("vibration_resonance", check_vibration_resonance, float(excitation), float(natural))
            if not ok: all_passed = False

        # Joule heating check (wiring harness)
        current  = np_dict.get("current_a")
        r_contact= np_dict.get("contact_resistance_mohm")
        if current is not None and r_contact is not None:
            ok = _run("contact_resistance", check_contact_resistance, float(r_contact), float(current))
            if not ok: all_passed = False

        # Motor thermal derating
        ambient = np_dict.get("temperature_c") or np_dict.get("ambient_temp_c")
        rated_p = np_dict.get("rated_power_w")
        if ambient is not None and rated_p is not None:
            ok = _run("motor_derating", check_motor_thermal_derating, float(ambient), float(rated_p))
            if not ok: all_passed = False

        # Bearing fatigue life
        c_kn   = np_dict.get("dynamic_load_kn")
        p_kn   = np_dict.get("equivalent_load_kn")
        rpm    = np_dict.get("rpm")
        if c_kn is not None and p_kn is not None and rpm is not None:
            ok = _run("bearing_fatigue", check_bearing_fatigue_life, float(c_kn), float(p_kn), float(rpm))
            if not ok: all_passed = False

        # Cavitation check (liquid cooling)
        static_p = np_dict.get("pressure_pa") or np_dict.get("static_pressure_pa")
        velocity = np_dict.get("flow_velocity_m_s")
        if static_p is not None and velocity is not None:
            ok = _run("cavitation", check_cavitation, float(static_p), float(velocity))
            if not ok: all_passed = False

        # Cooling capacity check
        flow_rate = np_dict.get("flow_rate_m3_s")
        delta_t   = np_dict.get("delta_t_k") or np_dict.get("delta_t_c")
        power_req = np_dict.get("required_power_w")
        if flow_rate is not None and delta_t is not None and power_req is not None:
            ok = _run("cooling_capacity", check_cooling_capacity,
                      float(flow_rate), float(delta_t), float(power_req))
            if not ok: all_passed = False

    # ── Score based on gap from limits ────────────────────────────────────────
    if not checks_run:
        # No params extracted — cannot verify physics. Penalize score but don't kill.
        # Ideas with no extractable params are suspicious: either too vague or not quantitative.
        score = 2.0
        logger.warning(
            f"[PhysicsGate] idea={idea.id[:8]} | NO PARAMS EXTRACTED — score=2.0 (unverifiable)"
        )
    elif not all_passed:
        score = 0.0
    else:
        # Score based on number of checks passed — more checks = higher confidence
        score = min(8.0, 4.0 + len(checks_run) * 0.5)

    logger.info(
        f"[PhysicsGate] idea={idea.id[:8]} | "
        f"passed={all_passed} | checks={checks_run} | score={score}"
    )

    return PhysicsVerdict(
        idea_id=idea.id,
        passed=all_passed,
        score=score,
        checks_run=checks_run,
        kill_reason=kill_reason,
        details=details,
    )
