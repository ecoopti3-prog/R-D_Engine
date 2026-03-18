"""
test_physics.py — Unit tests for all physics validators.
Run: python main.py --test-physics
"""
from __future__ import annotations
import sys, logging

logger = logging.getLogger(__name__)

def run_all_tests():
    passed = 0
    failed = 0

    def check(name, result, expected_pass, expected_fragment=""):
        nonlocal passed, failed
        ok, detail = result
        if ok == expected_pass and (not expected_fragment or expected_fragment.lower() in detail.lower()):
            print(f"  ✓  {name}")
            passed += 1
        else:
            print(f"  ✗  {name}")
            print(f"       Expected pass={expected_pass}, got pass={ok}")
            print(f"       Detail: {detail}")
            failed += 1

    print("\n══════════════════════════════════════")
    print("  Physics Gate — Unit Tests")
    print("══════════════════════════════════════\n")

    # ── Thermal ──────────────────────────────────────────────────────────
    print("[ Thermal ]")
    from physics.thermal import (
        check_heat_flux, check_carnot_efficiency,
        check_junction_temp, check_thermal_resistance, check_spreading_resistance
    )
    check("Heat flux OK (copper, 50 W/cm2)",           check_heat_flux(50, "copper"),        True)
    check("Heat flux KILL (copper, 200 W/cm2)",        check_heat_flux(200, "copper"),       False)
    check("Heat flux OK (diamond, 1000 W/cm2)",        check_heat_flux(1000, "diamond"),     True)
    check("Heat flux KILL (diamond, 3000 W/cm2)",      check_heat_flux(3000, "diamond"),     False)
    check("Carnot OK (COP=2, 80→30°C)",                check_carnot_efficiency(80, 30, 2.0), True)
    check("Carnot KILL (COP=20, 80→30°C)",             check_carnot_efficiency(80, 30, 20.0),False)
    check("Junction OK (100°C)",                       check_junction_temp(100.0),           True)
    check("Junction KILL (150°C standard)",            check_junction_temp(150.0),           False)
    check("Junction OK (150°C automotive)",            check_junction_temp(150.0, "automotive"), True)
    check("Thermal resistance OK",                     check_thermal_resistance(0.1, 100, 25.0), True)
    check("Thermal resistance KILL",                   check_thermal_resistance(5.0, 200, 25.0), False)

    # ── Electrical ───────────────────────────────────────────────────────
    print("\n[ Electrical ]")
    from physics.electrical import (
        check_energy_per_op, check_power_density, check_voltage_scaling,
        check_ir_drop, check_pdn_impedance, check_power_bump_density, check_decap_sufficiency
    )
    check("Landauer OK (1 pJ/op)",                     check_energy_per_op(1.0),             True)
    check("Landauer KILL (below limit)",               check_energy_per_op(1e-12),            False)
    check("Power density OK (80 W/cm2)",               check_power_density(80),              True)
    check("Power density KILL (2000 W/cm2)",           check_power_density(2000),            False)
    check("Voltage scaling OK (0.8V, 5nm)",            check_voltage_scaling(0.8, 5),        True)
    check("Voltage scaling KILL (5V, 3nm)",            check_voltage_scaling(5.0, 3),        False)

    print("\n[ PDN ]")
    check("IR drop OK (5mV drop, VDD=1V)",             check_ir_drop(10, 0.5, 1.0),          True)
    check("IR drop KILL (100mV drop, VDD=1V)",         check_ir_drop(100, 1.0, 1.0),         False)
    check("PDN impedance OK (5mohm @ 2GHz)",           check_pdn_impedance(5.0, 2.0),        True)
    check("PDN impedance KILL (20mohm @ 2GHz)",        check_pdn_impedance(20.0, 2.0),       False)
    check("Decap OK (100nF, 1A/ns, VDD=1V)",           check_decap_sufficiency(100, 1.0, 1.0), True)
    check("Decap KILL (1nF, 100A/ns, VDD=1V)",         check_decap_sufficiency(1, 100.0, 1.0), False)

    # ── Data movement ────────────────────────────────────────────────────
    print("\n[ Data Movement ]")
    from physics.data_movement import check_bandwidth_wall, check_memory_latency, check_interconnect
    check("Bandwidth OK (HBM3 level)",                 check_bandwidth_wall(1000, 100),      True)
    check("Bandwidth KILL (impossible)",               check_bandwidth_wall(100000, 10),     False)
    check("DRAM latency OK (40ns)",                    check_memory_latency(40, "dram"),     True)
    check("DRAM latency KILL (1ns)",                   check_memory_latency(1, "dram"),      False)
    check("Interconnect OK (100 GB/s, 100mm)",         check_interconnect(100, 100),         True)
    check("Interconnect KILL (long distance)",         check_interconnect(500, 600),         False)


    # ── Mechanical / Robotics ────────────────────────────────────────────
    print("\n[ Mechanical / Robotics ]")
    from physics.mechanical import (
        check_stress_vs_yield, check_fatigue_life,
        check_vibration_resonance, check_deflection
    )
    check("Stress OK (100MPa, steel)",
          check_stress_vs_yield(100, "structural_steel"),          True)
    check("Stress KILL (300MPa, structural steel SF=2)",
          check_stress_vs_yield(200, "structural_steel"),          False)
    check("Fatigue OK (100MPa amplitude, UTS=500MPa, steel)",
          check_fatigue_life(100, 500, "steel"),                   True)
    check("Fatigue KILL (300MPa amplitude, UTS=500MPa, steel)",
          check_fatigue_life(300, 500, "steel"),                   False)
    check("Vibration OK (far from resonance)",
          check_vibration_resonance(10.0, 50.0),                   True)
    check("Vibration KILL (at resonance)",
          check_vibration_resonance(50.0, 50.0),                   False)

    # ── Fluid Dynamics / Liquid Cooling ──────────────────────────────────
    print("\n[ Fluid Dynamics / Liquid Cooling ]")
    from physics.fluid_dynamics import (
        check_reynolds_number, check_cavitation,
        check_pressure_drop, check_cooling_capacity, check_galvanic_corrosion
    )
    check("Reynolds OK (turbulent flow)",
          check_reynolds_number(1.5, 0.01, "water"),               True)
    check("Reynolds WARN (laminar, poor cooling)",
          check_reynolds_number(0.001, 0.01, "water"),             False)
    check("Cavitation OK (high pressure)",
          check_cavitation(200000, 1.0, "water"),                  True)
    check("Cavitation KILL (low pressure, high velocity)",
          check_cavitation(3000, 10.0, "water"),                   False)
    check("Cooling capacity OK",
          check_cooling_capacity(0.001, 10.0, 40.0, "water"),     True)
    check("Cooling capacity KILL (insufficient flow)",
          check_cooling_capacity(0.00001, 5.0, 5000.0, "water"),  False)
    check("Galvanic OK (nickel + stainless, diff=0.12V)",
          check_galvanic_corrosion("nickel", "stainless"),         True)
    check("Galvanic KILL (copper + aluminum)",
          check_galvanic_corrosion("copper", "aluminum"),          False)

    # ── Electromechanical (wiring harness + motors) ───────────────────────
    print("\n[ Electromechanical ]")
    from physics.electromechanical import (
        check_joule_heating, check_voltage_drop,
        check_contact_resistance, check_motor_thermal_derating,
        check_bearing_fatigue_life, check_back_emf_limit
    )
    check("Joule heating OK (5A, 0.1Ω, PVC)",
          check_joule_heating(5, 0.1, 40, "pvc"),                  True)
    check("Joule heating KILL (50A, 1Ω, PVC)",
          check_joule_heating(50, 1.0, 40, "pvc"),                 False)
    check("Voltage drop OK (10A, 5m, 2.5mm², 24V, 25°C)",
          check_voltage_drop(10, 5, 2.5, 24, temp_c=25),           True)
    check("Voltage drop KILL (thin wire, long run)",
          check_voltage_drop(50, 50, 0.5, 12),                     False)
    check("Contact resistance OK (5mΩ)",
          check_contact_resistance(5, 10),                         True)
    check("Contact resistance KILL (60mΩ)",
          check_contact_resistance(60, 10),                        False)
    check("Motor derating OK (40°C ambient)",
          check_motor_thermal_derating(40, 1000),                  True)
    check("Motor derating FAIL (80°C, >25% loss)",
          check_motor_thermal_derating(80, 1000),                  False)
    check("Bearing life OK (C/P=5, 1000RPM, target 2000h)",
          check_bearing_fatigue_life(50, 10, 1000, 2000),          True)
    check("Bearing life KILL (overloaded)",
          check_bearing_fatigue_life(10, 20, 3000, 20000),         False)
    check("Back-EMF OK (Kv=100, 2000RPM, 24V)",
          check_back_emf_limit(100, 2000, 24),                     True)
    check("Back-EMF KILL (motor can't reach RPM)",
          check_back_emf_limit(100, 5000, 24),                     False)


    # ── NumPy/SciPy Advanced Physics ─────────────────────────────────────────
    print("\n[ NumPy/SciPy — Advanced Physics ]")

    # Thermal network
    from physics.thermal import solve_thermal_network
    power4   = [100.0, 150.0, 120.0, 80.0]
    r_matrix = [
        [0.20,  2.0,  5.0, 10.0],
        [ 2.0, 0.15,  2.0,  5.0],
        [ 5.0,  2.0, 0.18,  2.0],
        [10.0,  5.0,  2.0, 0.25],
    ]
    ok_net, _, temps, hot = solve_thermal_network(power4, r_matrix, t_ambient_c=40.0)
    check("Thermal network OK (4 chiplets, coupled)",
          (ok_net, f"Hottest chip {hot}: {temps[hot]:.1f}°C"), True)

    # Thermal network FAIL — extreme power
    power_extreme = [500.0, 600.0, 550.0, 480.0]
    fail_net, detail_net, _, _ = solve_thermal_network(power_extreme, r_matrix, t_ambient_c=40.0)
    check("Thermal network FAIL (extreme power)",
          (fail_net, detail_net), False)

    # Rainflow fatigue
    from physics.mechanical import rainflow_fatigue_damage
    import math
    # Safe: low amplitude cycling
    t_safe = [i * 0.1 for i in range(200)]
    sig_safe = [60 * math.sin(2 * math.pi * 0.5 * t) for t in t_safe]
    check("Rainflow OK (low amplitude, σ_a=60MPa, UTS=500MPa)",
          rainflow_fatigue_damage(sig_safe, uts_mpa=500, material="steel"), True)

    # Danger: high amplitude
    sig_high = [280 * math.sin(2 * math.pi * 0.5 * t) +
                120 * math.sin(2 * math.pi * 3 * t) for t in t_safe]
    # design_life_cycles=50: with 65 counted cycles at high amplitude → D=4.37 >> 1
    check("Rainflow FAIL (high amplitude, D>=1, design_life=50 cycles)",
          rainflow_fatigue_damage(sig_high, uts_mpa=400, material="aluminum",
                                   design_life_cycles=50), False)

    # Weibull bearing reliability
    from physics.mechanical import weibull_bearing_reliability
    check("Weibull OK (C/P=5, 2000h, 90% reliability)",
          weibull_bearing_reliability(50, 10, 1000, target_hours=2000, reliability_pct=90), True)
    check("Weibull FAIL (overloaded bearing, 99% reliability required)",
          weibull_bearing_reliability(10, 20, 3000, target_hours=20000, reliability_pct=99), False)

    # Cold plate temperature distribution
    from physics.fluid_dynamics import solve_coldplate_temperature
    check("Cold plate OK (50W/cm², 5L/min, 0.3m plate)",
          solve_coldplate_temperature(50.0, 0.3, 0.1, 10.0,
                                       outlet_temp_limit_c=45.0), True)
    check("Cold plate FAIL (high flux, low flow)",
          solve_coldplate_temperature(80.0, 0.5, 0.15, 1.0,
                                       outlet_temp_limit_c=45.0), False)

    # ── Summary ──────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n══════════════════════════════════════")
    print(f"  Results: {passed}/{total} passed")
    if failed > 0:
        print(f"  ⚠️  {failed} tests FAILED")
        sys.exit(1)
    else:
        print(f"  ✓  All tests passed")
    print("══════════════════════════════════════\n")


if __name__ == "__main__":
    run_all_tests()
