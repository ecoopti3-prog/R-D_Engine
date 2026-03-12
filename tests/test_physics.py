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
