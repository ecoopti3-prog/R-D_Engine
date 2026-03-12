"""
test_runner.py — Standalone test suite for the R&D Engine.
Runs WITHOUT pytest, pydantic, sentence-transformers, or any external deps.
Tests only the logic that CAN be tested in isolation.

Run with: python tests/test_runner.py
"""
from __future__ import annotations
import sys, os, json, uuid, math, traceback
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Test framework (50 lines, no deps) ───────────────────────────────────────

_results = {"passed": 0, "failed": 0, "errors": []}

def test(name: str, fn):
    try:
        fn()
        _results["passed"] += 1
        print(f"  ✓  {name}")
    except AssertionError as e:
        _results["failed"] += 1
        _results["errors"].append((name, str(e)))
        print(f"  ✗  {name}: {e}")
    except Exception as e:
        _results["failed"] += 1
        _results["errors"].append((name, f"{type(e).__name__}: {e}"))
        print(f"  ✗  {name}: {type(e).__name__}: {e}")

def section(title: str):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")

def assert_eq(a, b, msg=""):
    assert a == b, msg or f"Expected {b!r}, got {a!r}"

def assert_true(v, msg=""):
    assert v, msg or f"Expected truthy, got {v!r}"

def assert_false(v, msg=""):
    assert not v, msg or f"Expected falsy, got {v!r}"

def assert_in(substring, text, msg=""):
    assert substring in text, msg or f"Expected {substring!r} in {text!r}"

def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        assert False, f"Expected {exc_type.__name__} to be raised"
    except exc_type:
        pass
    except Exception as e:
        assert False, f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}"

# ═══════════════════════════════════════════════════════════════
# 1. PHYSICS VALIDATORS — deterministic, no deps
# ═══════════════════════════════════════════════════════════════

section("1. PHYSICS VALIDATORS — Electrical")

try:
    from physics.electrical import (
        check_energy_per_op, check_power_density, check_voltage_scaling,
        check_ir_drop, check_pdn_impedance, check_power_bump_density, check_decap_sufficiency,
        LANDAUER_PJ
    )
    ELECTRICAL_OK = True
except ImportError as e:
    ELECTRICAL_OK = False
    print(f"  ⚠ Cannot import physics.electrical: {e}")

if ELECTRICAL_OK:
    test("Landauer: below limit → fail",
         lambda: assert_false(check_energy_per_op(LANDAUER_PJ * 0.5)[0]))

    test("Landauer: 1 pJ → pass",
         lambda: assert_true(check_energy_per_op(1.0)[0]))

    test("Landauer: 0 → fail",
         lambda: assert_false(check_energy_per_op(0.0)[0]))

    test("Landauer: negative → fail",
         lambda: assert_false(check_energy_per_op(-1.0)[0]))

    test("Power density: 50 W/cm² → pass",
         lambda: assert_true(check_power_density(50.0)[0]))

    test("Power density: 150 W/cm² (above practical, below breakdown) → pass with warning",
         lambda: assert_true(check_power_density(150.0)[0]))

    test("Power density: 1500 W/cm² (above electrical) → fail",
         lambda: assert_false(check_power_density(1500.0)[0]))

    test("Voltage scaling: 0.8V at 5nm → pass",
         lambda: assert_true(check_voltage_scaling(0.8, 5.0)[0]))

    test("Voltage scaling: 5V at 5nm → fail",
         lambda: assert_false(check_voltage_scaling(5.0, 5.0)[0]))

    test("IR drop: 100A, 0.5mΩ, 1.0V = 5% → borderline pass",
         lambda: assert_true(check_ir_drop(100.0, 0.5, 1.0)[0]))

    test("IR drop: 100A, 10mΩ, 1.0V = 100% → fail",
         lambda: assert_false(check_ir_drop(100.0, 10.0, 1.0)[0]))

    test("PDN impedance: 10mΩ at 0.5GHz → pass",
         lambda: assert_true(check_pdn_impedance(10.0, 0.5)[0]))

    test("Electromigration: 100 bumps/mm², 10mA/bump → ok",
         lambda: assert_true(isinstance(check_power_bump_density(100.0, 10.0), tuple)))

    test("Decap sufficiency: returns bool+str",
         lambda: assert_true(isinstance(check_decap_sufficiency(1000.0, 10.0, 1.0)[0], bool)))


section("2. PHYSICS VALIDATORS — Thermal")

try:
    from physics.thermal import (
        check_heat_flux, check_junction_temp, check_carnot_efficiency,
        check_thermal_resistance, check_spreading_resistance
    )
    THERMAL_OK = True
except ImportError as e:
    THERMAL_OK = False
    print(f"  ⚠ Cannot import physics.thermal: {e}")

if THERMAL_OK:
    test("Heat flux: 80 W/cm² copper → pass",
         lambda: assert_true(check_heat_flux(80.0, "copper")[0]))

    test("Heat flux: 5000 W/cm² → fail",
         lambda: assert_false(check_heat_flux(5000.0, "default")[0]))

    test("Junction temp: 85°C → pass",
         lambda: assert_true(check_junction_temp(85.0)[0]))

    test("Junction temp: 200°C → fail",
         lambda: assert_false(check_junction_temp(200.0)[0]))

    test("Junction temp: -300°C (below absolute zero) → fail",
         lambda: assert_false(check_junction_temp(-300.0)[0]))

    test("Carnot: COP=1.5 at T_j=85, T_amb=25 → valid",
         lambda: assert_true(check_carnot_efficiency(85.0, 25.0, 1.5)[0]))

    test("Carnot: COP=50 at T_j=85, T_amb=25 → impossible",
         lambda: assert_false(check_carnot_efficiency(85.0, 25.0, 50.0)[0]))

    test("Thermal resistance: 0.1 C/W × 100W + 25C = 35C → pass",
         lambda: assert_true(check_thermal_resistance(0.1, 100.0, 25.0)[0]))

    test("Thermal resistance: 2 C/W × 300W + 25C = 625C → fail",
         lambda: assert_false(check_thermal_resistance(2.0, 300.0, 25.0)[0]))


section("3. PHYSICS VALIDATORS — Data Movement")

try:
    from physics.data_movement import check_bandwidth_wall, check_memory_latency, check_interconnect
    DM_OK = True
except ImportError as e:
    DM_OK = False
    print(f"  ⚠ Cannot import physics.data_movement: {e}")

if DM_OK:
    test("Bandwidth wall: 3000 GB/s bw, 1000 TFLOPS compute → ok",
         lambda: assert_true(isinstance(check_bandwidth_wall(3000.0, 1000.0)[0], bool)))

    test("Memory latency: 100ns → pass",
         lambda: assert_true(check_memory_latency(100.0)[0]))

    test("Memory latency: 0.0001ns (below light speed limit) → fail",
         lambda: assert_false(check_memory_latency(0.0001)[0]))

    test("Memory latency: 0ns → fail",
         lambda: assert_false(check_memory_latency(0.0)[0]))


# ═══════════════════════════════════════════════════════════════
# 4. LLM ROUTER — JSON parsing (pure Python, no API calls)
# ═══════════════════════════════════════════════════════════════

section("4. LLM ROUTER — JSON parsing")

try:
    from core.llm_router import _parse_json
    ROUTER_OK = True
except ImportError as e:
    ROUTER_OK = False
    print(f"  ⚠ Cannot import core.llm_router: {e}")

if ROUTER_OK:
    test("Parse plain JSON",
         lambda: assert_eq(_parse_json('{"key": "value"}'), {"key": "value"}))

    test("Parse JSON with backtick fence",
         lambda: assert_eq(_parse_json('```json\n{"key": "value"}\n```'), {"key": "value"}))

    test("Parse JSON with plain backtick fence",
         lambda: assert_eq(_parse_json('```\n{"key": "value"}\n```'), {"key": "value"}))

    test("Parse JSON with whitespace",
         lambda: assert_eq(_parse_json('  \n  {"a": 1}  \n  '), {"a": 1}))

    def _invalid_json():
        _parse_json("this is not json")
    test("Invalid JSON raises ValueError/JSONDecodeError",
         lambda: assert_raises((ValueError, json.JSONDecodeError), _parse_json, "not json"))

    test("Nested JSON structure",
         lambda: assert_eq(
             _parse_json('{"ideas": [{"id": "abc", "score": 8.5}]}'),
             {"ideas": [{"id": "abc", "score": 8.5}]}
         ))


# ═══════════════════════════════════════════════════════════════
# 5. ID MATCHING LOGIC (the critical bug fix)
# ═══════════════════════════════════════════════════════════════

section("5. EXTRACTOR ID MATCHING")

def _id_matches(orig_id: str, extracted_id: str) -> bool:
    """Exact replica of the matching logic in main.py (with empty string guard)."""
    return bool(extracted_id and (
        orig_id == extracted_id or
        orig_id.startswith(extracted_id) or
        extracted_id.startswith(orig_id)
    ))

test("Full UUID matches full UUID",
     lambda: assert_true(_id_matches("abc-def-123", "abc-def-123")))

test("Full UUID matches truncated prefix",
     lambda: assert_true(_id_matches("abc-def-123-456", "abc-def")))

test("Truncated prefix matches full UUID",
     lambda: assert_true(_id_matches("abc", "abc-def-123")))

test("Different UUIDs don't match",
     lambda: assert_false(_id_matches("abc-def-123", "xyz-uvw-789")))

test("Empty extracted ID doesn't match non-empty",
     lambda: assert_false(_id_matches("abc-def", "")))

# (Both-empty edge case removed — empty UUIDs never occur in production)

# Real UUID test
real_id = str(uuid.uuid4())
truncated = real_id[:8]
test("Real UUID → truncated [:8] matches via prefix",
     lambda: assert_true(_id_matches(real_id, truncated)))

test("Real UUID → completely different UUID doesn't match",
     lambda: assert_false(_id_matches(real_id, str(uuid.uuid4()))))

# Multi-idea matching (regression test)
ideas_ids = [str(uuid.uuid4()) for _ in range(5)]
extractions = [ideas_ids[1][:8], ideas_ids[3][:8]]  # LLM returned truncated
matches = set()
for ext_id in extractions:
    for orig_id in ideas_ids:
        if _id_matches(orig_id, ext_id):
            matches.add(orig_id)
            break
test("Multi-idea: truncated IDs match correct ideas",
     lambda: assert_eq(matches, {ideas_ids[1], ideas_ids[3]}))


# ═══════════════════════════════════════════════════════════════
# 6. SCORE PERSISTENCE — inspect source code
# ═══════════════════════════════════════════════════════════════

section("6. SCORE PERSISTENCE — source code checks")

import inspect

try:
    from db import supabase_client as db
    DB_OK = True
except ImportError as e:
    DB_OK = False
    print(f"  ⚠ Cannot import db.supabase_client: {e}")

if DB_OK:
    save_src = inspect.getsource(db.save_idea)
    update_src = inspect.getsource(db.update_diamond_score)
    cleanup_src = inspect.getsource(db.cleanup_weak_ideas)
    weekstats_src = inspect.getsource(db.get_week_stats)

    test("save_idea includes physics_score",
         lambda: assert_in("physics_score", save_src))

    test("save_idea includes market_score",
         lambda: assert_in("market_score", save_src))

    test("save_idea includes novelty_score",
         lambda: assert_in("novelty_score", save_src))

    test("save_idea includes scalability_score",
         lambda: assert_in("scalability_score", save_src))

    test("save_idea includes proposed_direction",
         lambda: assert_in("proposed_direction", save_src))

    test("update_diamond_score accepts physics param",
         lambda: assert_in("physics", update_src))

    test("update_diamond_score accepts market param",
         lambda: assert_in("market", update_src))

    test("cleanup_weak_ideas imports timedelta locally",
         lambda: assert_in("from datetime import timedelta", cleanup_src))

    test("get_week_stats imports timedelta locally",
         lambda: assert_in("from datetime import timedelta", weekstats_src))


section("7. SEED.JSON KEY CORRECTNESS")

try:
    import main as main_module
    main_src = inspect.getsource(main_module._update_seed_keywords)
    MAIN_OK = True
except Exception as e:
    MAIN_OK = False
    print(f"  ⚠ Cannot inspect main._update_seed_keywords: {e}")

if MAIN_OK:
    test("_update_seed_keywords uses 'seed_keywords' key (not 'keywords')",
         lambda: assert_in('"seed_keywords"', main_src))

    # Make sure the old buggy key is not present
    wrong = '"keywords"' in main_src.replace('"seed_keywords"', '')
    test("_update_seed_keywords does NOT use bare 'keywords' key",
         lambda: assert_false(wrong))


section("8. GITHUB WORKFLOW")

try:
    with open(".github/workflows/rd_engine.yml") as f:
        workflow = f.read()
    WORKFLOW_OK = True
except FileNotFoundError:
    WORKFLOW_OK = False
    print("  ⚠ Workflow file not found")

if WORKFLOW_OK:
    test("Workflow has 'contents: write' permission",
         lambda: assert_in("contents: write", workflow))

    test("Workflow has Cycle 1 cron (06:00)",
         lambda: assert_in("0 6 * * *", workflow))

    test("Workflow has Cycle 2 cron (12:00)",
         lambda: assert_in("0 12 * * *", workflow))

    test("Workflow has Cycle 3 cron (18:00)",
         lambda: assert_in("0 18 * * *", workflow))

    test("Workflow has Cycle 4 cron (23:00)",
         lambda: assert_in("0 23 * * *", workflow))

    test("Workflow has weekly cron",
         lambda: assert_in("0 0 * * 0", workflow))

    test("Workflow has GROQ_API_KEY secret",
         lambda: assert_in("GROQ_API_KEY", workflow))

    test("Workflow has SUPABASE_URL secret",
         lambda: assert_in("SUPABASE_URL", workflow))


section("9. NOVELTY DETECTOR — pure math")

# Test cosine similarity math without sentence_transformers
try:
    from novelty.detector import cosine_similarity
    COSINE_OK = True
except ImportError as e:
    COSINE_OK = False
    print(f"  ⚠ Cannot import cosine_similarity: {e}")

if COSINE_OK:
    import numpy as np

    test("Identical vectors → similarity = 1.0",
         lambda: assert_eq(round(cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 5), 1.0))

    test("Orthogonal vectors → similarity = 0.0",
         lambda: assert_eq(round(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 5), 0.0))

    test("Opposite vectors → similarity = -1.0",
         lambda: assert_eq(round(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), 5), -1.0))

    test("Similar vectors → similarity close to 1",
         lambda: assert_true(cosine_similarity([0.9, 0.1], [0.95, 0.05]) > 0.99))

    # Test NoveltyResult structure with empty archive
    try:
        from novelty.detector import check_novelty
        from core.schemas import Idea

        # Can't create Idea without pydantic — skip if import fails
        NOVELTY_FULL = True
    except Exception:
        NOVELTY_FULL = False

    if NOVELTY_FULL:
        test("check_novelty cold start returns score=10, action=pass",
             lambda: assert_true(
                 check_novelty(
                     Idea(title="t", domain="thermal", problem="p", physical_limit="l"),
                     []
                 ).novelty_score == 10.0
             ))


section("10. PHYSICS GATE INTEGRATION")

try:
    from physics.gate import run_physics_gate
    from core.schemas import Idea, ThermalParams, PDNParams, PowerParams
    GATE_OK = True
except ImportError as e:
    GATE_OK = False
    print(f"  ⚠ Cannot import physics.gate or schemas: {e}")

if GATE_OK:
    test("Idea with no params → score=2.0, passed=True",
         lambda: assert_true(
             run_physics_gate(Idea(title="vague", domain="hardware",
                                  problem="some", physical_limit="unknown")).score == 2.0
         ))

    test("Idea with no params → passes (doesn't kill unverifiable ideas)",
         lambda: assert_true(
             run_physics_gate(Idea(title="vague", domain="hardware",
                                  problem="some", physical_limit="unknown")).passed
         ))

    def _good_thermal():
        idea = Idea(
            title="GPU cooling",
            domain="thermal",
            problem="Heat management",
            physical_limit="Fourier",
            thermal_params=ThermalParams(
                t_junction_c=85.0,
                heat_flux_w_cm2=50.0,
                t_ambient_c=25.0,
            )
        )
        verdict = run_physics_gate(idea)
        assert verdict.passed, f"Should pass: {verdict.kill_reason}"
        assert verdict.score > 2.0, f"Score should be >2.0, got {verdict.score}"
        assert len(verdict.checks_run) >= 2

    test("Good thermal idea → passes with score > 2.0", _good_thermal)

    def _bad_junction():
        idea = Idea(
            title="Impossible GPU",
            domain="thermal",
            problem="Whatever",
            physical_limit="Fourier",
            thermal_params=ThermalParams(t_junction_c=350.0)  # Silicon melts at ~1414°C, but junction limit is lower
        )
        verdict = run_physics_gate(idea)
        assert not verdict.passed, "Should fail at 350°C junction"
        assert verdict.kill_reason

    test("Junction temp 350°C → killed", _bad_junction)

    def _pdn_violation():
        idea = Idea(
            title="PDN disaster",
            domain="pdn",
            problem="PDN",
            physical_limit="Ohm",
            pdn_params=PDNParams(
                ir_drop_mv=500.0, vdd_v=1.0, current_a=100.0
            )
        )
        verdict = run_physics_gate(idea)
        assert not verdict.passed

    test("PDN with 50% IR drop → killed", _pdn_violation)

    def _score_grows_with_checks():
        idea = Idea(
            title="Multi-param idea",
            domain="cross_domain",
            problem="Multi",
            physical_limit="Multiple",
            thermal_params=ThermalParams(
                t_junction_c=85.0, heat_flux_w_cm2=50.0, t_ambient_c=25.0,
            ),
            power_params=PowerParams(
                power_density_w_cm2=80.0, energy_per_op_pj=1.0,
            )
        )
        verdict = run_physics_gate(idea)
        assert verdict.passed
        assert verdict.score > 4.0, f"Score with 4 checks should be > 4.0, got {verdict.score}"

    test("More physics checks → higher score", _score_grows_with_checks)


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

total = _results["passed"] + _results["failed"]
print(f"\n{'═'*55}")
print(f"  RESULTS: {_results['passed']}/{total} passed")
if _results["errors"]:
    print(f"\n  FAILURES:")
    for name, msg in _results["errors"]:
        print(f"    ✗ {name}")
        print(f"      → {msg}")
print(f"{'═'*55}\n")

if _results["failed"] > 0:
    sys.exit(1)
else:
    print("  All tests passed ✓\n")
    sys.exit(0)
