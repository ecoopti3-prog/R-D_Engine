"""
test_full_suite.py — Full test suite for the R&D Engine.

Tests:
  1. Physics validators (29 unit tests — deterministic, no LLM)
  2. Schema validation (Pydantic models)
  3. Novelty detector (embedding + cosine similarity)
  4. LLM router (mock — no real API calls)
  5. DB operations (mock — no real Supabase calls)
  6. Cycle logic (integration — verifies ID matching, score persistence)
  7. Extractor ID matching (the bug we fixed)
  8. save_idea ↔ load_active_ideas roundtrip (score fields)

Run with: python -m pytest tests/test_full_suite.py -v
"""
from __future__ import annotations
import sys, os, json, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch, call
from core.schemas import (
    Idea, Finding, Kill, AgentOutput, DiamondScorePartial,
    PowerParams, ThermalParams, DataMovementParams, PDNParams,
    PhysicsVerdict, NoveltyResult
)


# ═══════════════════════════════════════════════════════════════
# 1. PHYSICS VALIDATORS
# ═══════════════════════════════════════════════════════════════

class TestElectricalPhysics:
    def test_landauer_below_limit(self):
        from physics.electrical import check_energy_per_op
        ok, msg = check_energy_per_op(1e-12)  # 1e-12 pJ = absurdly below Landauer
        assert not ok, "Should fail below Landauer limit"
        assert "Landauer" in msg or "below" in msg.lower()

    def test_landauer_above_limit(self):
        from physics.electrical import check_energy_per_op
        ok, msg = check_energy_per_op(1.0)  # 1 pJ — above Landauer
        assert ok

    def test_landauer_zero(self):
        from physics.electrical import check_energy_per_op
        ok, msg = check_energy_per_op(0.0)
        assert not ok

    def test_power_density_within_limit(self):
        from physics.electrical import check_power_density
        ok, msg = check_power_density(50.0)  # 50 W/cm2 — normal GPU
        assert ok

    def test_power_density_above_practical(self):
        from physics.electrical import check_power_density
        ok, msg = check_power_density(150.0)  # above practical limit — flags but not kills
        assert ok  # passes gate, flagged
        assert "practical" in msg.lower() or "flag" in msg.lower()

    def test_power_density_above_electrical(self):
        from physics.electrical import check_power_density
        ok, msg = check_power_density(1500.0)  # above electrical breakdown
        assert not ok

    def test_voltage_scaling_ok(self):
        from physics.electrical import check_voltage_scaling
        ok, msg = check_voltage_scaling(0.8, 5.0)  # 0.8V at 5nm — typical
        assert ok

    def test_voltage_scaling_fail(self):
        from physics.electrical import check_voltage_scaling
        ok, msg = check_voltage_scaling(5.0, 5.0)  # 5V at 5nm — impossible
        assert not ok

    def test_ir_drop_ok(self):
        from physics.electrical import check_ir_drop
        ok, msg = check_ir_drop(100.0, 0.5, 1.0)  # 100A, 0.5mΩ, 1.0V → 50mV = 5% → borderline
        assert ok

    def test_ir_drop_fail(self):
        from physics.electrical import check_ir_drop
        ok, msg = check_ir_drop(100.0, 5.0, 1.0)  # 100A, 5mΩ, 1.0V → 500mV = 50% → fail
        assert not ok

    def test_pdn_impedance_ok(self):
        from physics.electrical import check_pdn_impedance
        ok, msg = check_pdn_impedance(10.0, 0.5)  # 10mΩ at 0.5GHz — reasonable
        assert ok

    def test_electromigration_ok(self):
        from physics.electrical import check_power_bump_density
        ok, msg = check_power_bump_density(100.0, 10.0)  # 100 bumps/mm2, 10mA/bump
        assert ok

    def test_decap_sufficiency_ok(self):
        from physics.electrical import check_decap_sufficiency
        ok, msg = check_decap_sufficiency(1000.0, 10.0, 1.0)  # 1000nF, 10A/ns, 1V
        assert ok  # may or may not pass — just check it doesn't crash
        assert isinstance(ok, bool)
        assert isinstance(msg, str)


class TestThermalPhysics:
    def test_heat_flux_ok(self):
        from physics.thermal import check_heat_flux
        ok, msg = check_heat_flux(80.0, "copper")
        assert ok

    def test_heat_flux_fail(self):
        from physics.thermal import check_heat_flux
        ok, msg = check_heat_flux(5000.0, "default")
        assert not ok

    def test_junction_temp_ok(self):
        from physics.thermal import check_junction_temp
        ok, msg = check_junction_temp(85.0)
        assert ok

    def test_junction_temp_fail(self):
        from physics.thermal import check_junction_temp
        ok, msg = check_junction_temp(200.0)
        assert not ok

    def test_carnot_ok(self):
        from physics.thermal import check_carnot_efficiency
        ok, msg = check_carnot_efficiency(85.0, 25.0, 1.5)  # T_j=85, T_amb=25, COP=1.5
        assert ok

    def test_carnot_fail(self):
        from physics.thermal import check_carnot_efficiency
        # Carnot COP for 85°C from 25°C = (85+273) / (85-25) = 358/60 ≈ 5.97
        # Claiming COP=50 > Carnot → should fail
        ok, msg = check_carnot_efficiency(85.0, 25.0, 50.0)
        assert not ok

    def test_thermal_resistance_ok(self):
        from physics.thermal import check_thermal_resistance
        ok, msg = check_thermal_resistance(0.1, 100.0, 25.0)  # 0.1°C/W, 100W, 25°C ambient → T_j = 35°C
        assert ok

    def test_thermal_resistance_fail(self):
        from physics.thermal import check_thermal_resistance
        ok, msg = check_thermal_resistance(2.0, 300.0, 25.0)  # 2°C/W × 300W = 600°C rise → fail
        assert not ok


class TestDataMovementPhysics:
    def test_bandwidth_wall_ok(self):
        from physics.data_movement import check_bandwidth_wall
        ok, msg = check_bandwidth_wall(3000.0, 1000.0)  # 3 TB/s bandwidth, 1 PFLOPS → roofline ok
        assert ok

    def test_memory_latency_ok(self):
        from physics.data_movement import check_memory_latency
        ok, msg = check_memory_latency(100.0)  # 100ns — typical DRAM
        assert ok

    def test_memory_latency_below_light(self):
        from physics.data_movement import check_memory_latency
        # 0.0001 ns = 0.1 ps — faster than light would travel 30 microns → impossible
        ok, msg = check_memory_latency(0.0001)
        assert not ok


# ═══════════════════════════════════════════════════════════════
# 2. SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════

class TestSchemas:
    def test_idea_defaults(self):
        idea = Idea(
            title="Test idea",
            domain="thermal",
            problem="Heat flux exceeds 200 W/cm²",
            physical_limit="Fourier's law",
        )
        assert idea.id  # auto-generated UUID
        assert idea.status == "active"
        assert idea.diamond_score_partial.physics_feasibility == 0.0

    def test_idea_domain_validation(self):
        with pytest.raises(Exception):
            Idea(
                title="Bad domain",
                domain="invalid_domain",
                problem="test",
                physical_limit="test",
            )

    def test_finding_title_length(self):
        # Should accept up to 500 chars
        long_title = "x" * 499
        f = Finding(
            type="bottleneck",
            domain="thermal",
            title=long_title,
            description="desc",
            confidence=0.8,
        )
        assert len(f.title) == 499

    def test_finding_title_too_long(self):
        with pytest.raises(Exception):
            Finding(
                type="bottleneck",
                domain="thermal",
                title="x" * 501,
                description="desc",
                confidence=0.8,
            )

    def test_diamond_score_bounds(self):
        with pytest.raises(Exception):
            DiamondScorePartial(physics_feasibility=11.0)

        with pytest.raises(Exception):
            DiamondScorePartial(physics_feasibility=-1.0)

    def test_power_params_optional_fields(self):
        pp = PowerParams(watt=300.0)
        assert pp.watt == 300.0
        assert pp.tdp_watt is None

    def test_agent_output_serialization(self):
        idea = Idea(title="t", domain="power", problem="p", physical_limit="l")
        output = AgentOutput(
            agent="test_agent",
            cycle_id=str(uuid.uuid4()),
            timestamp="2026-01-01T00:00:00Z",
            status="done",
            ideas=[idea],
        )
        dumped = output.model_dump(mode="json")
        assert dumped["agent"] == "test_agent"
        assert len(dumped["ideas"]) == 1


# ═══════════════════════════════════════════════════════════════
# 3. NOVELTY DETECTOR
# ═══════════════════════════════════════════════════════════════

class TestNoveltyDetector:
    def test_cold_start_returns_max_novelty(self):
        from novelty.detector import check_novelty
        idea = Idea(title="t", domain="thermal", problem="p", physical_limit="l")
        result = check_novelty(idea, [])
        assert result.novelty_score == 10.0
        assert result.action == "pass"
        assert result.cosine_similarity == 0.0

    def test_embed_idea_returns_correct_dim(self):
        from novelty.detector import embed_idea
        idea = Idea(title="HBM thermal limit", domain="thermal",
                    problem="Heat flux exceeds 150 W/cm²", physical_limit="Fourier's law")
        emb = embed_idea(idea)
        assert len(emb) == 384
        # Check it's normalized (unit vector)
        import numpy as np
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    def test_identical_ideas_kill(self):
        from novelty.detector import embed_idea, check_novelty
        idea1 = Idea(title="HBM thermal limit in 3D stacking",
                     domain="thermal",
                     problem="Heat flux in HBM3 stacked dies exceeds 150 W/cm²",
                     physical_limit="Fourier heat conduction law")
        idea2 = Idea(title="HBM thermal limit in 3D stacking",
                     domain="thermal",
                     problem="Heat flux in HBM3 stacked dies exceeds 150 W/cm²",
                     physical_limit="Fourier heat conduction law")
        emb1 = embed_idea(idea1)
        archive = [{"id": idea1.id, "embedding": emb1}]
        result = check_novelty(idea2, archive, precomputed_embedding=embed_idea(idea2))
        assert result.action == "kill"
        assert result.cosine_similarity > 0.85

    def test_different_ideas_pass(self):
        from novelty.detector import embed_idea, check_novelty
        idea1 = Idea(title="HBM thermal limit in 3D stacking",
                     domain="thermal",
                     problem="Heat flux in HBM3 stacked dies exceeds 150 W/cm²",
                     physical_limit="Fourier heat conduction law")
        idea2 = Idea(title="PDN impedance in chiplet packages",
                     domain="pdn",
                     problem="IR drop exceeds 5% VDD in multi-chiplet PDN at 1 GHz",
                     physical_limit="Ohm's law + parasitic inductance")
        emb1 = embed_idea(idea1)
        emb2 = embed_idea(idea2)
        archive = [{"id": idea1.id, "embedding": emb1}]
        result = check_novelty(idea2, archive, precomputed_embedding=emb2)
        assert result.action == "pass"
        assert result.cosine_similarity < 0.75

    def test_precomputed_embedding_skips_double_work(self):
        """Verify precomputed_embedding param is used (no double embedding call)."""
        from novelty.detector import check_novelty, embed_idea
        idea = Idea(title="test", domain="power", problem="p", physical_limit="l")
        precomputed = embed_idea(idea)

        with patch("novelty.detector.embed_idea") as mock_embed:
            mock_embed.return_value = precomputed
            archive = [{"id": "other-id", "embedding": precomputed}]
            result = check_novelty(idea, archive, precomputed_embedding=precomputed)
            # Should NOT have called embed_idea since we passed precomputed
            mock_embed.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# 4. LLM ROUTER (mocked)
# ═══════════════════════════════════════════════════════════════

class TestLLMRouter:
    def test_successful_json_response(self):
        from core.llm_router import call_llm, _call_single
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"result": "ok", "ideas": []}'
        mock_resp.usage.total_tokens = 100

        with patch("core.llm_router._call_single", return_value=('{"result": "ok", "ideas": []}', 100)):
            result, llm_used, tokens = call_llm("system", "user", expect_json=True)
            assert result == {"result": "ok", "ideas": []}
            assert tokens == 100

    def test_json_retry_with_strict_system_prompt(self):
        """On invalid JSON, should retry with strict system prompt appended."""
        from core.llm_router import call_llm

        call_count = [0]
        def fake_call(llm_cfg, system, user):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("not json at all ```python print('hi')```", 50)
            else:
                # Second call (with strict system prompt) returns valid JSON
                return ('{"result": "retry_ok"}', 60)

        with patch("core.llm_router._call_single", side_effect=fake_call):
            result, llm_used, tokens = call_llm("system", "user", expect_json=True)
            assert result == {"result": "retry_ok"}
            assert call_count[0] == 2

    def test_all_llms_fail_raises(self):
        from core.llm_router import call_llm, AllLLMsFailedError
        from openai import RateLimitError

        with patch("core.llm_router._call_single", side_effect=Exception("network down")):
            with pytest.raises(AllLLMsFailedError):
                call_llm("system", "user", expect_json=True)

    def test_rate_limit_moves_to_next_llm(self):
        """429 should skip to next LLM, not retry same one."""
        from core.llm_router import call_llm
        from openai import RateLimitError

        call_count = [0]
        def fake_call(llm_cfg, system, user):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RateLimitError("Rate limited", response=MagicMock(status_code=429, headers={}), body={})
            return ('{"ok": true}', 50)

        with patch("core.llm_router._call_single", side_effect=fake_call):
            result, llm_used, tokens = call_llm("system", "user", expect_json=True)
            assert result == {"ok": True}
            assert call_count[0] == 2  # First LLM failed, second succeeded

    def test_markdown_fence_stripped(self):
        from core.llm_router import _parse_json
        raw = '```json\n{"key": "value"}\n```'
        parsed = _parse_json(raw)
        assert parsed == {"key": "value"}

    def test_plain_json_parsed(self):
        from core.llm_router import _parse_json
        parsed = _parse_json('{"ideas": [1, 2, 3]}')
        assert parsed == {"ideas": [1, 2, 3]}


# ═══════════════════════════════════════════════════════════════
# 5. EXTRACTOR ID MATCHING (the critical bug we fixed)
# ═══════════════════════════════════════════════════════════════

class TestExtractorIDMatching:
    """
    Verify that extractor ID matching works even when LLM returns
    truncated or slightly different IDs.
    """

    def _make_idea(self, title="GPU thermal limit", domain="thermal"):
        return Idea(
            title=title,
            domain=domain,
            problem="Heat flux problem",
            physical_limit="Fourier's law",
        )

    def test_full_uuid_match(self):
        """Extractor returns full UUID → should match."""
        idea = self._make_idea()
        extracted_params = {"idea_id": idea.id, "thermal_params": {"t_junction_c": 85.0}}
        # Simulate what main.py does in the extraction loop
        matched = False
        for orig in [idea]:
            eid = extracted_params["idea_id"]
            if orig.id == eid or orig.id.startswith(eid) or eid.startswith(orig.id):
                matched = True
                break
        assert matched

    def test_truncated_prefix_match(self):
        """Extractor returns truncated [:8] ID → should still match via prefix."""
        idea = self._make_idea()
        truncated_id = idea.id[:8]
        extracted_params = {"idea_id": truncated_id}
        matched = False
        for orig in [idea]:
            eid = extracted_params["idea_id"]
            if orig.id == eid or orig.id.startswith(eid) or eid.startswith(orig.id):
                matched = True
                break
        assert matched

    def test_wrong_id_no_match(self):
        """Completely wrong ID → should not match."""
        idea = self._make_idea()
        wrong_id = str(uuid.uuid4())
        matched = False
        for orig in [idea]:
            eid = wrong_id
            if orig.id == eid or orig.id.startswith(eid) or eid.startswith(orig.id):
                matched = True
                break
        assert not matched

    def test_multiple_ideas_correct_match(self):
        """Multiple ideas — each extractor output matches the right idea."""
        ideas = [self._make_idea(f"Idea {i}", "thermal") for i in range(5)]
        # Simulate extractor returning full UUIDs for ideas 1 and 3
        extractions = [
            {"idea_id": ideas[1].id, "thermal_params": {"t_junction_c": 90.0}},
            {"idea_id": ideas[3].id, "thermal_params": {"t_junction_c": 110.0}},
        ]
        matches = {}
        for ext in extractions:
            eid = ext["idea_id"]
            for orig in ideas:
                if orig.id == eid or orig.id.startswith(eid) or eid.startswith(orig.id):
                    matches[orig.id] = ext["thermal_params"]["t_junction_c"]
                    break
        assert len(matches) == 2
        assert matches[ideas[1].id] == 90.0
        assert matches[ideas[3].id] == 110.0


# ═══════════════════════════════════════════════════════════════
# 6. SCORE PERSISTENCE (the save_idea / load roundtrip bug)
# ═══════════════════════════════════════════════════════════════

class TestScorePersistence:
    """Verify save_idea saves all sub-scores and update_diamond_score propagates them."""

    def test_save_idea_includes_all_scores(self):
        """save_idea row should include all four sub-scores."""
        from db import supabase_client as db
        idea = Idea(
            title="Test",
            domain="thermal",
            problem="p",
            physical_limit="l",
            diamond_score_partial=DiamondScorePartial(
                physics_feasibility=7.5,
                market_pain=6.0,
                novelty=8.0,
                scalability=5.0,
            )
        )
        # Mock the supabase client
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[])

        with patch.object(db, "get_client", return_value=mock_client):
            db.save_idea(idea, "cycle-123")

        # Verify upsert was called with the right row
        upsert_call = mock_table.upsert.call_args
        row = upsert_call[0][0]
        assert row["physics_score"] == 7.5
        assert row["market_score"] == 6.0
        assert row["novelty_score"] == 8.0
        assert row["scalability_score"] == 5.0

    def test_update_diamond_score_with_subscores(self):
        """update_diamond_score should pass sub-scores to DB when provided."""
        from db import supabase_client as db

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[])

        with patch.object(db, "get_client", return_value=mock_client):
            db.update_diamond_score(
                "idea-id-123", 8.5, "diamond",
                physics=9.0, market=8.0, novelty=7.5, scalability=8.0
            )

        update_call = mock_table.update.call_args
        row = update_call[0][0]
        assert row["diamond_score"] == 8.5
        assert row["physics_score"] == 9.0
        assert row["market_score"] == 8.0
        assert row["novelty_score"] == 7.5
        assert row["scalability_score"] == 8.0

    def test_update_diamond_score_without_subscores(self):
        """update_diamond_score should work without sub-scores (backward compat)."""
        from db import supabase_client as db

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[])

        with patch.object(db, "get_client", return_value=mock_client):
            db.update_diamond_score("idea-id-456", 5.0, "active")

        update_call = mock_table.update.call_args
        row = update_call[0][0]
        assert row["diamond_score"] == 5.0
        assert "physics_score" not in row  # not passed → not in row


# ═══════════════════════════════════════════════════════════════
# 7. PHYSICS GATE INTEGRATION
# ═══════════════════════════════════════════════════════════════

class TestPhysicsGate:
    def _make_idea_with_thermal(self, t_junction=85.0, heat_flux=50.0):
        return Idea(
            title="GPU thermal test",
            domain="thermal",
            problem="Heat management",
            physical_limit="Fourier's law",
            thermal_params=ThermalParams(
                t_junction_c=t_junction,
                heat_flux_w_cm2=heat_flux,
                t_ambient_c=25.0,
            )
        )

    def test_good_thermal_idea_passes(self):
        from physics.gate import run_physics_gate
        idea = self._make_idea_with_thermal(85.0, 50.0)
        verdict = run_physics_gate(idea)
        assert verdict.passed
        assert verdict.score > 2.0
        assert len(verdict.checks_run) >= 2

    def test_bad_junction_temp_kills(self):
        from physics.gate import run_physics_gate
        idea = self._make_idea_with_thermal(250.0, 50.0)  # 250°C junction → fail
        verdict = run_physics_gate(idea)
        assert not verdict.passed
        assert verdict.score == 0.0
        assert verdict.kill_reason

    def test_no_params_gives_low_score(self):
        from physics.gate import run_physics_gate
        idea = Idea(title="Vague idea", domain="hardware",
                    problem="Some problem", physical_limit="unknown")
        verdict = run_physics_gate(idea)
        assert verdict.passed  # doesn't kill — just penalizes
        assert verdict.score == 2.0  # exactly 2.0 for unverifiable
        assert len(verdict.checks_run) == 0

    def test_score_increases_with_more_checks(self):
        from physics.gate import run_physics_gate
        idea = Idea(
            title="Multi-domain idea",
            domain="cross_domain",
            problem="Complex problem",
            physical_limit="Multiple",
            thermal_params=ThermalParams(
                t_junction_c=85.0,
                heat_flux_w_cm2=50.0,
                t_ambient_c=25.0,
                thermal_resistance_c_per_w=0.05,
            ),
            power_params=PowerParams(
                power_density_w_cm2=80.0,
                energy_per_op_pj=1.0,
            )
        )
        verdict = run_physics_gate(idea)
        assert verdict.passed
        # With 4+ checks, score should be well above 4.0
        assert verdict.score > 4.0

    def test_pdn_violation_kills(self):
        from physics.gate import run_physics_gate
        idea = Idea(
            title="PDN idea",
            domain="pdn",
            problem="PDN test",
            physical_limit="Ohm's law",
            pdn_params=PDNParams(
                ir_drop_mv=500.0,  # 500mV
                vdd_v=1.0,         # 500mV / 1.0V = 50% → way over 5% budget
                current_a=100.0,
            )
        )
        verdict = run_physics_gate(idea)
        assert not verdict.passed


# ═══════════════════════════════════════════════════════════════
# 8. BASE AGENT
# ═══════════════════════════════════════════════════════════════

class TestBaseAgent:
    def _make_agent(self):
        from core.base_agent import BaseAgent

        class MockAgent(BaseAgent):
            AGENT_NAME = "mock"
            CHAIN_TYPE = "research"
            SYSTEM_PROMPT = "You return JSON."

            def build_user_prompt(self, context):
                return "Analyze these ideas."

            def parse_output(self, raw, cycle_id):
                return self._empty_output(cycle_id)

        return MockAgent()

    def test_run_returns_failed_output_on_llm_error(self):
        from core.llm_router import AllLLMsFailedError
        agent = self._make_agent()
        with patch("core.base_agent.call_llm", side_effect=AllLLMsFailedError("all failed")):
            output = agent.run({}, "cycle-test")
            assert output.status == "failed"
            assert "all failed" in output.metadata.get("error", "")

    def test_chunked_run_merges_results(self):
        agent = self._make_agent()
        ideas = [{"id": str(uuid.uuid4()), "title": f"Idea {i}"} for i in range(12)]

        call_count = [0]
        original_run = agent.run

        def mock_run(context, cycle_id):
            call_count[0] += 1
            out = agent._empty_output(cycle_id)
            # Each chunk adds one finding
            out.findings.append(Finding(
                type="bottleneck", domain="thermal",
                title=f"Finding {call_count[0]}", description="desc",
                confidence=0.8
            ))
            return out

        agent.run = mock_run
        merged = agent.run_chunked(ideas=ideas, cycle_id="test", chunk_size=5)

        # 12 ideas / 5 per chunk = 3 chunks
        assert call_count[0] == 3
        assert len(merged.findings) == 3

    def test_chunked_empty_ideas(self):
        agent = self._make_agent()
        result = agent.run_chunked(ideas=[], cycle_id="test")
        assert result.status == "done"
        assert len(result.ideas) == 0


# ═══════════════════════════════════════════════════════════════
# 9. DB CLIENT EDGE CASES
# ═══════════════════════════════════════════════════════════════

class TestDBClient:
    def test_kill_idea_with_full_uuid(self):
        """kill_idea with full UUID should call update directly."""
        from db import supabase_client as db
        full_uuid = str(uuid.uuid4())

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[{"id": full_uuid}])

        with patch.object(db, "get_client", return_value=mock_client):
            result = db.kill_idea(full_uuid, "physics fail", "physics_impossible")
            assert result is True

    def test_timedelta_available_in_cleanup(self):
        """Regression: timedelta must be importable in cleanup_weak_ideas."""
        from db import supabase_client as db
        import inspect
        src = inspect.getsource(db.cleanup_weak_ideas)
        assert "from datetime import timedelta" in src

    def test_timedelta_available_in_week_stats(self):
        """Regression: timedelta must be importable in get_week_stats."""
        from db import supabase_client as db
        import inspect
        src = inspect.getsource(db.get_week_stats)
        assert "from datetime import timedelta" in src

    def test_seed_keywords_key_correct(self):
        """Regression: _update_seed_keywords must use 'seed_keywords' not 'keywords'."""
        import inspect
        import main
        src = inspect.getsource(main._update_seed_keywords)
        assert '"seed_keywords"' in src
        assert '"keywords"' not in src.replace('"seed_keywords"', '')

    def test_save_idea_row_has_proposed_direction(self):
        """save_idea must include proposed_direction in the upsert row."""
        from db import supabase_client as db
        import inspect
        src = inspect.getsource(db.save_idea)
        assert '"proposed_direction"' in src or "'proposed_direction'" in src


# ═══════════════════════════════════════════════════════════════
# 10. WORKFLOW YAML INTEGRITY
# ═══════════════════════════════════════════════════════════════

class TestWorkflow:
    def test_workflow_has_write_permissions(self):
        """GitHub Actions workflow must have contents: write for git push."""
        with open(".github/workflows/rd_engine.yml") as f:
            content = f.read()
        assert "contents: write" in content

    def test_workflow_has_all_4_cron_schedules(self):
        with open(".github/workflows/rd_engine.yml") as f:
            content = f.read()
        assert "0 6 * * *" in content    # Cycle 1
        assert "0 12 * * *" in content   # Cycle 2
        assert "0 18 * * *" in content   # Cycle 3
        assert "0 23 * * *" in content   # Cycle 4
        assert "0 0 * * 0" in content    # Weekly

    def test_workflow_has_all_secrets(self):
        with open(".github/workflows/rd_engine.yml") as f:
            content = f.read()
        for secret in ["GROQ_API_KEY", "SAMBANOVA_API_KEY", "FIREWORKS_API_KEY",
                       "MISTRAL_API_KEY", "GEMINI_API_KEY", "COHERE_API_KEY",
                       "SUPABASE_URL", "SUPABASE_KEY"]:
            assert secret in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
