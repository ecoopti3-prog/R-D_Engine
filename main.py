"""
main.py — Entry point for the autonomous R&D engine.
"""
from __future__ import annotations
import argparse, logging, sys, json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from config.settings import AGENT_DELAY_SECONDS, HEARTBEAT_INTERVAL_SEC
from core.schemas import Idea, AgentOutput
import db.supabase_client as db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rd_engine")

# FIX: map each agent to the source types it actually uses
# This prevents every idea from being linked to all 12+ source types
AGENT_SOURCE_TYPES = {
    "paper_researcher":        ["arxiv", "semantic_scholar", "huggingface_daily", "osti_doe", "openreview"],
    "patent_researcher":       ["patent"],
    "infra_researcher":        ["github_issue", "ocp_spec_discussion", "ieee_spectrum", "ee_times",
                                "semianalysis", "tomshardware", "semiwiki", "theregister_dc",
                                "arstechnica", "techpowerup", "anandtech", "sec_edgar", "darpa_baa"],
    "intelligence_researcher": ["job_posting", "isscc", "forum"],
    "cross_domain_synthesizer": None,  # None = use all sources (cross-domain by definition)
}


def load_seed():
    seed_path = Path("config/seed.json")
    try:
        with seed_path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("[FATAL] config/seed.json not found! Rename seed_backup.json → seed.json")
        sys.exit(1)


def run_cycle_1_harvest(cycle_id: str, seed: dict) -> list[Idea]:
    from agents.research.paper_researcher import PaperResearcher
    from agents.research.patent_researcher import PatentResearcher
    from agents.research.infra_researcher import InfraResearcher
    from agents.research.intelligence_researcher import IntelligenceResearcher
    from agents.research.robotics_researcher import RoboticsResearcher
    from agents.research.physics_limit_mapper import PhysicsLimitMapper
    from agents.synthesis.cross_domain_synthesizer import CrossDomainSynthesizer
    from utils.sources import fetch_all_for_cycle1
    from db.lineage import get_top_yielding_sources, save_idea_lineage

    logger.info("=" * 60)
    logger.info("CYCLE 1 — HARVEST (06:00 UTC)")
    logger.info("=" * 60)

    # ── Step 0: Load strategy + deep memory from previous cycles ────────────
    search_strategy  = db.load_search_strategy()
    cycle_memory     = db.load_cycle_memory(days_back=14)   # deep memory: killed/diamonds/keywords
    recent_findings  = db.load_recent_findings(days_back=7, limit=30)
    if not recent_findings:
        recent_findings = db.load_today_findings()[:15]
    domain_rates     = db.get_domain_success_rates(days_back=14)

    if domain_rates:
        top_domains  = [d for d, r in sorted(domain_rates.items(), key=lambda x: x[1] or 0, reverse=True) if r is not None][:3]
        weak_domains = [d for d, r in sorted(domain_rates.items(), key=lambda x: x[1] or 0) if r is not None and r < 0.15]
        search_strategy["top_domains"]  = top_domains
        search_strategy["weak_domains"] = weak_domains
        logger.info(f"[Strategy] Top domains: {top_domains} | Weak: {weak_domains}")

    # Merge deep memory into search_strategy so it flows through to agents
    # Deep memory wins for lists (more comprehensive than single-day strategy)
    if cycle_memory.get("killed_idea_titles"):
        search_strategy["killed_idea_titles"] = cycle_memory["killed_idea_titles"]
    if cycle_memory.get("evolved_keywords"):
        existing_kw = search_strategy.get("new_keywords", [])
        merged_kw   = list(dict.fromkeys(cycle_memory["evolved_keywords"] + existing_kw))[:20]
        search_strategy["new_keywords"] = merged_kw
    if cycle_memory.get("diamond_titles"):
        search_strategy["diamond_titles"] = cycle_memory["diamond_titles"]
    logger.info(
        f"[Memory] Loaded — "
        f"{len(search_strategy.get('killed_idea_titles', []))} killed titles, "
        f"{len(cycle_memory.get('diamond_titles', []))} diamonds, "
        f"{len(search_strategy.get('new_keywords', []))} evolved keywords"
    )

    # ── Load open hypotheses for injection into PhysicsLimitMapper ───────────
    open_hypotheses = db.load_open_hypotheses(limit=5)
    if open_hypotheses:
        logger.info(f"[Hypotheses] Injecting {len(open_hypotheses)} open hypotheses into Cycle 1")

    # ── Load top yielding sources for researcher context ──────────────────────
    try:
        top_sources = get_top_yielding_sources(db.get_client(), top_n=8, days_back=30)
        if top_sources:
            logger.info(f"[Lineage] Top yielding venues: {[s['venue'] for s in top_sources[:3]]}")
    except Exception as e:
        logger.warning(f"[Lineage] Could not load source yields (non-blocking): {e}")
        top_sources = []

    # ── Step 1: Physics Limit Mapper ─────────────────────────────────────────
    logger.info("[PhysicsLimitMapper] Mapping current physical bottlenecks...")
    mapper_measured_limits = []
    try:
        from db.sim_feedback_loop import load_measured_physics_limits
        mapper_measured_limits = load_measured_physics_limits(db.get_client())
        if mapper_measured_limits:
            logger.info(f"[PhysicsLimitMapper] Injecting {len(mapper_measured_limits)} measured limits from Sim Engine")
    except Exception:
        pass

    mapper_context = {
        "keywords":                seed.get("seed_keywords", []),
        "recent_findings":         recent_findings,
        "search_strategy":         search_strategy,
        "kill_patterns":           search_strategy.get("kill_patterns", []),
        "opportunity_filters":     seed.get("opportunity_filters", {}),
        "open_hypotheses":         open_hypotheses,
        "measured_physics_limits": mapper_measured_limits,
    }
    time.sleep(AGENT_DELAY_SECONDS)
    mapper_output = PhysicsLimitMapper().run(mapper_context, cycle_id)
    db.save_agent_output(mapper_output)

    physics_limits   = mapper_output.metadata.get("physics_limits", [])
    priority_domains = mapper_output.metadata.get("priority_domains", [])
    focus_summary    = mapper_output.metadata.get("focus_summary", "")
    eng_limits       = mapper_output.metadata.get("engineering_limits_worth_challenging", [])
    logger.info(f"[PhysicsLimitMapper] {len(physics_limits)} limits mapped — focus: {focus_summary}")

    # ── Step 2: Fetch all 12 sources ─────────────────────────────────────────
    logger.info("[Sources] Fetching from all 13 sources...")
    evolved_keywords = list(dict.fromkeys(
        search_strategy.get("new_keywords", []) + seed.get("seed_keywords", [])
    ))[:20]

    # Memory: log previously killed idea titles so we don't re-harvest the same ideas
    killed_titles = search_strategy.get("killed_idea_titles", [])
    if killed_titles:
        logger.info(f"[Memory] {len(killed_titles)} previously killed idea titles loaded — agents will avoid repeating")

    evolved_seed = {**seed, "seed_keywords": evolved_keywords}
    source_data = fetch_all_for_cycle1(evolved_seed)

    # ── Log source counts ─────────────────────────────────────────────────────
    logger.info(
        f"[Sources] Fetched — "
        f"papers: {len(source_data.get('papers', []))} | "
        f"patents: {len(source_data.get('patents', []))} | "
        f"jobs: {len(source_data.get('job_signals', []))} | "
        f"rss: {len(source_data.get('rss_signals', []))} | "
        f"github: {len(source_data.get('github_signals', []))} | "
        f"edgar: {len(source_data.get('edgar_signals', []))} | "
        f"darpa: {len(source_data.get('darpa_signals', []))} | "
        f"nasa: {len(source_data.get('nasa_signals', []))}"
    )

    all_ideas: list[Idea] = []
    all_findings = []

    limits_text = ""
    if physics_limits:
        for lim in physics_limits[:6]:
            mechs = ", ".join(lim.get("mechanisms_to_explore", [])[:2])
            ltype = lim.get("limit_type", "physics")
            limits_text += (
                f"\n• [{lim.get('domain','?')}] {lim.get('name','')} — "
                f"gap: {lim.get('gap_percent','?')}% — "
                f"type: {ltype} — mechanisms: {mechs}"
            )

    # ── FIX: Pass ALL source types to context so agents receive real data ─────
    context = {
        "keywords":             evolved_keywords,
        "focus_domain":         ", ".join(priority_domains) if priority_domains else "all domains",
        "target_companies":     seed.get("target_companies", []),
        "intelligence_signals": seed.get("intelligence_signals", []),
        "date":                 datetime.now().strftime("%Y-%m-%d"),
        # papers (arXiv + HuggingFace + OSTI + OpenReview + OpenAlex)
        "papers":               source_data.get("papers", []),
        # patents (Lens.org)
        "patents":              source_data.get("patents", []),
        # job signals (Greenhouse + Lever + Workday + HN)
        "job_signals":          source_data.get("job_signals", []),
        # FIX: these 4 were missing — infra_researcher needs them
        "rss_signals":          source_data.get("rss_signals", []),
        "github_signals":       source_data.get("github_signals", []),
        "edgar_signals":        source_data.get("edgar_signals", []),
        "darpa_signals":        source_data.get("darpa_signals", []),
        "nasa_signals":         source_data.get("nasa_signals", []),
        # shared context
        "findings":             [],
        "physics_limits":       limits_text,
        "focus_summary":        focus_summary,
        "kill_patterns":        search_strategy.get("kill_patterns", []),
        "killed_idea_titles":   search_strategy.get("killed_idea_titles", []),  # memory: avoid repeating
        "diamond_titles":       search_strategy.get("diamond_titles", []),       # memory: successful directions
        "weak_domains":         search_strategy.get("weak_domains", []),
        "opportunity_filters":  seed.get("opportunity_filters", {}),
        "engineering_limits":   eng_limits,
        "open_hypotheses":      open_hypotheses,
        "top_yielding_sources": top_sources,
    }

    for agent in [PaperResearcher(), PatentResearcher(), InfraResearcher(), IntelligenceResearcher(), RoboticsResearcher()]:
        time.sleep(AGENT_DELAY_SECONDS)
        db.update_heartbeat(cycle_id)
        output: AgentOutput = agent.run(context, cycle_id)
        db.save_agent_output(output)
        for idea in output.ideas:
            idea._source_agent = agent.AGENT_NAME
        all_ideas.extend(output.ideas)
        all_findings.extend(output.findings)
        context["findings"] = [
            f.model_dump(mode="json") if hasattr(f, "model_dump") else f
            for f in all_findings
        ]

    # ── CrossDomainSynthesizer ────────────────────────────────────────────────
    logger.info("[CrossDomainSynthesizer] Searching for cross-domain couplings...")
    time.sleep(AGENT_DELAY_SECONDS)
    db.update_heartbeat(cycle_id)
    synth_output = CrossDomainSynthesizer().run({
        "findings":       context["findings"],
        "ideas":          [i.model_dump(mode="json") for i in all_ideas],
        "physics_limits": limits_text,
        "focus_summary":  focus_summary,
    }, cycle_id)
    db.save_agent_output(synth_output)
    for idea in synth_output.ideas:
        idea._source_agent = "cross_domain_synthesizer"
    all_ideas.extend(synth_output.ideas)
    coupling_map = synth_output.metadata.get("coupling_map", [])
    logger.info(
        f"[CrossDomainSynthesizer] {len(synth_output.ideas)} cross-domain ideas, "
        f"{len(coupling_map)} couplings"
    )

    # ── Embed + save ──────────────────────────────────────────────────────────
    from novelty.detector import embed_idea as _embed_idea

    source_url_map: dict[str, list[str]] = {}
    for paper in source_data.get("papers", []):
        url = paper.get("url", "")
        src = paper.get("source") or "unknown"   # FIX: .get() with None value → "unknown"
        if url:
            source_url_map.setdefault(src, []).append(url)
    for patent in source_data.get("patents", []):
        url = patent.get("url", "")
        if url:
            source_url_map.setdefault("patent", []).append(url)
    for sig in source_data.get("rss_signals", []):
        url = sig.get("url", "")
        src = sig.get("source") or "rss"         # FIX
        if url:
            source_url_map.setdefault(src, []).append(url)
    for sig in source_data.get("github_signals", []):
        url = sig.get("url", "")
        src = sig.get("signal_type") or "github"  # FIX
        if url:
            source_url_map.setdefault(src, []).append(url)
    for sig in source_data.get("edgar_signals", []):
        url = sig.get("url", "")
        if url:
            source_url_map.setdefault("sec_edgar", []).append(url)
    for sig in source_data.get("darpa_signals", []):
        url = sig.get("url", "")
        if url:
            source_url_map.setdefault("darpa_baa", []).append(url)
    for sig in source_data.get("nasa_signals", []):
        url = sig.get("url", "")
        src = sig.get("source") or "nasa_ntrs"    # FIX
        if url:
            source_url_map.setdefault(src, []).append(url)
    for sig in source_data.get("job_signals", []):
        url = sig.get("url", "")
        src = sig.get("source_type") or "job_posting"  # FIX
        if url:
            source_url_map.setdefault(src, []).append(url)

    for idea in all_ideas:
        idea.cycle_id = cycle_id
        try:
            emb = _embed_idea(idea)
        except Exception:
            emb = None
        primary_source_url = None
        for urls in source_url_map.values():
            if urls:
                primary_source_url = urls[0]
                break
        db.save_idea(idea, cycle_id, emb, source_url=primary_source_url)

        try:
            client = db.get_client()
            agent_name = getattr(idea, "_source_agent", "researcher")
            
            # תיקון: מושך רק את הלינקים שהוצמדו לרעיון הספציפי הזה בתוך ה-Agent
            idea_source_urls = getattr(idea, "source_urls", [])
            
            def get_venue(url: str):
                if 'arxiv.org' in url: return 'arxiv'
                if 'patents.google' in url: return 'google_patents'
                if 'huggingface.co' in url: return 'huggingface'
                if 'github.com' in url: return 'github'
                return 'other'

            idea_source_types = [get_venue(u) for u in idea_source_urls]
            # שומר את המקור העיקרי בשדה החדש שהוספנו ל-DB
            idea.source_venue = idea_source_types[0] if idea_source_types else "unknown"

            from db.lineage import save_idea_lineage
            save_idea_lineage(
                idea_id=idea.id,
                source_urls=idea_source_urls,
                source_types=idea_source_types,
                finding_ids=[f.id for f in all_findings[:3] if hasattr(f, "id")],
                agent_name=agent_name,
                db_client=client,
            )
        except Exception as e:
            logger.debug(f"[Lineage] Save failed for {idea.id[:8]}: {e}")

    for finding in all_findings:
        db.save_finding(finding, cycle_id)

    if coupling_map or eng_limits:
        strategy = db.load_search_strategy()
        if coupling_map:
            strategy["coupling_map"] = coupling_map
        if eng_limits:
            strategy["engineering_limits_worth_challenging"] = eng_limits
        db.save_search_strategy(strategy)

    logger.info(
        f"[Cycle1] Complete — {len(all_ideas)} ideas "
        f"({len(synth_output.ideas)} cross-domain), {len(all_findings)} findings"
    )
    return all_ideas


def run_cycle_2_physics_market(cycle_id: str, ideas: list[Idea]) -> list[Idea]:
    from physics.gate import run_physics_gate
    from agents.extraction.pdn_extractor import PDNExtractor
    from agents.extraction.power_extractor import PowerExtractor
    from agents.extraction.thermal_extractor import ThermalExtractor
    from agents.extraction.data_movement_extractor import DataMovementExtractor
    from agents.physics_agents.electrical_engineer import ElectricalEngineer
    from agents.physics_agents.thermal_engineer import ThermalEngineer
    from agents.physics_agents.systems_architect import SystemsArchitect
    from agents.market.market_analyst import MarketAnalyst
    from agents.market.cost_analyst import CostAnalyst
    from novelty.detector import check_novelty, embed_idea
    from config.settings import PHYSICS_GATE_WEIGHT, PHYSICS_AGENT_WEIGHT

    logger.info("=" * 60)
    logger.info("CYCLE 2 — PHYSICS + MARKET (12:00 UTC)")
    logger.info("=" * 60)

    all_findings      = db.load_today_findings()
    ideas_cycle_id    = getattr(ideas[0], "cycle_id", None) if ideas else None

    # FIX: Novelty cold start — was excluding cycle 1 ideas (ideas_cycle_id) from
    # the archive, leaving archive empty on day 1 → every idea got novelty=10 (no comparison).
    # Cycle 2 processes ideas FROM cycle 1, so we should NOT exclude cycle 1 from archive.
    # Only exclude the current cycle 2 ID (ideas being scored right now) to avoid self-comparison.
    archive_embeddings = db.get_all_embeddings(exclude_cycle_id=cycle_id)

    ideas_as_dicts    = [i.model_dump(mode="json") for i in ideas]
    findings_as_dicts = [f if isinstance(f, dict) else f for f in all_findings[:30]]

    extraction_context = {"ideas": ideas_as_dicts, "findings": findings_as_dicts}
    ideas_by_id: dict[str, Idea] = {idea.id: idea for idea in ideas}

    for ExtractorClass in [PDNExtractor, PowerExtractor, ThermalExtractor, DataMovementExtractor]:
        time.sleep(AGENT_DELAY_SECONDS)
        db.update_heartbeat(cycle_id)
        output = ExtractorClass().run_chunked(
            ideas=ideas_as_dicts,
            cycle_id=cycle_id,
            extra_context={"findings": findings_as_dicts},
        )
        db.save_agent_output(output)
        matched = 0
        for extracted in output.ideas:
            eid  = extracted.id
            orig = ideas_by_id.get(eid)
            if orig is None and eid:
                candidates = [v for k, v in ideas_by_id.items()
                              if k.startswith(eid) or eid.startswith(k)]
                if len(candidates) == 1:
                    orig = candidates[0]
            if orig is not None:
                if extracted.pdn_params:           orig.pdn_params = extracted.pdn_params
                if extracted.power_params:         orig.power_params = extracted.power_params
                if extracted.thermal_params:       orig.thermal_params = extracted.thermal_params
                if extracted.data_movement_params: orig.data_movement_params = extracted.data_movement_params
                matched += 1
        logger.info(f"[{ExtractorClass.__name__}] Matched {matched}/{len(output.ideas)} extractions")

    surviving: list[Idea] = []
    for idea in ideas:
        db.update_heartbeat(cycle_id)
        verdict = run_physics_gate(idea)
        if not verdict.passed:
            idea.status = "killed"
            idea.kill_reason = verdict.kill_reason
            db.kill_idea(idea.id, verdict.kill_reason or "", "physics_impossible")
            logger.warning(f"[PhysicsGate] KILLED {idea.id[:8]}: {verdict.kill_reason}")
            continue
        idea.diamond_score_partial.physics_feasibility = verdict.score
        surviving.append(idea)

    logger.info(f"[PhysicsGate] {len(surviving)}/{len(ideas)} survived")

    novel_surviving = []
    for idea in surviving:
        try:
            embedding = embed_idea(idea)
            novelty   = check_novelty(idea, archive_embeddings, precomputed_embedding=embedding)
            idea.diamond_score_partial.novelty = novelty.novelty_score
            db.save_idea(idea, cycle_id, embedding)
            archive_embeddings.append({"id": idea.id, "embedding": embedding})
            if novelty.action == "kill":
                idea.status = "killed"
                db.kill_idea(idea.id, f"Duplicate of {novelty.similar_to}", "duplicate")
                logger.info(f"[Novelty] KILLED {idea.id[:8]} (sim={novelty.cosine_similarity:.2f})")
                continue
        except Exception as e:
            logger.error(f"[Novelty] Failed for {idea.id[:8]}: {e}")
            idea.status = "physics_unverified"
        novel_surviving.append(idea)
        time.sleep(AGENT_DELAY_SECONDS)

    physics_ideas_dicts = [i.model_dump(mode="json") for i in novel_surviving]
    physics_assessments = {}

    for PhysicsAgentClass in [ElectricalEngineer, ThermalEngineer, SystemsArchitect]:
        time.sleep(AGENT_DELAY_SECONDS)
        db.update_heartbeat(cycle_id)
        extra = {}
        if PhysicsAgentClass.__name__ == "SystemsArchitect":
            extra = {"physics_assessments": physics_assessments}
        output = PhysicsAgentClass().run_chunked(
            ideas=physics_ideas_dicts,
            cycle_id=cycle_id,
            extra_context=extra,
        )
        db.save_agent_output(output)
        killed_by = {k.idea_id for k in output.kills}
        for kill in output.kills:
            db.kill_idea(kill.idea_id, kill.reason, kill.kill_category)

        agent_key = PhysicsAgentClass.__name__.lower().replace("engineer", "")
        for assessed in output.ideas:
            aid = assessed.id[:8]
            physics_assessments[f"{agent_key}_{aid}"] = {
                "score": assessed.diamond_score_partial.physics_feasibility,
                "id": assessed.id,
            }

        score_map: dict[str, float] = {}
        for assessed in output.ideas:
            score_map[assessed.id] = assessed.diamond_score_partial.physics_feasibility

        for orig in novel_surviving:
            new_score = score_map.get(orig.id)
            if new_score is None:
                candidates = [(k, v) for k, v in score_map.items()
                              if k and (orig.id.startswith(k) or k.startswith(orig.id))]
                if len(candidates) == 1:
                    new_score = candidates[0][1]
            if new_score is not None:
                existing = orig.diamond_score_partial.physics_feasibility
                blended  = existing * PHYSICS_GATE_WEIGHT + new_score * PHYSICS_AGENT_WEIGHT if existing > 0 else new_score
                orig.diamond_score_partial.physics_feasibility = round(blended, 2)

        novel_surviving = [i for i in novel_surviving if i.id not in killed_by]

    for idea in novel_surviving:
        db.save_idea(idea, cycle_id)

    time.sleep(AGENT_DELAY_SECONDS)
    db.update_heartbeat(cycle_id)
    market_output = MarketAnalyst().run_chunked(
        ideas=[i.model_dump(mode="json") for i in novel_surviving],
        cycle_id=cycle_id,
        extra_context={"findings": findings_as_dicts},
    )
    db.save_agent_output(market_output)

    novel_by_id_market: dict[str, Idea] = {idea.id: idea for idea in novel_surviving}
    market_assessments = {}
    for a in market_output.metadata.get("market_assessments", []):
        aid = a.get("idea_id", "")
        market_assessments[aid[:8]] = a
        orig = novel_by_id_market.get(aid)
        if orig is None and aid:
            candidates = [v for k, v in novel_by_id_market.items()
                          if k.startswith(aid) or aid.startswith(k)]
            if len(candidates) == 1:
                orig = candidates[0]
        if orig is not None:
            orig.diamond_score_partial.market_pain = float(a.get("market_pain_score", 0))

    time.sleep(AGENT_DELAY_SECONDS)
    db.update_heartbeat(cycle_id)
    cost_output = CostAnalyst().run_chunked(
        ideas=[i.model_dump(mode="json") for i in novel_surviving],
        cycle_id=cycle_id,
        extra_context={"market_assessments": market_assessments},
    )
    db.save_agent_output(cost_output)

    novel_by_id_cost: dict[str, Idea] = {idea.id: idea for idea in novel_surviving}
    for a in cost_output.metadata.get("cost_assessments", []):
        aid  = a.get("idea_id", "")
        orig = novel_by_id_cost.get(aid)
        if orig is None and aid:
            candidates = [v for k, v in novel_by_id_cost.items()
                          if k.startswith(aid) or aid.startswith(k)]
            if len(candidates) == 1:
                orig = candidates[0]
        if orig is not None:
            orig.diamond_score_partial.scalability = float(a.get("scalability_score", 0))

    for idea in novel_surviving:
        db.save_idea(idea, cycle_id)

    logger.info(f"[Cycle2] Complete — {len(novel_surviving)} ideas survived")
    return novel_surviving


def run_cycle_3_kill_round(cycle_id: str, ideas: list[Idea]):
    from agents.market.competition_analyst import CompetitionAnalyst
    from agents.critics.devils_advocate import DevilsAdvocate

    logger.info("=" * 60)
    logger.info("CYCLE 3 — KILL ROUND (18:00 UTC)")
    logger.info("=" * 60)

    db.update_heartbeat(cycle_id)

    time.sleep(AGENT_DELAY_SECONDS)
    comp_output = CompetitionAnalyst().run_chunked(
        ideas=[i.model_dump(mode="json") for i in ideas],
        cycle_id=cycle_id,
    )
    db.save_agent_output(comp_output)
    active_ids_comp = {i.id for i in ideas}
    comp_killed = {k.idea_id for k in comp_output.kills if k.idea_id in active_ids_comp}
    for kill in comp_output.kills:
        if kill.idea_id in active_ids_comp:
            db.kill_idea(kill.idea_id, kill.reason, kill.kill_category)
    ideas = [i for i in ideas if i.id not in comp_killed]
    logger.info(f"[Competition] Killed {len(comp_killed)}, {len(ideas)} remaining")

    time.sleep(AGENT_DELAY_SECONDS)
    db.update_heartbeat(cycle_id)
    devil_output = DevilsAdvocate().run_chunked(
        ideas=[i.model_dump(mode="json") for i in ideas],
        cycle_id=cycle_id,
    )
    db.save_agent_output(devil_output)
    active_ids  = {i.id for i in ideas}
    devil_killed = {k.idea_id for k in devil_output.kills if k.idea_id in active_ids}
    for kill in devil_output.kills:
        if kill.idea_id in active_ids:
            db.kill_idea(kill.idea_id, kill.reason, kill.kill_category)
    surviving = [i for i in ideas if i.id not in devil_killed]
    diamond_candidates = devil_output.metadata.get("diamond_candidates", [])
    logger.info(f"[DevilsAdvocate] Killed {len(devil_killed)}, {len(surviving)} surviving")
    if diamond_candidates:
        logger.info(f"[DevilsAdvocate] DIAMOND CANDIDATES identified: {diamond_candidates}")
    return surviving, devil_output.open_questions


def run_cycle_4_director(cycle_id: str, ideas: list[Idea], open_questions: list, all_findings: list) -> None:
    from agents.management.chief_scientist import ChiefScientist
    from agents.meta.hypothesis_generator import HypothesisGenerator
    from config.settings import SCORE_KILL, SCORE_ARCHIVE, SCORE_ACTIVE, SCORE_PRIORITY
    from db.feedback import load_feedback_signals, apply_feedback_to_scores
    from db.lineage import get_top_yielding_sources

    logger.info("=" * 60)
    logger.info("CYCLE 4 — DIRECTOR (23:00 UTC)")
    logger.info("=" * 60)

    findings_dicts = [
        f if isinstance(f, dict) else f.model_dump(mode="json") if hasattr(f, "model_dump") else f
        for f in all_findings[:50]
    ]

    feedback_signals = []
    try:
        feedback_signals = load_feedback_signals(db.get_client(), limit=50)
        if feedback_signals:
            logger.info(f"[Feedback] {len(feedback_signals)} human ratings loaded")
    except Exception as e:
        logger.warning(f"[Feedback] Could not load feedback (non-blocking): {e}")

    top_sources = []
    try:
        top_sources = get_top_yielding_sources(db.get_client(), top_n=8, days_back=30)
    except Exception:
        pass

    strategy     = db.load_search_strategy()
    coupling_map = strategy.get("coupling_map", [])
    eng_limits   = strategy.get("engineering_limits_worth_challenging", [])

    chief_near_miss = []
    try:
        from db.sim_feedback_loop import load_near_miss_ideas
        chief_near_miss = load_near_miss_ideas(db.get_client(), limit=15)
    except Exception:
        pass

    context = {
        "surviving_ideas":                     [i.model_dump(mode="json") for i in ideas],
        "kills":                               [],
        "findings":                            findings_dicts,
        "open_questions":                      open_questions,
        "feedback_signals":                    feedback_signals,
        "top_yielding_sources":                top_sources,
        "coupling_map":                        coupling_map,
        "engineering_limits_worth_challenging": eng_limits,
        "near_miss_ideas":                     chief_near_miss,
    }

    db.update_heartbeat(cycle_id)
    output = ChiefScientist().run(context, cycle_id)
    db.save_agent_output(output)

    scored_ideas = output.metadata.get("scored_ideas", [])
    if feedback_signals:
        scored_ideas = apply_feedback_to_scores(scored_ideas, feedback_signals)

    for scored in scored_ideas:
        score  = float(scored.get("diamond_score", 0))
        status = scored.get("status", "active")
        if score < SCORE_KILL:
            status = "killed"
        elif score < SCORE_ARCHIVE:
            status = "archived"
        elif score < SCORE_ACTIVE:
            status = "archived"
        elif score < SCORE_PRIORITY:
            status = "active"
        else:
            status = "diamond"
            logger.info(f"DIAMOND FOUND: {scored.get('idea_id','?')[:8]} — score {score:.2f}")
        db.update_diamond_score(
            scored.get("idea_id", ""),
            score,
            status,
            physics=float(scored.get("physics_score", 0)) or None,
            market=float(scored.get("market_score", 0)) or None,
            novelty=float(scored.get("novelty_score", 0)) or None,
            scalability=float(scored.get("scalability_score", 0)) or None,
        )

    logger.info("[HypothesisGenerator] Analyzing kill patterns for assumed constraints...")
    try:
        killed_ideas = db.load_killed_ideas_full(limit=60)
        from config.settings import MIN_KILLS_FOR_HYPOTHESIS
        if len(killed_ideas) < MIN_KILLS_FOR_HYPOTHESIS:
            logger.info(
                f"[HypothesisGenerator] Skipped — only {len(killed_ideas)} kills "
                f"(need {MIN_KILLS_FOR_HYPOTHESIS})"
            )
        else:
            sim_kill_patterns = []
            near_miss_ideas   = []
            measured_limits   = []
            try:
                from db.sim_feedback_loop import (
                    load_sim_kill_patterns, load_near_miss_ideas, load_measured_physics_limits
                )
                sim_kill_patterns = load_sim_kill_patterns(db.get_client(), limit=100)
                near_miss_ideas   = load_near_miss_ideas(db.get_client(), limit=20)
                measured_limits   = load_measured_physics_limits(db.get_client())
                logger.info(
                    f"[SimFeedback] {len(sim_kill_patterns)} sim patterns, "
                    f"{len(near_miss_ideas)} near-miss, {len(measured_limits)} measured limits"
                )
            except Exception as sim_e:
                logger.warning(f"[SimFeedback] Load failed (non-blocking): {sim_e}")

            hyp_context = {
                "killed_ideas":              killed_ideas,
                "surviving_ideas":           [i.model_dump(mode="json") for i in ideas],
                "kill_patterns":             strategy.get("kill_patterns", []),
                "physics_limits_this_cycle": [],
                "sim_kill_patterns":         sim_kill_patterns,
                "near_miss_ideas":           near_miss_ideas,
                "measured_physics_limits":   measured_limits,
            }
            time.sleep(AGENT_DELAY_SECONDS)
            hyp_output = HypothesisGenerator().run(hyp_context, cycle_id)
            db.save_agent_output(hyp_output)

            hypotheses = hyp_output.metadata.get("hypotheses", [])
            if hypotheses:
                db.save_hypotheses(hypotheses, cycle_id)
                logger.info(f"[HypothesisGenerator] {len(hypotheses)} hypotheses saved")
                high_priority = [h for h in hypotheses if h.get("priority") == "high"]
                if high_priority:
                    hyp_searches = hyp_output.metadata.get("recommended_searches", [])
                    strategy     = db.load_search_strategy()
                    existing_kw  = strategy.get("new_keywords", [])
                    strategy["new_keywords"] = list(dict.fromkeys(existing_kw + hyp_searches[:3]))[:20]
                    db.save_search_strategy(strategy)
                    logger.info(f"[HypothesisGenerator] Added {len(hyp_searches[:3])} hypothesis-driven search terms")
    except Exception as e:
        logger.warning(f"[HypothesisGenerator] Failed (non-blocking): {e}")

    today_ideas    = db.load_active_ideas()
    today_findings = db.load_today_findings()
    day_counts = {
        "generated": len(today_ideas),
        "killed":    sum(1 for s in scored_ideas if s.get("status") == "killed"),
        "diamonds":  sum(1 for s in scored_ideas if s.get("status") == "diamond"),
    }
    logger.info(f"[Cycle4] Complete — {output.metadata.get('executive_summary', '')}")
    _write_daily_summary(output.metadata, cycle_id, today_findings, day_counts)

    try:
        plan          = output.metadata.get("next_cycle_plan", {})
        domain_rates  = db.get_domain_success_rates(days_back=14)
        killed_sample = db.load_killed_ideas_sample(limit=20)

        top_domains  = [d for d, r in sorted(domain_rates.items(), key=lambda x: x[1] or 0, reverse=True) if r is not None][:4]
        weak_domains = [d for d, r in sorted(domain_rates.items(), key=lambda x: x[1] or 0) if r is not None and r < 0.15]

        # ── Memory: collect titles of killed ideas to avoid repeating them ──
        kill_patterns = list({k.get("domain", "?") for k in killed_sample})
        killed_titles = [k.get("title", "")[:80] for k in killed_sample if k.get("title")][:15]

        # ── Memory: extract keyword signals from successful findings ─────────
        recent_findings_kw = []
        try:
            recent = db.load_recent_findings(days_back=7, limit=30)
            for f in recent:
                title = (f.get("title") or "")
                # Extract short 2-3 word phrases from finding titles
                words = title.lower().split()
                for i in range(len(words)-1):
                    phrase = " ".join(words[i:i+2])
                    if len(phrase) > 5 and phrase not in recent_findings_kw:
                        recent_findings_kw.append(phrase)
            recent_findings_kw = recent_findings_kw[:10]
        except Exception:
            recent_findings_kw = []

        # Merge new_keywords from Director output + finding signals
        evolved_new_keywords = list(dict.fromkeys(
            plan.get("new_keywords", []) + recent_findings_kw
        ))[:20]

        strategy = {
            "priority_domains":     plan.get("priority_domains", top_domains),
            "new_keywords":         evolved_new_keywords,
            "target_companies":     plan.get("target_companies", []),
            "specific_targets":     plan.get("specific_targets", []),
            "top_domains":          top_domains,
            "weak_domains":         weak_domains,
            "domain_success_rates": domain_rates,
            "kill_patterns":        kill_patterns,
            "killed_idea_titles":   killed_titles,   # memory: don't repeat these
        }
        db.save_search_strategy(strategy)
        logger.info(
            f"[StrategyFeedback] Saved — top: {top_domains} | "
            f"weak: {weak_domains} | kill patterns: {strategy['kill_patterns']} | "
            f"evolved_keywords: {len(evolved_new_keywords)}"
        )
    except Exception as e:
        logger.warning(f"[StrategyFeedback] Failed (non-blocking): {e}")


def _write_daily_summary(metadata: dict, cycle_id: str, findings: list = None, counts: dict = None) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    findings = findings or []
    counts   = counts or {}
    plan     = metadata.get("next_cycle_plan", {})

    db.save_daily_summary(cycle_id, metadata, date_str, counts)

    path = Path(f"summaries/{date_str}.md")
    path.parent.mkdir(exist_ok=True)

    top_findings    = sorted(findings, key=lambda f: f.get("confidence", 0) if isinstance(f, dict) else 0, reverse=True)[:6]
    company_signals = [f for f in findings if isinstance(f, dict) and f.get("company_signal")][:4]

    with path.open("w") as f:
        f.write(f"# R&D Engine — {date_str}\n\n")
        f.write(
            f"**Ideas:** {counts.get('generated','?')} generated · "
            f"{counts.get('killed','?')} killed · "
            f"{counts.get('diamonds','?')} diamonds\n\n"
        )
        f.write("## What Was Researched Today\n")
        if top_findings:
            for fn in top_findings:
                conf = fn.get("confidence", 0)
                f.write(f"- **[{fn.get('domain','?')}]** {fn.get('title','')} *(confidence: {conf:.0%})*\n")
        else:
            f.write("- No findings logged\n")

        if company_signals:
            f.write("\n## Company Signals\n")
            for fn in company_signals:
                f.write(f"- **{fn.get('company_signal')}**: {fn.get('title','')}\n")

        f.write(f"\n## Executive Summary\n{metadata.get('executive_summary', '—')}\n")

        diamonds = metadata.get("diamonds", [])
        f.write(f"\n## Diamonds ({len(diamonds)})\n")
        for d in diamonds:
            f.write(f"- {d}\n")
        if not diamonds:
            f.write("- None today\n")

        patterns = metadata.get("cross_domain_patterns", [])
        if patterns:
            f.write("\n## Patterns\n")
            for p in patterns:
                f.write(f"- {p}\n")

        f.write("\n## Tomorrow\n")
        f.write(f"**Domains:** {', '.join(plan.get('priority_domains', ['—']))}\n")
        f.write(f"**Keywords:** {', '.join(plan.get('new_keywords', ['—']))}\n")
        if plan.get("specific_targets"):
            f.write("**Targets:**\n")
            for t in plan.get("specific_targets", []):
                f.write(f"- {t}\n")

    logger.info(f"[Summary] Written to {path} and saved to Supabase")


def run_weekly_maintenance() -> None:
    from agents.meta.weekly_analyst import WeeklyAnalyst
    from datetime import timedelta

    logger.info("=" * 60)
    logger.info("WEEKLY MAINTENANCE")
    logger.info("=" * 60)

    cleanup_counts = db.cleanup_weak_ideas(
        score_delete=3.0,
        score_archive=5.0,
        min_age_days=7,
    )
    logger.info(f"[Cleanup] {cleanup_counts['deleted']} deleted, {cleanup_counts['archived']} archived")

    week_stats = db.get_week_stats(days_back=7)
    if not week_stats or not week_stats.get("stats", {}).get("generated", 0):
        logger.info("[Weekly] No week stats yet — running cleanup only")
        try:
            strategy = db.load_search_strategy()
            kw = strategy.get("new_keywords", [])
            if kw:
                _update_seed_keywords(kw)
        except Exception:
            pass
        return

    today      = datetime.now()
    week_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    week_end   = today.strftime("%Y-%m-%d")

    context = {
        "week_start":       week_start,
        "week_end":         week_end,
        "week_stats":       {**week_stats.get("stats", {}), **cleanup_counts},
        "top_ideas":        week_stats.get("top_ideas", []),
        "dead_end_domains": week_stats.get("dead_end_domains", []),
    }

    cycle_id = str(uuid.uuid4())
    output   = WeeklyAnalyst().run(context, cycle_id)

    report_md = output.metadata.get("report_markdown", "")
    quality   = output.metadata.get("week_quality", "unknown")
    logger.info(f"[Weekly] Report generated — quality: {quality}")

    db.save_weekly_report(report_md, output.metadata, week_start)
    _update_seed_keywords(output.metadata.get("priority_keywords_next_week", []))

    path = Path(f"summaries/weekly-{week_start}.md")
    path.parent.mkdir(exist_ok=True)
    path.write_text(report_md)
    logger.info(f"[Weekly] Report written to {path}")


def _update_seed_keywords(new_keywords: list) -> None:
    if not new_keywords:
        return
    seed_path = Path("config/seed.json")
    try:
        with seed_path.open() as f:
            seed = json.load(f)
        existing = seed.get("seed_keywords", [])
        merged   = list(dict.fromkeys(existing[:15] + new_keywords))[:20]
        seed["seed_keywords"] = merged
        with seed_path.open("w") as f:
            json.dump(seed, f, indent=2)
        logger.info(f"[Seed] Updated with {len(new_keywords)} new keywords")
    except Exception as e:
        logger.warning(f"[Seed] Could not update seed.json: {e}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous R&D Engine")
    parser.add_argument("--cycle",        type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--test-physics", action="store_true")
    parser.add_argument("--cold-start",   action="store_true")
    parser.add_argument("--weekly",       action="store_true")
    parser.add_argument("--force",        action="store_true",
                        help="Force run even if cycle already done today (bypass daily guard)")
    args = parser.parse_args()

    if args.test_physics:
        from tests.test_physics import run_all_tests
        run_all_tests()
        return

    if args.weekly:
        run_weekly_maintenance()
        return

    seed         = load_seed()
    cycle_number = args.cycle or 1
    cycle_id     = str(uuid.uuid4())

    if not args.force and db.is_cycle_already_done_today(cycle_number):
        logger.info(f"[Guard] Cycle {cycle_number} already done today — skipping (use --force to override)")
        return
    if db.is_another_worker_running(cycle_number):
        logger.info(f"[Guard] Another worker running cycle {cycle_number} — skipping")
        return

    cold_start = args.cold_start or db.is_cold_start()
    if cold_start:
        logger.info("[ColdStart] Starting fresh from seed.json")

    db.create_cycle(cycle_id, cycle_number)
    logger.info(f"[Engine] Starting cycle {cycle_number} — ID: {cycle_id}")

    try:
        if cycle_number == 1:
            run_cycle_1_harvest(cycle_id, seed)

        elif cycle_number == 2:
            cycle1_id = db.load_today_cycle_id(cycle_number=1)
            ideas     = db.load_active_ideas(cycle_id=cycle1_id)
            if not ideas:
                logger.warning("[Cycle2] No Cycle 1 ideas found — running harvest first")
                run_cycle_1_harvest(cycle_id, seed)
                ideas = db.load_active_ideas(cycle_id=cycle_id)
            run_cycle_2_physics_market(cycle_id, ideas)

        elif cycle_number == 3:
            cycle2_id = db.load_today_cycle_id(cycle_number=2)
            ideas     = db.load_active_ideas(cycle_id=cycle2_id)
            if not ideas:
                ideas = db.load_active_ideas()
            run_cycle_3_kill_round(cycle_id, ideas)

        elif cycle_number == 4:
            ideas          = db.load_active_ideas()
            all_findings   = db.load_today_findings()
            open_questions = []
            try:
                cycle3_id = db.load_today_cycle_id(cycle_number=3)
                if cycle3_id:
                    client = db.get_client()
                    res    = client.table("agent_outputs") \
                        .select("output").eq("cycle_id", cycle3_id) \
                        .eq("agent_name", "devils_advocate").limit(1).execute()
                    if res.data:
                        open_questions = res.data[0]["output"].get("open_questions", [])
            except Exception as e:
                logger.warning(f"[Cycle4] Could not load open questions: {e}")
            run_cycle_4_director(cycle_id, ideas, open_questions, all_findings)

        db.complete_cycle(cycle_id, "done")
        logger.info(f"[Engine] Cycle {cycle_number} completed successfully")

    except Exception as e:
        logger.error(f"[Engine] Cycle {cycle_number} failed: {e}", exc_info=True)
        db.complete_cycle(cycle_id, "failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
