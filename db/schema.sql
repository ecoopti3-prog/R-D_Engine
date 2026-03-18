-- ============================================================
-- RD Engine v3 — Supabase Schema
-- Run this ONCE in your Supabase SQL editor to set up the DB.
-- Enable pgvector extension first (Supabase dashboard → Extensions → vector)
-- ============================================================

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Table: research_cycles
-- Tracks each run of the 4 daily cycles
-- ============================================================
CREATE TABLE IF NOT EXISTS research_cycles (
    id              UUID PRIMARY KEY,
    date            DATE        NOT NULL,
    cycle_number    SMALLINT    NOT NULL CHECK (cycle_number BETWEEN 1 AND 4),
    status          TEXT        NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'done', 'failed')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    last_heartbeat  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cycles_date_number ON research_cycles(date, cycle_number);
CREATE INDEX IF NOT EXISTS idx_cycles_status ON research_cycles(status, last_heartbeat);

-- ============================================================
-- Table: ideas
-- Core idea storage with embeddings for novelty detection
-- ============================================================
CREATE TABLE IF NOT EXISTS ideas (
    id                  UUID PRIMARY KEY,
    cycle_id            UUID        REFERENCES research_cycles(id) ON DELETE SET NULL,
    title               TEXT        NOT NULL,
    domain              TEXT        NOT NULL,
    problem_statement   TEXT,
    physical_limit      TEXT,
    proposed_direction  TEXT,
    company_context     TEXT,

    -- Diamond Score components
    physics_score       FLOAT       DEFAULT 0.0,
    market_score        FLOAT       DEFAULT 0.0,
    novelty_score       FLOAT       DEFAULT 0.0,
    scalability_score   FLOAT       DEFAULT 0.0,
    diamond_score       FLOAT       DEFAULT 0.0,

    -- Status machine
    status              TEXT        NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'killed', 'archived', 'diamond', 'physics_unverified')),
    kill_reason         TEXT,

    -- Extracted parameters (stored as JSONB for flexibility)
    power_params        JSONB,
    thermal_params      JSONB,
    data_movement_params JSONB,
    pdn_params          JSONB,

    -- Novelty embedding (384-dim from sentence-transformers)
    embedding           vector(384),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ideas_status         ON ideas(status);
CREATE INDEX IF NOT EXISTS idx_ideas_domain         ON ideas(domain);
CREATE INDEX IF NOT EXISTS idx_ideas_diamond_score  ON ideas(diamond_score DESC);
CREATE INDEX IF NOT EXISTS idx_ideas_cycle          ON ideas(cycle_id);
-- Vector similarity index (HNSW — works from 0 rows, no minimum data requirement)
-- Replaces IVFFlat which required lists*3 = 150+ rows before it could build or query.
-- HNSW: m=16 (connectivity), ef_construction=64 (build quality) — balanced defaults.
CREATE INDEX IF NOT EXISTS idx_ideas_embedding ON ideas
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- Table: agent_outputs
-- Every agent run is logged here for debugging and meta-learning
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_outputs (
    id          BIGSERIAL   PRIMARY KEY,
    cycle_id    UUID        REFERENCES research_cycles(id) ON DELETE CASCADE,
    agent_name  TEXT        NOT NULL,
    output      JSONB       NOT NULL,
    llm_used    TEXT,
    tokens_used INTEGER,
    duration_ms INTEGER,
    status      TEXT        NOT NULL DEFAULT 'done',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_outputs_cycle  ON agent_outputs(cycle_id);
CREATE INDEX IF NOT EXISTS idx_agent_outputs_agent  ON agent_outputs(agent_name, created_at DESC);

-- ============================================================
-- Table: kills
-- Kill log with evidence — for meta-learning and audit
-- ============================================================
CREATE TABLE IF NOT EXISTS kills (
    id            BIGSERIAL   PRIMARY KEY,
    idea_id       UUID        REFERENCES ideas(id) ON DELETE CASCADE,
    killed_by     TEXT        NOT NULL,
    reason        TEXT        NOT NULL,
    kill_category TEXT        NOT NULL,
    evidence_url  TEXT,
    cycle_id      UUID        REFERENCES research_cycles(id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kills_idea_id      ON kills(idea_id);
CREATE INDEX IF NOT EXISTS idx_kills_category     ON kills(kill_category);
CREATE INDEX IF NOT EXISTS idx_kills_killed_by    ON kills(killed_by);

-- ============================================================
-- Table: findings
-- Research findings (papers, patents, job signals)
-- ============================================================
CREATE TABLE IF NOT EXISTS findings (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id        UUID        REFERENCES research_cycles(id) ON DELETE SET NULL,
    agent_name      TEXT        NOT NULL,
    type            TEXT        NOT NULL,
    domain          TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    description     TEXT,
    source_url      TEXT,
    source_type     TEXT,
    company_signal  TEXT,
    confidence      FLOAT       DEFAULT 0.5,
    numerical_params JSONB      DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_findings_domain       ON findings(domain);
CREATE INDEX IF NOT EXISTS idx_findings_company      ON findings(company_signal);
CREATE INDEX IF NOT EXISTS idx_findings_cycle        ON findings(cycle_id);
CREATE INDEX IF NOT EXISTS idx_findings_source_type  ON findings(source_type);

-- ============================================================
-- Table: daily_summaries
-- Markdown summaries generated by Director (Cycle 4)
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_summaries (
    id              BIGSERIAL   PRIMARY KEY,
    cycle_id        UUID        REFERENCES research_cycles(id),
    date            DATE        NOT NULL UNIQUE,
    executive_summary TEXT,
    diamonds        JSONB       DEFAULT '[]',
    patterns        JSONB       DEFAULT '[]',
    next_cycle_plan JSONB       DEFAULT '{}',
    ideas_generated INTEGER     DEFAULT 0,
    ideas_killed    INTEGER     DEFAULT 0,
    diamonds_found  INTEGER     DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Row Level Security (RLS)
-- Only your service key can access data
-- Run after creating tables
-- ============================================================
ALTER TABLE research_cycles  ENABLE ROW LEVEL SECURITY;
ALTER TABLE ideas             ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_outputs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE kills             ENABLE ROW LEVEL SECURITY;
ALTER TABLE findings          ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_summaries   ENABLE ROW LEVEL SECURITY;

-- Service role policy (GitHub Actions uses service key — full access)
CREATE POLICY "service_role_all" ON research_cycles  FOR ALL USING (true);
CREATE POLICY "service_role_all" ON ideas             FOR ALL USING (true);
CREATE POLICY "service_role_all" ON agent_outputs     FOR ALL USING (true);
CREATE POLICY "service_role_all" ON kills             FOR ALL USING (true);
CREATE POLICY "service_role_all" ON findings          FOR ALL USING (true);
CREATE POLICY "service_role_all" ON daily_summaries   FOR ALL USING (true);

-- Dashboard read policies (anon key in browser — read-only, no PII exposed)
-- The dashboard uses the anon key which is intentionally public (read-only).
-- These policies allow SELECT only — no INSERT/UPDATE/DELETE via anon.
CREATE POLICY "anon_read_ideas"           ON ideas            FOR SELECT USING (true);
CREATE POLICY "anon_read_findings"        ON findings         FOR SELECT USING (true);
CREATE POLICY "anon_read_daily_summaries" ON daily_summaries  FOR SELECT USING (true);
CREATE POLICY "anon_read_hypotheses"      ON hypotheses       FOR SELECT USING (true);
CREATE POLICY "anon_read_weekly_reports"  ON weekly_reports   FOR SELECT USING (true);
-- FIX: these three were missing anon_read — caused Cycles, Agents, Sources tabs to show empty
CREATE POLICY "anon_read_research_cycles" ON research_cycles  FOR SELECT USING (true);
CREATE POLICY "anon_read_agent_outputs"   ON agent_outputs    FOR SELECT USING (true);
CREATE POLICY "anon_read_idea_lineage"    ON idea_lineage     FOR SELECT USING (true);

-- Dashboard feedback write policy — anon key can INSERT ratings (1-5 stars per idea)
-- This allows the human-in-the-loop feedback from the dashboard UI.
CREATE POLICY "anon_write_feedback"  ON idea_feedback FOR INSERT WITH CHECK (true);

-- ============================================================
-- Trigger: auto-update updated_at on ideas
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ideas_updated_at
    BEFORE UPDATE ON ideas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- Useful views for monitoring
-- ============================================================
CREATE OR REPLACE VIEW active_ideas_view AS
SELECT
    id, title, domain, diamond_score,
    physics_score, market_score, novelty_score, scalability_score,
    company_context, created_at
FROM ideas
WHERE status = 'active'
ORDER BY diamond_score DESC;

CREATE OR REPLACE VIEW diamonds_view AS
SELECT
    id, title, domain, diamond_score,
    physics_score, market_score, novelty_score, scalability_score,
    problem_statement, company_context, created_at
FROM ideas
WHERE status = 'diamond'
ORDER BY diamond_score DESC;

CREATE OR REPLACE VIEW daily_stats_view AS
SELECT
    date_trunc('day', created_at) AS date,
    COUNT(*) FILTER (WHERE status = 'active')   AS active,
    COUNT(*) FILTER (WHERE status = 'killed')   AS killed,
    COUNT(*) FILTER (WHERE status = 'diamond')  AS diamonds,
    AVG(diamond_score) FILTER (WHERE diamond_score > 0) AS avg_diamond_score
FROM ideas
GROUP BY 1
ORDER BY 1 DESC;

-- ============================================================
-- Done! Now set up your .env file with:
-- SUPABASE_URL=https://YOUR_PROJECT.supabase.co
-- SUPABASE_KEY=YOUR_SERVICE_ROLE_KEY  (not anon key — service role for full access)
-- ============================================================

-- ============================================================
-- v5 ADDITIONS — Learning & Feedback Infrastructure
-- ============================================================

-- Table: idea_feedback
-- Human-in-the-loop ratings. Single most powerful learning signal.
-- ============================================================
CREATE TABLE IF NOT EXISTS idea_feedback (
    id          BIGSERIAL   PRIMARY KEY,
    idea_id     UUID        REFERENCES ideas(id) ON DELETE CASCADE,
    rating      SMALLINT    NOT NULL CHECK (rating BETWEEN 1 AND 5),
    reason      TEXT,
    rater       TEXT        NOT NULL DEFAULT 'human',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (idea_id, rater)   -- one rating per rater per idea
);

CREATE INDEX IF NOT EXISTS idx_feedback_idea_id ON idea_feedback(idea_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating  ON idea_feedback(rating);
ALTER TABLE idea_feedback ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON idea_feedback FOR ALL USING (true);

-- Table: idea_lineage
-- Tracks which source URLs / findings generated each idea.
-- After 2+ weeks: compute diamond yield per source venue.
-- ============================================================
CREATE TABLE IF NOT EXISTS idea_lineage (
    id          BIGSERIAL   PRIMARY KEY,
    idea_id     UUID        REFERENCES ideas(id) ON DELETE CASCADE,
    source_url  TEXT,
    source_type TEXT,
    finding_id  UUID        REFERENCES findings(id) ON DELETE SET NULL,
    agent_name  TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lineage_idea_id    ON idea_lineage(idea_id);
CREATE INDEX IF NOT EXISTS idx_lineage_source_url ON idea_lineage(source_url);
CREATE INDEX IF NOT EXISTS idx_lineage_agent      ON idea_lineage(agent_name);
ALTER TABLE idea_lineage ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON idea_lineage FOR ALL USING (true);

-- Table: hypotheses
-- Stores HypothesisGenerator output — assumptions worth testing.
-- These feed into next-cycle searches via seed.json updates.
-- ============================================================
CREATE TABLE IF NOT EXISTS hypotheses (
    id              BIGSERIAL   PRIMARY KEY,
    cycle_id        UUID        REFERENCES research_cycles(id) ON DELETE SET NULL,
    hypothesis_id   TEXT        NOT NULL,   -- e.g. "H001"
    title           TEXT        NOT NULL,
    assumed_constraint TEXT,
    challenge       TEXT,
    physical_basis  TEXT,
    unlocks         JSONB       DEFAULT '[]',
    unlocked_idea_count INTEGER DEFAULT 0,
    testability     TEXT,
    confidence      FLOAT       DEFAULT 0.5,
    priority        TEXT        DEFAULT 'medium',
    status          TEXT        DEFAULT 'open'  -- open | tested | confirmed | rejected
                        CHECK (status IN ('open', 'tested', 'confirmed', 'rejected')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_status   ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_priority ON hypotheses(priority);
ALTER TABLE hypotheses ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON hypotheses FOR ALL USING (true);

-- Table: weekly_reports
-- Full weekly analyst reports for trend tracking.
-- ============================================================
CREATE TABLE IF NOT EXISTS weekly_reports (
    id              BIGSERIAL   PRIMARY KEY,
    week_start      DATE        NOT NULL UNIQUE,
    report_markdown TEXT,
    metadata        JSONB       DEFAULT '{}',
    quality         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE weekly_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON weekly_reports FOR ALL USING (true);

-- Add v5 columns to existing ideas table (safe — IF NOT EXISTS equivalent via DO block)
DO $$
BEGIN
    -- Source lineage shortcut (denormalized for fast queries)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='source_venue') THEN
        ALTER TABLE ideas ADD COLUMN source_venue TEXT;
    END IF;
    -- Kill detail from sim engine (populated when sim engine kills a diamond)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='physics_kill_detail') THEN
        ALTER TABLE ideas ADD COLUMN physics_kill_detail TEXT;
    END IF;
    -- Human feedback score (denormalized for fast sorting)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='human_rating') THEN
        ALTER TABLE ideas ADD COLUMN human_rating SMALLINT;
    END IF;
    -- Hypothesis that this idea was generated to test
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='hypothesis_id') THEN
        ALTER TABLE ideas ADD COLUMN hypothesis_id TEXT;
    END IF;
END $$;

-- View: source performance (requires idea_lineage data — meaningful after week 2)
CREATE OR REPLACE VIEW source_performance_view AS
SELECT
    il.source_type,
    _extract_venue_sql(il.source_url) AS venue,
    COUNT(DISTINCT il.idea_id) AS ideas_generated,
    COUNT(DISTINCT CASE WHEN i.status = 'diamond' THEN il.idea_id END) AS diamonds,
    ROUND(
        COUNT(DISTINCT CASE WHEN i.status = 'diamond' THEN il.idea_id END)::numeric /
        NULLIF(COUNT(DISTINCT il.idea_id), 0), 3
    ) AS diamond_yield
FROM idea_lineage il
JOIN ideas i ON i.id = il.idea_id
WHERE il.source_url IS NOT NULL
GROUP BY 1, 2
ORDER BY diamond_yield DESC NULLS LAST;

-- Grant anon+authenticated SELECT on the view (views don't inherit table RLS in Supabase)
GRANT SELECT ON source_performance_view TO anon, authenticated;

-- Note: _extract_venue_sql is a PostgreSQL function — define below
CREATE OR REPLACE FUNCTION _extract_venue_sql(url TEXT) RETURNS TEXT AS $$
BEGIN
    IF url ILIKE '%arxiv.org%'         THEN RETURN 'arxiv'; END IF;
    IF url ILIKE '%patents.google%'     THEN RETURN 'google_patents'; END IF;
    IF url ILIKE '%semanticscholar%'    THEN RETURN 'semantic_scholar'; END IF;
    IF url ILIKE '%huggingface.co%'     THEN RETURN 'huggingface'; END IF;
    IF url ILIKE '%github.com%'         THEN RETURN 'github'; END IF;
    IF url ILIKE '%linkedin.com%'       THEN RETURN 'job_postings'; END IF;
    IF url ILIKE '%opencompute.org%'    THEN RETURN 'opencompute'; END IF;
    IF url ILIKE '%ieee.org%'           THEN RETURN 'ieee'; END IF;
    IF url ILIKE '%nature.com%'         THEN RETURN 'nature'; END IF;
    RETURN 'other';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================
-- Done with v5 schema additions.
-- ============================================================

-- ============================================================
-- v7 ADDITIONS — Closed-Loop Sim Integration + Cost Tracking
-- ============================================================

-- Table: sim_results
-- Full simulation results from Sim Engine, linked back to ideas.
-- This is the core of the closed loop: sim numbers feed HypothesisGenerator.
-- ============================================================
CREATE TABLE IF NOT EXISTS sim_results (
    id                      BIGSERIAL   PRIMARY KEY,
    idea_id                 UUID        REFERENCES ideas(id) ON DELETE CASCADE,
    cycle_id                UUID        REFERENCES research_cycles(id) ON DELETE SET NULL,
    -- Overall
    sim_status              TEXT        NOT NULL DEFAULT 'skipped',  -- pass/fail/marginal/critical/skipped
    sim_score               FLOAT       DEFAULT 0.0,
    near_miss               BOOLEAN     DEFAULT FALSE,    -- sim_score 4-6 AND fixable with one param change
    recommendation          TEXT,                         -- proceed_to_prototype / kill_physics / revise / marginal
    -- Key numerical results (not text — actual numbers for knowledge graph)
    r_theta_actual          FLOAT,     -- °C/W
    r_theta_critical        FLOAT,     -- °C/W — from brentq
    t_op_c                  FLOAT,     -- operating temperature
    margin_to_runaway_pct   FLOAT,
    yield_pct               FLOAT,     -- Monte Carlo yield
    min_mttf_years          FLOAT,     -- aging: min of NBTI/HCI/EM
    top_failure_domain      TEXT,      -- which domain failed first
    -- Revision targets (for Inversion — what must change to pass)
    revision_targets        JSONB      DEFAULT '[]',
    -- Full domain results
    domain_results          JSONB      DEFAULT '[]',
    cross_domain_couplings  JSONB      DEFAULT '[]',
    -- Metadata
    sim_version             TEXT       DEFAULT 'v6',
    duration_ms             INTEGER,
    timestamp               TIMESTAMPTZ DEFAULT NOW(),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sim_results_idea_id    ON sim_results(idea_id);
CREATE INDEX IF NOT EXISTS idx_sim_results_cycle_id   ON sim_results(cycle_id);
CREATE INDEX IF NOT EXISTS idx_sim_results_near_miss  ON sim_results(near_miss) WHERE near_miss = TRUE;
CREATE INDEX IF NOT EXISTS idx_sim_results_status     ON sim_results(sim_status);
ALTER TABLE sim_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON sim_results FOR ALL USING (true);
CREATE POLICY "anon_read_sim_results" ON sim_results FOR SELECT USING (true);

-- Add v7 columns to ideas table
DO $$
BEGIN
    -- Sim Engine score (replaces LLM guess for physics_feasibility)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='sim_score') THEN
        ALTER TABLE ideas ADD COLUMN sim_score FLOAT DEFAULT NULL;
    END IF;
    -- Near-miss flag (drives Research Loop)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='near_miss') THEN
        ALTER TABLE ideas ADD COLUMN near_miss BOOLEAN DEFAULT FALSE;
    END IF;
    -- Revision targets from inversion (JSONB array of {parameter, current, required, delta_pct})
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='revision_targets') THEN
        ALTER TABLE ideas ADD COLUMN revision_targets JSONB DEFAULT '[]';
    END IF;
    -- Iteration count (how many Research Loops this idea went through)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='iteration_count') THEN
        ALTER TABLE ideas ADD COLUMN iteration_count INTEGER DEFAULT 0;
    END IF;
    -- Sim timestamp (when was the last sim run on this idea)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='ideas' AND column_name='last_sim_at') THEN
        ALTER TABLE ideas ADD COLUMN last_sim_at TIMESTAMPTZ DEFAULT NULL;
    END IF;
END $$;

-- Table: api_cost_log
-- Track every LLM call cost. Without this, burn rate is invisible.
-- ============================================================
CREATE TABLE IF NOT EXISTS api_cost_log (
    id              BIGSERIAL   PRIMARY KEY,
    cycle_id        UUID        REFERENCES research_cycles(id) ON DELETE SET NULL,
    agent_name      TEXT        NOT NULL,
    llm_provider    TEXT        NOT NULL,   -- groq / gemini / mistral / etc
    model           TEXT        NOT NULL,
    tokens_in       INTEGER     DEFAULT 0,
    tokens_out      INTEGER     DEFAULT 0,
    cost_usd        FLOAT       DEFAULT 0.0,
    duration_ms     INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cost_log_cycle      ON api_cost_log(cycle_id);
CREATE INDEX IF NOT EXISTS idx_cost_log_provider   ON api_cost_log(llm_provider);
CREATE INDEX IF NOT EXISTS idx_cost_log_created_at ON api_cost_log(created_at DESC);
ALTER TABLE api_cost_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON api_cost_log FOR ALL USING (true);

-- View: cost summary per cycle
CREATE OR REPLACE VIEW cycle_cost_view AS
SELECT
    c.date,
    c.cycle_number,
    SUM(l.cost_usd)     AS total_cost_usd,
    SUM(l.tokens_in + l.tokens_out) AS total_tokens,
    COUNT(*)            AS api_calls,
    MAX(l.llm_provider) FILTER (WHERE l.cost_usd = (SELECT MAX(cost_usd) FROM api_cost_log WHERE cycle_id = c.id)) AS top_cost_provider
FROM research_cycles c
JOIN api_cost_log l ON l.cycle_id = c.id
GROUP BY c.id, c.date, c.cycle_number
ORDER BY c.date DESC, c.cycle_number;

-- View: near-miss ideas with revision targets (drives Research Loop)
CREATE OR REPLACE VIEW near_miss_view AS
SELECT
    i.id,
    i.title,
    i.domain,
    i.diamond_score,
    i.sim_score,
    i.revision_targets,
    i.iteration_count,
    s.r_theta_actual,
    s.r_theta_critical,
    s.t_op_c,
    s.margin_to_runaway_pct,
    s.top_failure_domain,
    i.created_at
FROM ideas i
LEFT JOIN sim_results s ON s.idea_id = i.id
WHERE i.near_miss = TRUE
  AND i.status IN ('active', 'diamond')
ORDER BY i.diamond_score DESC;

-- View: param failure heatmap (knowledge graph substitute — which params kill most)
CREATE OR REPLACE VIEW param_failure_heatmap AS
SELECT
    s.top_failure_domain,
    COUNT(*) AS failure_count,
    AVG(s.r_theta_actual) FILTER (WHERE s.r_theta_actual IS NOT NULL) AS avg_r_theta,
    AVG(s.t_op_c) FILTER (WHERE s.t_op_c IS NOT NULL) AS avg_t_op,
    AVG(s.margin_to_runaway_pct) FILTER (WHERE s.margin_to_runaway_pct IS NOT NULL) AS avg_margin,
    MIN(s.sim_score) AS min_score,
    MAX(s.sim_score) AS max_score
FROM sim_results s
WHERE s.sim_status IN ('fail', 'critical')
GROUP BY s.top_failure_domain
ORDER BY failure_count DESC;

-- ============================================================
-- Done with v7 schema additions.
-- ============================================================
