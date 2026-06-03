"""Initial schema — mirrors scripts/init.sql

Revision ID: 0001
Revises:
Create Date: 2026-06-02

All DDL uses IF NOT EXISTS so this migration is safe to run against a database
that was already initialised by init.sql via the Docker entrypoint.
Run `make db-stamp` on those databases instead of `make db-upgrade`.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')

    op.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            prompt_id       UUID PRIMARY KEY,
            name            TEXT NOT NULL,
            version         INTEGER NOT NULL,
            content         TEXT NOT NULL,
            content_hash    TEXT GENERATED ALWAYS AS (md5(content)) STORED,
            model           TEXT NOT NULL DEFAULT 'claude-sonnet-4-20250514',
            temperature     FLOAT NOT NULL DEFAULT 0.7,
            max_tokens      INTEGER NOT NULL DEFAULT 2048,
            status          TEXT NOT NULL DEFAULT 'draft'
                            CHECK (status IN ('draft','testing','staging','production','deprecated','rolled_back')),
            tags            JSONB NOT NULL DEFAULT '[]',
            min_eval_score  FLOAT NOT NULL DEFAULT 0.8,
            created_by      TEXT NOT NULL DEFAULT 'system',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            promoted_at     TIMESTAMPTZ,
            metadata        JSONB NOT NULL DEFAULT '{}',
            UNIQUE (name, version)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_prompts_name_status ON prompts(name, status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_prompts_status ON prompts(status)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS prompt_audit (
            id          BIGSERIAL PRIMARY KEY,
            prompt_id   UUID NOT NULL REFERENCES prompts(prompt_id),
            action      TEXT NOT NULL,
            actor       TEXT NOT NULL,
            metadata    JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_prompt_audit_prompt_id ON prompt_audit(prompt_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS ab_tests (
            test_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            prompt_id           TEXT NOT NULL,
            control_version     INTEGER NOT NULL,
            treatment_version   INTEGER NOT NULL,
            traffic_split       FLOAT NOT NULL DEFAULT 0.5,
            min_samples         INTEGER NOT NULL DEFAULT 100,
            significance_level  FLOAT NOT NULL DEFAULT 0.05,
            active              BOOLEAN NOT NULL DEFAULT TRUE,
            started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended_at            TIMESTAMPTZ,
            conclusion          TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ab_tests_prompt_active ON ab_tests(prompt_id, active)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS ab_assignments (
            request_id          TEXT NOT NULL,
            test_id             UUID NOT NULL REFERENCES ab_tests(test_id),
            version_assigned    INTEGER NOT NULL,
            assigned_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (request_id, test_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ab_assignments_test_id ON ab_assignments(test_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS eval_queue (
            id              BIGSERIAL PRIMARY KEY,
            test_case_json  JSONB NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending','processing','done','failed')),
            priority        INTEGER NOT NULL DEFAULT 5,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at      TIMESTAMPTZ,
            completed_at    TIMESTAMPTZ,
            error           TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_eval_queue_status_priority ON eval_queue(status, priority, created_at)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            eval_run_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            test_case_id        UUID NOT NULL,
            request_id          TEXT NOT NULL,
            prompt_id           TEXT NOT NULL,
            prompt_version      INTEGER NOT NULL,
            overall_score       FLOAT NOT NULL,
            passed              BOOLEAN NOT NULL,
            metrics_json        JSONB NOT NULL DEFAULT '[]',
            judge_model         TEXT NOT NULL DEFAULT 'claude-sonnet-4-20250514',
            total_latency_ms    FLOAT,
            evaluated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_prompt ON eval_results(prompt_id, prompt_version, evaluated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_request_id ON eval_results(request_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS guardrail_log (
            id                  BIGSERIAL PRIMARY KEY,
            request_id          TEXT NOT NULL,
            session_id          TEXT,
            phase               TEXT NOT NULL CHECK (phase IN ('input','output')),
            verdict             TEXT NOT NULL,
            checks_json         JSONB NOT NULL DEFAULT '[]',
            redacted            BOOLEAN NOT NULL DEFAULT FALSE,
            blocked_reason      TEXT,
            total_latency_ms    FLOAT,
            logged_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_guardrail_log_verdict ON guardrail_log(verdict, logged_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_guardrail_log_request ON guardrail_log(request_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS quality_snapshots (
            snapshot_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            prompt_id               TEXT NOT NULL,
            prompt_version          INTEGER NOT NULL,
            window_start            TIMESTAMPTZ NOT NULL,
            window_end              TIMESTAMPTZ NOT NULL,
            sample_count            INTEGER NOT NULL,
            avg_overall_score       FLOAT NOT NULL,
            avg_hallucination_score FLOAT NOT NULL DEFAULT 0,
            avg_toxicity_score      FLOAT NOT NULL DEFAULT 0,
            avg_relevancy_score     FLOAT NOT NULL DEFAULT 0,
            pass_rate               FLOAT NOT NULL,
            p50_eval_latency_ms     FLOAT NOT NULL DEFAULT 0,
            p99_eval_latency_ms     FLOAT NOT NULL DEFAULT 0,
            guardrail_block_rate    FLOAT NOT NULL DEFAULT 0,
            pii_detection_rate      FLOAT NOT NULL DEFAULT 0,
            created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_quality_snapshots_prompt_time ON quality_snapshots(prompt_id, prompt_version, window_end DESC)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS drift_alerts (
            alert_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            prompt_id       TEXT NOT NULL,
            prompt_version  INTEGER NOT NULL,
            metric          TEXT NOT NULL,
            baseline_score  FLOAT NOT NULL,
            current_score   FLOAT NOT NULL,
            drop_pct        FLOAT NOT NULL,
            severity        TEXT NOT NULL CHECK (severity IN ('warning','critical')),
            auto_rollback   BOOLEAN NOT NULL DEFAULT FALSE,
            triggered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            resolved_at     TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_drift_alerts_prompt ON drift_alerts(prompt_id, triggered_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_drift_alerts_unresolved ON drift_alerts(resolved_at) WHERE resolved_at IS NULL")

    op.execute("""
        CREATE TABLE IF NOT EXISTS ci_gate_results (
            id              BIGSERIAL PRIMARY KEY,
            prompt_id       UUID NOT NULL REFERENCES prompts(prompt_id),
            passed          BOOLEAN NOT NULL,
            score           FLOAT NOT NULL,
            pass_rate       FLOAT NOT NULL,
            sample_count    INTEGER NOT NULL,
            failures_json   JSONB NOT NULL DEFAULT '[]',
            run_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # Seed data — ON CONFLICT DO NOTHING so it's safe to re-run
    op.execute("""
        INSERT INTO prompts (prompt_id, name, version, content, status, created_by)
        VALUES (
            uuid_generate_v4(),
            'default-assistant',
            1,
            'You are a helpful, accurate, and concise AI assistant. Always respond in the same language as the user. If you are unsure about something, say so clearly rather than guessing.',
            'production',
            'system'
        )
        ON CONFLICT (name, version) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ci_gate_results CASCADE")
    op.execute("DROP TABLE IF EXISTS drift_alerts CASCADE")
    op.execute("DROP TABLE IF EXISTS quality_snapshots CASCADE")
    op.execute("DROP TABLE IF EXISTS guardrail_log CASCADE")
    op.execute("DROP TABLE IF EXISTS eval_results CASCADE")
    op.execute("DROP TABLE IF EXISTS eval_queue CASCADE")
    op.execute("DROP TABLE IF EXISTS ab_assignments CASCADE")
    op.execute("DROP TABLE IF EXISTS ab_tests CASCADE")
    op.execute("DROP TABLE IF EXISTS prompt_audit CASCADE")
    op.execute("DROP TABLE IF EXISTS prompts CASCADE")
