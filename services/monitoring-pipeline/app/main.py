"""
Monitoring Pipeline Service

Runs continuously to:
  1. Aggregate eval scores into rolling quality snapshots (TimescaleDB)
  2. Detect quality drift vs baseline (statistical process control)
  3. Fire alerts and trigger auto-rollback when thresholds breached
  4. Expose Prometheus metrics for Grafana dashboards

This is the "flywheel closing" service — without it, you collect eval data
but never act on it. This service is what turns eval infrastructure into
a self-healing AI system.

Statistical approach: Cumulative Sum (CUSUM) chart
  CUSUM is more sensitive to sustained small drifts than single-point alerts.
  It's used in industrial process control and is the right tool for detecting
  LLM quality degradation (which tends to be gradual, not sudden).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import asyncpg
import httpx
import structlog
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from shared.logging_config import configure_logging, CorrelationIDMiddleware
from shared.models.domain import DriftAlert, QualitySnapshot

configure_logging("monitoring-pipeline")
logger = structlog.get_logger("monitoring-pipeline")

DB_DSN               = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/aibackend")
PROMPT_REGISTRY_URL  = os.getenv("PROMPT_REGISTRY_URL", "http://prompt-registry:8000")
SNAPSHOT_INTERVAL_S  = int(os.getenv("SNAPSHOT_INTERVAL_SECONDS", "300"))   # 5 min
DRIFT_WINDOW_HOURS   = int(os.getenv("DRIFT_WINDOW_HOURS", "4"))
DRIFT_WARNING_DROP   = float(os.getenv("DRIFT_WARNING_DROP_PCT", "0.05"))   # 5% drop
DRIFT_CRITICAL_DROP  = float(os.getenv("DRIFT_CRITICAL_DROP_PCT", "0.10"))  # 10% drop
AUTO_ROLLBACK_ENABLED = os.getenv("AUTO_ROLLBACK_ENABLED", "true").lower() == "true"
PROMETHEUS_PORT      = int(os.getenv("PROMETHEUS_PORT", "9090"))


# ---------------------------------------------------------------------------
# Prometheus metrics — these are what Grafana reads
# ---------------------------------------------------------------------------

try:
    start_http_server(PROMETHEUS_PORT)
    logger.info("Prometheus metrics on :%d", PROMETHEUS_PORT)
except OSError:
    pass

# Quality metrics (labelled by prompt_id and version)
quality_score_gauge = Gauge(
    "llm_quality_score",
    "Rolling average quality score for a prompt version",
    ["prompt_id", "version"],
)
hallucination_rate_gauge = Gauge(
    "llm_hallucination_rate",
    "Fraction of responses with hallucination detected",
    ["prompt_id", "version"],
)
toxicity_rate_gauge = Gauge(
    "llm_toxicity_rate",
    "Fraction of responses with toxicity detected",
    ["prompt_id", "version"],
)
pass_rate_gauge = Gauge(
    "llm_eval_pass_rate",
    "Fraction of eval runs that passed all metrics",
    ["prompt_id", "version"],
)

# Guardrail metrics
guardrail_block_counter = Counter(
    "guardrail_blocks_total",
    "Number of requests blocked by guardrails",
    ["phase", "reason"],
)
pii_detection_counter = Counter(
    "guardrail_pii_detections_total",
    "Number of PII entities detected in requests",
    ["phase", "entity_type"],
)
guardrail_latency_histogram = Histogram(
    "guardrail_latency_milliseconds",
    "Guardrail pipeline latency in ms",
    ["phase"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

# Drift and rollback
drift_alert_counter = Counter(
    "llm_drift_alerts_total",
    "Number of drift alerts fired",
    ["prompt_id", "severity"],
)
rollback_counter = Counter(
    "llm_rollbacks_total",
    "Number of automatic rollbacks triggered",
    ["prompt_id"],
)

# A/B test metrics
ab_test_delta_gauge = Gauge(
    "ab_test_score_delta",
    "Score delta between treatment and control in active A/B tests",
    ["test_id", "prompt_id"],
)


# ---------------------------------------------------------------------------
# Quality snapshot computation
# ---------------------------------------------------------------------------

async def compute_quality_snapshot(
    pool: asyncpg.Pool,
    prompt_id: str,
    version: int,
    window_hours: int = 1,
) -> Optional[QualitySnapshot]:
    """
    Aggregate eval results for a prompt version over the last N hours.
    Writes a QualitySnapshot and updates Prometheus gauges.
    """
    window_start = datetime.utcnow() - timedelta(hours=window_hours)

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT overall_score, passed, metrics_json
            FROM eval_results
            WHERE prompt_id=$1 AND prompt_version=$2 AND evaluated_at >= $3
        """, prompt_id, version, window_start)

    if not rows:
        return None

    scores = [r["overall_score"] for r in rows]
    n = len(scores)
    avg_score = sum(scores) / n
    pass_rate = sum(1 for r in rows if r["passed"]) / n

    # Extract per-metric averages
    hallucination_scores, toxicity_scores, relevancy_scores = [], [], []
    for r in rows:
        for m in json.loads(r["metrics_json"]):
            if m["metric_type"] == "hallucination":
                hallucination_scores.append(1.0 - m["score"])  # invert: higher = more hallucination
            elif m["metric_type"] == "toxicity":
                toxicity_scores.append(m["score"])
            elif m["metric_type"] == "answer_relevancy":
                relevancy_scores.append(m["score"])

    def safe_mean(vals: list) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    snapshot = QualitySnapshot(
        prompt_id=prompt_id,
        prompt_version=version,
        window_start=window_start,
        window_end=datetime.utcnow(),
        sample_count=n,
        avg_overall_score=round(avg_score, 4),
        avg_hallucination_score=round(safe_mean(hallucination_scores), 4),
        avg_toxicity_score=round(safe_mean(toxicity_scores), 4),
        avg_relevancy_score=round(safe_mean(relevancy_scores), 4),
        pass_rate=round(pass_rate, 4),
        p50_eval_latency_ms=0.0,   # TODO: compute from latency column
        p99_eval_latency_ms=0.0,
        guardrail_block_rate=0.0,  # TODO: join with guardrail_log table
        pii_detection_rate=0.0,
    )

    # Update Prometheus
    quality_score_gauge.labels(prompt_id=prompt_id, version=str(version)).set(avg_score)
    pass_rate_gauge.labels(prompt_id=prompt_id, version=str(version)).set(pass_rate)
    if hallucination_scores:
        hallucination_rate_gauge.labels(prompt_id=prompt_id, version=str(version)).set(
            safe_mean(hallucination_scores)
        )

    logger.info(
        "Snapshot: prompt=%s v%d n=%d score=%.3f pass_rate=%.1f%%",
        prompt_id, version, n, avg_score, pass_rate * 100,
    )
    return snapshot


# ---------------------------------------------------------------------------
# CUSUM drift detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) control chart for detecting sustained quality drift.

    CUSUM is more sensitive than simple threshold alerts:
    - A single bad score doesn't trigger it
    - A sustained 3% drop over many samples will trigger it
    - Resets when quality recovers

    Parameters:
        k: reference value (allowable slack per sample, ~half the target shift)
        h: decision threshold (how large the cumulative sum can get before alert)
    """
    def __init__(self, target: float = 0.85, k: float = 0.025, h: float = 5.0) -> None:
        self.target = target
        self.k = k
        self.h = h
        self._cusum_neg = 0.0   # tracks downward shifts

    def update(self, score: float) -> float:
        """Update CUSUM with a new score. Returns current CUSUM statistic."""
        # Negative CUSUM detects downward shifts
        self._cusum_neg = min(0.0, self._cusum_neg + score - self.target + self.k)
        return abs(self._cusum_neg)

    @property
    def is_alarmed(self) -> bool:
        return abs(self._cusum_neg) >= self.h

    def reset(self) -> None:
        self._cusum_neg = 0.0


# Per-prompt CUSUM state (in-memory — resets on restart, acceptable)
_cusum_state: dict[str, CUSUMDetector] = {}


# ---------------------------------------------------------------------------
# Drift detection and alerting
# ---------------------------------------------------------------------------

async def check_for_drift(
    pool: asyncpg.Pool,
    prompt_id: str,
    version: int,
    current_score: float,
) -> Optional[DriftAlert]:
    """
    Compare current score against baseline and CUSUM detector.
    Returns a DriftAlert if drift detected, None otherwise.
    """
    # Get baseline (average of first 24h of eval results for this version)
    async with pool.acquire() as conn:
        baseline_row = await conn.fetchrow("""
            SELECT AVG(overall_score) as baseline
            FROM eval_results
            WHERE prompt_id=$1 AND prompt_version=$2
              AND evaluated_at < NOW() - INTERVAL '24 hours'
        """, prompt_id, version)

    baseline = float(baseline_row["baseline"]) if baseline_row and baseline_row["baseline"] else None
    if baseline is None:
        return None   # Not enough history to establish baseline

    drop_pct = (baseline - current_score) / baseline if baseline > 0 else 0.0

    # Update CUSUM
    key = f"{prompt_id}:v{version}"
    if key not in _cusum_state:
        _cusum_state[key] = CUSUMDetector(target=baseline)
    cusum = _cusum_state[key]
    cusum.update(current_score)

    if drop_pct < DRIFT_WARNING_DROP and not cusum.is_alarmed:
        return None

    severity = "critical" if drop_pct >= DRIFT_CRITICAL_DROP or cusum.is_alarmed else "warning"

    alert = DriftAlert(
        prompt_id=prompt_id,
        prompt_version=version,
        metric="overall_score",
        baseline_score=round(baseline, 4),
        current_score=round(current_score, 4),
        drop_pct=round(drop_pct, 4),
        severity=severity,
    )

    drift_alert_counter.labels(prompt_id=prompt_id, severity=severity).inc()
    logger.warning(
        "DRIFT ALERT [%s] prompt=%s v%d baseline=%.3f current=%.3f drop=%.1f%%",
        severity.upper(), prompt_id, version,
        baseline, current_score, drop_pct * 100,
    )

    # Auto-rollback on critical drift
    if severity == "critical" and AUTO_ROLLBACK_ENABLED:
        await trigger_rollback(prompt_id, version, alert.alert_id)
        alert.auto_rollback = True
        cusum.reset()

    return alert


async def trigger_rollback(prompt_id: str, current_version: int, alert_id: str) -> None:
    """
    Trigger automatic rollback to the previous production version.
    Calls the prompt registry to swap versions.
    """
    logger.critical(
        "AUTO-ROLLBACK triggered for prompt=%s v%d alert_id=%s",
        prompt_id, current_version, alert_id,
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{PROMPT_REGISTRY_URL}/prompts/{prompt_id}/rollback",
                json={
                    "from_version": current_version,
                    "reason":       f"Auto-rollback triggered by drift alert {alert_id}",
                    "actor":        "monitoring-pipeline",
                },
            )
        rollback_counter.labels(prompt_id=prompt_id).inc()
    except Exception as e:
        logger.error("Rollback request failed: %s", e)


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------

async def monitoring_loop(pool: asyncpg.Pool) -> None:
    """
    Runs every SNAPSHOT_INTERVAL_S seconds.
    For each active prompt version, computes a quality snapshot and checks drift.
    """
    logger.info(
        "Monitoring loop started (interval=%ds, auto_rollback=%s)",
        SNAPSHOT_INTERVAL_S, AUTO_ROLLBACK_ENABLED,
    )

    while True:
        try:
            async with pool.acquire() as conn:
                active_versions = await conn.fetch("""
                    SELECT DISTINCT prompt_id, prompt_version
                    FROM eval_results
                    WHERE evaluated_at >= NOW() - INTERVAL '24 hours'
                """)

            for row in active_versions:
                pid, ver = row["prompt_id"], row["prompt_version"]
                try:
                    snapshot = await compute_quality_snapshot(pool, pid, ver, window_hours=1)
                    if snapshot:
                        await check_for_drift(pool, pid, ver, snapshot.avg_overall_score)
                except Exception as e:
                    logger.error("Error processing prompt=%s v%d: %s", pid, ver, e)

            # Update A/B test gauges
            async with pool.acquire() as conn:
                ab_rows = await conn.fetch("SELECT * FROM ab_tests WHERE active=true")
            for ab in ab_rows:
                # Fetch mean scores for both variants from recent results
                async with pool.acquire() as conn:
                    c_score = await conn.fetchval("""
                        SELECT AVG(er.overall_score)
                        FROM eval_results er
                        JOIN ab_assignments aa ON er.request_id = aa.request_id
                        WHERE aa.test_id=$1 AND aa.version_assigned=$2
                          AND er.evaluated_at >= NOW() - INTERVAL '4 hours'
                    """, ab["test_id"], ab["control_version"])
                    t_score = await conn.fetchval("""
                        SELECT AVG(er.overall_score)
                        FROM eval_results er
                        JOIN ab_assignments aa ON er.request_id = aa.request_id
                        WHERE aa.test_id=$1 AND aa.version_assigned=$2
                          AND er.evaluated_at >= NOW() - INTERVAL '4 hours'
                    """, ab["test_id"], ab["treatment_version"])

                if c_score and t_score:
                    delta = float(t_score) - float(c_score)
                    ab_test_delta_gauge.labels(
                        test_id=ab["test_id"],
                        prompt_id=ab["prompt_id"],
                    ).set(delta)

        except Exception as e:
            logger.error("Monitoring loop error: %s", e, exc_info=True)

        await asyncio.sleep(SNAPSHOT_INTERVAL_S)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=8)
    app.state.pool = pool
    task = asyncio.create_task(monitoring_loop(pool))
    logger.info("Monitoring pipeline started")
    yield
    task.cancel()
    await pool.close()


app = FastAPI(title="Monitoring Pipeline", version="0.1.0", lifespan=lifespan)
app.add_middleware(CorrelationIDMiddleware)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "monitoring-pipeline",
        "auto_rollback": AUTO_ROLLBACK_ENABLED,
    }

@app.get("/snapshots/{prompt_id}")
async def get_snapshots(prompt_id: str, hours: int = 24) -> list[dict]:
    """Return recent quality snapshots for a prompt."""
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM quality_snapshots
            WHERE prompt_id=$1 AND window_end >= NOW() - INTERVAL '1 hour' * $2
            ORDER BY window_end DESC LIMIT 100
        """, prompt_id, hours)
    return [dict(r) for r in rows]

@app.get("/alerts")
async def get_alerts(resolved: bool = False) -> list[dict]:
    """Return recent drift alerts."""
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM drift_alerts
            WHERE ($1 OR resolved_at IS NULL)
            ORDER BY triggered_at DESC LIMIT 50
        """, resolved)
    return [dict(r) for r in rows]
