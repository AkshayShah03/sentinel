"""
Prompt Registry Service

The source of truth for all prompt versions. Every LLM call in your system
should fetch its system prompt from here — never hardcode prompts.

Features:
  1. Versioned prompt store (immutable history, bump version on any change)
  2. CI gate (eval harness must pass before promotion to production)
  3. A/B testing (traffic split between control and treatment versions)
  4. Automatic rollback (triggered by monitoring pipeline on quality drop)
  5. Audit trail (who changed what, when, and why)

WHY this matters in interviews:
  "Prompt drifting" — prompts change over time without tracking — is the #1
  source of silent quality regressions in production AI systems. This service
  prevents that by making prompts first-class versioned artifacts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import asyncpg
import httpx
import structlog
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from shared.logging_config import configure_logging, CorrelationIDMiddleware
from shared.models.domain import (
    ABTestConfig,
    ABTestAssignment,
    PromotionStatus,
    PromptTemplate,
)

configure_logging("prompt-registry")
logger = structlog.get_logger("prompt-registry")

DB_DSN           = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/aibackend")
EVAL_HARNESS_URL = os.getenv("EVAL_HARNESS_URL", "http://eval-harness:8000")


def _row(record: asyncpg.Record) -> dict:
    """
    Convert asyncpg Record to dict safe for Pydantic v2:
    - UUID objects → str (Pydantic v2 won't coerce automatically)
    - JSONB strings → Python objects (asyncpg doesn't auto-decode JSONB by default)
    """
    import uuid as _uuid
    result = {}
    for k, v in dict(record).items():
        if isinstance(v, _uuid.UUID):
            result[k] = str(v)
        elif isinstance(v, str) and len(v) > 0 and v[0] in ('{', '['):
            try:
                result[k] = json.loads(v)
            except (ValueError, TypeError):
                result[k] = v
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def get_pool() -> asyncpg.Pool:
    async def _init(conn):
        await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
        await conn.set_type_codec("json",  encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    return await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10, init=_init)


# ---------------------------------------------------------------------------
# Prompt CRUD
# ---------------------------------------------------------------------------

async def create_prompt(
    pool: asyncpg.Pool,
    name: str,
    content: str,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    tags: list[str] | None = None,
    created_by: str = "system",
) -> PromptTemplate:
    """
    Create a new prompt. Version is always 1 for a new name.
    For updates to an existing name, use update_prompt() which bumps the version.
    """
    async with pool.acquire() as conn:
        existing = await conn.fetchval(
            "SELECT MAX(version) FROM prompts WHERE name = $1", name
        )
        version = 1 if existing is None else existing + 1

        prompt = PromptTemplate(
            name=name,
            version=version,
            content=content,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tags=tags or [],
            created_by=created_by,
            status=PromotionStatus.DRAFT,
        )

        await conn.execute("""
            INSERT INTO prompts
              (prompt_id, name, version, content, model, temperature,
               max_tokens, status, tags, min_eval_score, created_by,
               created_at, metadata)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
        """,
            prompt.prompt_id,
            prompt.name,
            prompt.version,
            prompt.content,
            prompt.model,
            prompt.temperature,
            prompt.max_tokens,
            prompt.status.value,
            json.dumps(prompt.tags),
            prompt.min_eval_score,
            prompt.created_by,
            prompt.created_at,
            json.dumps(prompt.metadata),
        )

        logger.info("Created prompt name=%s version=%d id=%s", name, version, prompt.prompt_id)
        return prompt


async def get_production_prompt(pool: asyncpg.Pool, name: str) -> Optional[PromptTemplate]:
    """Get the current production version of a named prompt."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM prompts WHERE name=$1 AND status='production' ORDER BY version DESC LIMIT 1",
            name,
        )
    return PromptTemplate(**_row(row)) if row else None


async def get_prompt_by_id(pool: asyncpg.Pool, prompt_id: str) -> Optional[PromptTemplate]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM prompts WHERE prompt_id=$1", prompt_id)
    return PromptTemplate(**_row(row)) if row else None


# ---------------------------------------------------------------------------
# CI Gate integration
# ---------------------------------------------------------------------------

async def promote_prompt(
    pool: asyncpg.Pool,
    prompt_id: str,
    test_cases: list[dict],
    promoted_by: str = "system",
) -> dict:
    """
    Promote a prompt from DRAFT → PRODUCTION by running the CI gate.

    Process:
    1. Call eval harness with the test cases
    2. If gate passes → set status=production, demote previous production
    3. If gate fails → keep status=draft, return failure details
    4. Write audit log entry either way
    """
    prompt = await get_prompt_by_id(pool, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    if prompt.status not in (PromotionStatus.DRAFT, PromotionStatus.TESTING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot promote prompt in status {prompt.status}",
        )

    # Run CI gate
    logger.info("Running CI gate for prompt=%s version=%d", prompt.name, prompt.version)

    from shared.models.domain import EvalTestCase
    eval_test_cases = [EvalTestCase(**tc) for tc in test_cases]

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{EVAL_HARNESS_URL}/eval/ci-gate",
            json={
                "prompt_id":     prompt.prompt_id,
                "version":       prompt.version,
                "test_cases":    [tc.model_dump(mode="json") for tc in eval_test_cases],
                "min_score":     prompt.min_eval_score,
                "min_pass_rate": 0.85,
            },
        )

    gate_result = resp.json()

    async with pool.acquire() as conn:
        if gate_result["passed"]:
            # Demote current production version
            await conn.execute(
                "UPDATE prompts SET status='deprecated' WHERE name=$1 AND status='production'",
                prompt.name,
            )
            # Promote this version
            await conn.execute(
                "UPDATE prompts SET status='production', promoted_at=NOW() WHERE prompt_id=$1",
                prompt_id,
            )
            logger.info(
                "Promoted prompt=%s version=%d score=%.3f",
                prompt.name, prompt.version, gate_result["score"],
            )
        else:
            logger.warning(
                "CI gate FAILED for prompt=%s version=%d score=%.3f",
                prompt.name, prompt.version, gate_result["score"],
            )

        # Audit entry
        await conn.execute("""
            INSERT INTO prompt_audit
              (prompt_id, action, actor, metadata, created_at)
            VALUES ($1, $2, $3, $4, NOW())
        """,
            prompt_id,
            "promote_success" if gate_result["passed"] else "promote_failed",
            promoted_by,
            json.dumps({
                "score":      gate_result["score"],
                "pass_rate":  gate_result["pass_rate"],
                "failures":   len(gate_result.get("blocking_failures", [])),
            }),
        )

    return gate_result


# ---------------------------------------------------------------------------
# A/B testing
# ---------------------------------------------------------------------------

async def create_ab_test(
    pool: asyncpg.Pool,
    prompt_id: str,  # base name
    control_version: int,
    treatment_version: int,
    traffic_split: float = 0.5,
) -> ABTestConfig:
    """Create a new A/B test between two versions of a prompt."""
    config = ABTestConfig(
        prompt_id=prompt_id,
        control_version=control_version,
        treatment_version=treatment_version,
        traffic_split=traffic_split,
    )

    async with pool.acquire() as conn:
        # Mark treatment version as TESTING
        await conn.execute(
            "UPDATE prompts SET status='testing' WHERE name=$1 AND version=$2",
            prompt_id, treatment_version,
        )
        await conn.execute("""
            INSERT INTO ab_tests
              (test_id, prompt_id, control_version, treatment_version,
               traffic_split, active, started_at)
            VALUES ($1,$2,$3,$4,$5,true,NOW())
        """,
            config.test_id,
            config.prompt_id,
            config.control_version,
            config.treatment_version,
            config.traffic_split,
        )

    logger.info(
        "A/B test started: %s — control=v%d treatment=v%d split=%.0f%%",
        prompt_id, control_version, treatment_version, traffic_split * 100,
    )
    return config


async def resolve_prompt_for_request(
    pool: asyncpg.Pool,
    prompt_name: str,
    request_id: str,
) -> tuple[PromptTemplate, Optional[str]]:
    """
    Resolve which prompt version to use for a request.
    Returns (PromptTemplate, test_id_if_in_ab_test).

    Assignment is deterministic per request_id within a test —
    same request always gets same version (important for reproducibility).
    """
    async with pool.acquire() as conn:
        # Check for active A/B test
        ab_row = await conn.fetchrow(
            "SELECT * FROM ab_tests WHERE prompt_id=$1 AND active=true LIMIT 1",
            prompt_name,
        )

        if ab_row:
            # Deterministic assignment by hashing request_id
            h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            use_treatment = (h % 1000) < int(ab_row["traffic_split"] * 1000)
            version = ab_row["treatment_version"] if use_treatment else ab_row["control_version"]

            # Record assignment for analysis
            await conn.execute("""
                INSERT INTO ab_assignments
                  (request_id, test_id, version_assigned, assigned_at)
                VALUES ($1,$2,$3,NOW())
                ON CONFLICT (request_id, test_id) DO NOTHING
            """, request_id, ab_row["test_id"], version)

            row = await conn.fetchrow(
                "SELECT * FROM prompts WHERE name=$1 AND version=$2", prompt_name, version
            )
            if row:
                return PromptTemplate(**_row(row)), str(ab_row["test_id"])

        # No A/B test — use production version
        row = await conn.fetchrow(
            "SELECT * FROM prompts WHERE name=$1 AND status='production' ORDER BY version DESC LIMIT 1",
            prompt_name,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No production prompt found for '{prompt_name}'",
        )
    return PromptTemplate(**_row(row)), None


async def analyze_ab_test(pool: asyncpg.Pool, test_id: str) -> dict:
    """
    Statistical analysis of an A/B test.
    Computes mean scores per variant and runs a two-sample t-test.
    """
    import math

    async with pool.acquire() as conn:
        test_row = await conn.fetchrow("SELECT * FROM ab_tests WHERE test_id=$1", test_id)
        if not test_row:
            raise HTTPException(status_code=404, detail="A/B test not found")

        control_scores = await conn.fetch("""
            SELECT er.overall_score FROM eval_results er
            JOIN ab_assignments aa ON er.request_id = aa.request_id
            WHERE aa.test_id=$1 AND aa.version_assigned=$2
        """, test_id, test_row["control_version"])

        treatment_scores = await conn.fetch("""
            SELECT er.overall_score FROM eval_results er
            JOIN ab_assignments aa ON er.request_id = aa.request_id
            WHERE aa.test_id=$1 AND aa.version_assigned=$2
        """, test_id, test_row["treatment_version"])

    c_vals = [r["overall_score"] for r in control_scores]
    t_vals = [r["overall_score"] for r in treatment_scores]

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": 0, "std": 0, "n": 0}
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(n - 1, 1))
        return {"mean": round(mean, 4), "std": round(std, 4), "n": n}

    c_stats = stats(c_vals)
    t_stats = stats(t_vals)

    # Welch's t-test (unequal variance)
    significant = False
    delta = t_stats["mean"] - c_stats["mean"]
    if c_stats["n"] >= 30 and t_stats["n"] >= 30:
        se = math.sqrt(
            (c_stats["std"] ** 2 / c_stats["n"]) +
            (t_stats["std"] ** 2 / t_stats["n"])
        )
        t_stat = abs(delta / se) if se > 0 else 0
        significant = t_stat > 2.0  # ~p<0.05 for large n

    return {
        "test_id":           test_id,
        "control":           c_stats,
        "treatment":         t_stats,
        "delta":             round(delta, 4),
        "significant":       significant,
        "recommendation":    (
            "promote treatment" if significant and delta > 0
            else "keep control"   if significant and delta < 0
            else "more data needed"
        ),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_pool()
    app.state.pool = pool
    logger.info("Prompt registry started")
    yield
    await pool.close()


app = FastAPI(title="Prompt Registry", version="0.1.0", lifespan=lifespan)
app.add_middleware(CorrelationIDMiddleware)


class CreatePromptRequest(BaseModel):
    name: str
    content: str
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 2048
    tags: list[str] = []
    created_by: str = "system"

class PromoteRequest(BaseModel):
    test_cases: list[dict]
    promoted_by: str = "system"

class ABTestRequest(BaseModel):
    control_version: int
    treatment_version: int
    traffic_split: float = 0.5

class ResolveRequest(BaseModel):
    prompt_name: str
    request_id: str


@app.post("/prompts", response_model=PromptTemplate, status_code=201)
async def create(req: CreatePromptRequest, request: Request) -> PromptTemplate:
    return await create_prompt(request.app.state.pool, **req.model_dump())


@app.post("/prompts/{prompt_id}/promote")
async def promote(prompt_id: str, req: PromoteRequest, request: Request) -> dict:
    return await promote_prompt(request.app.state.pool, prompt_id, req.test_cases, req.promoted_by)


@app.post("/prompts/{name}/ab-test")
async def start_ab_test(name: str, req: ABTestRequest, request: Request) -> ABTestConfig:
    return await create_ab_test(request.app.state.pool, name, req.control_version, req.treatment_version, req.traffic_split)


@app.get("/prompts/{name}/ab-test/{test_id}/analysis")
async def ab_analysis(name: str, test_id: str, request: Request) -> dict:
    return await analyze_ab_test(request.app.state.pool, test_id)


@app.post("/prompts/resolve")
async def resolve(req: ResolveRequest, request: Request) -> dict:
    prompt, test_id = await resolve_prompt_for_request(request.app.state.pool, req.prompt_name, req.request_id)
    return {"prompt": prompt.model_dump(), "ab_test_id": test_id}


@app.get("/prompts/{name}/production")
async def get_production(name: str, request: Request) -> PromptTemplate:
    prompt = await get_production_prompt(request.app.state.pool, name)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"No production prompt for {name!r}")
    return prompt


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "prompt-registry"}
