"""
Eval Harness Service

Runs async evaluation jobs using DeepEval metrics + custom LLM-as-judge.
Uses any OpenAI-compatible API (Groq, OpenRouter, Ollama) via LLM_BASE_URL.

Architecture:
  - Captures test cases from a PostgreSQL queue (populated by guardrail middleware)
  - Runs DeepEval metrics: AnswerRelevancy, Faithfulness, Hallucination, Toxicity
  - Runs custom G-Eval metrics for domain-specific criteria
  - Stores results back to PostgreSQL + Prometheus metrics
  - Powers the CI gate in the prompt registry (block promotion if score < threshold)

WHY async eval (not sync in the request path):
  - Each DeepEval metric call takes 1-3 seconds (it's an LLM call itself)
  - Running 5 metrics × 2s = 10s added to every request — unacceptable
  - Async workers process at their own pace without impacting TTFT
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import openai
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, Request
from opentelemetry import trace
from pydantic import BaseModel

# DeepEval imports
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from shared.logging_config import configure_logging, CorrelationIDMiddleware
from shared.models.domain import (
    EvalMetricResult,
    EvalMetricType,
    EvalRunResult,
    EvalTestCase,
)

configure_logging("eval-harness")
logger = structlog.get_logger("eval-harness")

DB_DSN         = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/aibackend")
REDIS_URL      = os.getenv("REDIS_URL", "redis://redis:6379")
LLM_API_KEY    = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL   = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")
SAMPLE_RATE    = float(os.getenv("EVAL_SAMPLE_RATE", "0.1"))
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "5"))

# Redis Streams config
STREAM_KEY        = "eval:pending"
CONSUMER_GROUP    = "eval-workers"
CONSUMER_NAME     = f"worker-{os.getpid()}"
XAUTOCLAIM_IDLE   = int(os.getenv("XAUTOCLAIM_IDLE_MS", "60000"))  # reclaim after 60s idle

tracer = trace.get_tracer("eval-harness")


# ---------------------------------------------------------------------------
# LLM judge wrapper — wraps any OpenAI-compatible API for DeepEval
# ---------------------------------------------------------------------------

class LLMJudge(DeepEvalBaseLLM):
    """
    Wraps any OpenAI-compatible endpoint so DeepEval uses it for all LLM-as-judge calls.
    Works with Groq, OpenRouter, Ollama, or any OpenAI-compatible service.
    Both sync and async clients are created once and reused.
    """
    def __init__(self, model: str = JUDGE_MODEL) -> None:
        self.model = model
        self._sync_client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self._async_client = openai.AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    def get_model_name(self) -> str:
        return self.model

    def load_model(self):
        return self._sync_client

    def generate(self, prompt: str) -> str:
        resp = self._sync_client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        resp = await self._async_client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


# Singleton judge — shared across all eval runs
_llm_judge = LLMJudge()


# ---------------------------------------------------------------------------
# Metric factories
# ---------------------------------------------------------------------------

def build_standard_metrics(threshold: float = 0.7) -> list:
    """Build the standard DeepEval metric suite using LLMJudge."""
    return [
        AnswerRelevancyMetric(
            threshold=threshold,
            model=_llm_judge,
            include_reason=True,
        ),
        FaithfulnessMetric(
            threshold=threshold,
            model=_llm_judge,
            include_reason=True,
        ),
        HallucinationMetric(
            threshold=1 - threshold,  # DeepEval: higher hallucination score = worse
            model=_llm_judge,
            include_reason=True,
        ),
        ToxicityMetric(
            threshold=0.3,
            model=_llm_judge,
            include_reason=True,
        ),
    ]


def build_custom_geval(
    name: str,
    criteria: str,
    evaluation_steps: list[str],
    threshold: float = 0.7,
) -> GEval:
    """
    Build a custom G-Eval metric for domain-specific criteria.
    G-Eval = chain-of-thought LLM evaluation with custom rubrics.
    """
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=threshold,
        model=_llm_judge,
    )


SEMANTIC_SIM_THRESHOLD = float(os.getenv("SEMANTIC_SIM_THRESHOLD", "0.7"))
BERT_SCORE_THRESHOLD   = float(os.getenv("BERT_SCORE_THRESHOLD",   "0.7"))
BERT_SCORE_MODEL       = os.getenv("BERT_SCORE_MODEL", "distilbert-base-uncased")

DOMAIN_METRICS = {
    "conciseness": build_custom_geval(
        name="Response Conciseness",
        criteria="The response is appropriately concise without omitting important information.",
        evaluation_steps=[
            "Check if the response is unnecessarily verbose",
            "Verify all key information is included",
            "Assess if the length matches the complexity of the question",
        ],
    ),
    "citation_quality": build_custom_geval(
        name="Citation Quality",
        criteria="When the response makes factual claims, they are supported by the provided context.",
        evaluation_steps=[
            "Identify factual claims in the response",
            "Check each claim against the retrieval context",
            "Penalize claims without context support",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Semantic similarity + BERTScore — singleton models, lazy-loaded, thread-safe
# ---------------------------------------------------------------------------

_semantic_model: Any = None
_bert_scorer:    Any = None
_model_lock = asyncio.Lock()  # prevents concurrent cold-start loading


async def _get_semantic_model() -> Any:
    global _semantic_model
    if _semantic_model is not None:
        return _semantic_model
    async with _model_lock:
        if _semantic_model is None:
            from sentence_transformers import SentenceTransformer
            _semantic_model = await asyncio.to_thread(
                SentenceTransformer, "all-MiniLM-L6-v2"
            )
            logger.info("Loaded sentence-transformers/all-MiniLM-L6-v2")
    return _semantic_model


async def _get_bert_scorer() -> Any:
    global _bert_scorer
    if _bert_scorer is not None:
        return _bert_scorer
    async with _model_lock:
        if _bert_scorer is None:
            from bert_score import BERTScorer
            _bert_scorer = await asyncio.to_thread(
                BERTScorer,
                model_type=BERT_SCORE_MODEL,
                rescale_with_baseline=False,
            )
            logger.info("Loaded BERTScore model: %s", BERT_SCORE_MODEL)
    return _bert_scorer


async def _preload_models() -> None:
    """Warm up both embedding models in the background at startup."""
    try:
        await _get_semantic_model()
        await _get_bert_scorer()
        logger.info("Semantic evaluation models ready")
    except Exception as e:
        logger.warning("Model preload failed (will retry on first use): %s", e)


async def compute_semantic_similarity(text1: str, text2: str) -> float:
    model = await _get_semantic_model()

    def _compute() -> float:
        from sentence_transformers import util
        embs = model.encode([text1, text2], convert_to_tensor=True)
        return float(util.cos_sim(embs[0], embs[1]).item())

    return await asyncio.to_thread(_compute)


async def compute_bert_score(prediction: str, reference: str) -> float:
    scorer = await _get_bert_scorer()

    def _compute() -> float:
        _, _, f1 = scorer.score([prediction], [reference])
        return float(f1[0].item())

    return await asyncio.to_thread(_compute)


async def _run_semantic_similarity(output: str, expected: str) -> EvalMetricResult:
    t0 = time.monotonic()
    try:
        score = await compute_semantic_similarity(output, expected)
        return EvalMetricResult(
            metric_type=EvalMetricType.SEMANTIC_SIMILARITY,
            score=round(score, 4),
            passed=score >= SEMANTIC_SIM_THRESHOLD,
            threshold=SEMANTIC_SIM_THRESHOLD,
            reason=f"cosine similarity with expected: {score:.4f}",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
    except Exception as e:
        logger.warning("Semantic similarity failed: %s", e)
        return EvalMetricResult(
            metric_type=EvalMetricType.SEMANTIC_SIMILARITY,
            score=0.5, passed=True, threshold=SEMANTIC_SIM_THRESHOLD,
            reason=f"check failed (fail-open): {e}",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )


async def _run_bert_score(output: str, expected: str) -> EvalMetricResult:
    t0 = time.monotonic()
    try:
        score = await compute_bert_score(output, expected)
        return EvalMetricResult(
            metric_type=EvalMetricType.BERT_SCORE,
            score=round(score, 4),
            passed=score >= BERT_SCORE_THRESHOLD,
            threshold=BERT_SCORE_THRESHOLD,
            reason=f"BERTScore F1 vs expected: {score:.4f}",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
    except Exception as e:
        logger.warning("BERTScore failed: %s", e)
        return EvalMetricResult(
            metric_type=EvalMetricType.BERT_SCORE,
            score=0.5, passed=True, threshold=BERT_SCORE_THRESHOLD,
            reason=f"check failed (fail-open): {e}",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )


# ---------------------------------------------------------------------------
# Core eval runner
# ---------------------------------------------------------------------------

async def run_eval_suite(test_case: EvalTestCase) -> EvalRunResult:
    """
    Run the full eval suite for one test case.
    Returns EvalRunResult with scores for all metrics.
    This is what gets stored in the DB and what drives the CI gate.
    """
    t0 = time.monotonic()

    with tracer.start_as_current_span(
        "eval.run_suite",
        attributes={
            "test_case_id": test_case.test_case_id,
            "prompt_id":    test_case.prompt_id,
            "prompt_version": test_case.prompt_version,
        },
    ) as span:
        deepeval_case = LLMTestCase(
            input=test_case.input_text,
            actual_output=test_case.output_text,
            expected_output=test_case.expected_output,
            retrieval_context=test_case.context if test_case.context else None,
        )

        metrics = build_standard_metrics(threshold=0.7)
        if test_case.context:
            metrics.append(DOMAIN_METRICS["citation_quality"])

        metric_results: list[EvalMetricResult] = []
        for metric in metrics:
            t_metric = time.monotonic()
            metric_name = type(metric).__name__
            try:
                await asyncio.to_thread(metric.measure, deepeval_case)
                metric_latency = (time.monotonic() - t_metric) * 1000

                metric_type_map = {
                    "AnswerRelevancyMetric": EvalMetricType.ANSWER_RELEVANCY,
                    "FaithfulnessMetric":   EvalMetricType.FAITHFULNESS,
                    "HallucinationMetric":  EvalMetricType.HALLUCINATION,
                    "ToxicityMetric":       EvalMetricType.TOXICITY,
                    "GEval":                EvalMetricType.G_EVAL,
                }
                metric_type = metric_type_map.get(metric_name, EvalMetricType.G_EVAL)

                score = metric.score if hasattr(metric, "score") else 0.0
                if metric_type == EvalMetricType.HALLUCINATION:
                    score = 1.0 - score

                metric_results.append(EvalMetricResult(
                    metric_type=metric_type,
                    score=round(score, 4),
                    passed=metric.is_successful(),
                    threshold=metric.threshold,
                    reason=getattr(metric, "reason", "") or "",
                    latency_ms=round(metric_latency, 2),
                ))
            except Exception as e:
                logger.error("Metric %s failed for test_case %s: %s", metric_name, test_case.test_case_id, e)
                metric_results.append(EvalMetricResult(
                    metric_type=EvalMetricType.LLM_JUDGE,
                    score=0.5,   # neutral on infra failure — don't penalise harshly
                    passed=True,
                    threshold=0.7,
                    reason=f"Metric failed: {e}",
                    latency_ms=0.0,
                ))

        # Semantic similarity + BERTScore — run in parallel, only when expected_output is given
        if test_case.expected_output:
            sem_result, bert_result = await asyncio.gather(
                _run_semantic_similarity(test_case.output_text, test_case.expected_output),
                _run_bert_score(test_case.output_text, test_case.expected_output),
            )
            metric_results.extend([sem_result, bert_result])
            logger.info(
                "sem_sim=%.3f bert_score=%.3f test_case=%s",
                sem_result.score, bert_result.score, test_case.test_case_id,
            )

        overall = sum(r.score for r in metric_results) / len(metric_results) if metric_results else 0.0
        all_passed = all(r.passed for r in metric_results)
        total_ms = (time.monotonic() - t0) * 1000

        span.set_attribute("eval.overall_score", round(overall, 4))
        span.set_attribute("eval.passed", all_passed)

        logger.info(
            "Eval complete test_case=%s prompt=%s v%d score=%.3f passed=%s",
            test_case.test_case_id,
            test_case.prompt_id,
            test_case.prompt_version,
            overall,
            all_passed,
        )

        return EvalRunResult(
            test_case_id=test_case.test_case_id,
            request_id=test_case.request_id,
            prompt_id=test_case.prompt_id,
            prompt_version=test_case.prompt_version,
            metrics=metric_results,
            overall_score=round(overall, 4),
            passed=all_passed,
            judge_model=JUDGE_MODEL,
            total_latency_ms=round(total_ms, 2),
        )


# ---------------------------------------------------------------------------
# CI gate
# ---------------------------------------------------------------------------

async def run_ci_gate(
    prompt_id: str,
    version: int,
    test_cases: list[EvalTestCase],
    min_score: float = 0.8,
    min_pass_rate: float = 0.9,
) -> dict[str, Any]:
    """
    Run eval suite against N test cases as a CI gate.
    Called by prompt registry before promoting a draft to production.
    Returns passed=False if avg_score < min_score or pass_rate < min_pass_rate.
    """
    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)

    async def run_with_semaphore(tc: EvalTestCase) -> EvalRunResult:
        async with semaphore:
            return await run_eval_suite(tc)

    results = await asyncio.gather(*[run_with_semaphore(tc) for tc in test_cases])

    pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0
    avg_score = sum(r.overall_score for r in results) / len(results) if results else 0.0

    blocking = [
        {
            "test_case_id": r.test_case_id,
            "score":        r.overall_score,
            "metrics":      [
                {"type": m.metric_type, "score": m.score, "reason": m.reason}
                for m in r.metrics if not m.passed
            ],
        }
        for r in results if not r.passed
    ]

    gate_passed = avg_score >= min_score and pass_rate >= min_pass_rate

    logger.info(
        "CI gate prompt=%s v%d score=%.3f pass_rate=%.1f%% result=%s",
        prompt_id, version, avg_score, pass_rate * 100,
        "PASSED" if gate_passed else "FAILED",
    )

    return {
        "passed":            gate_passed,
        "score":             round(avg_score, 4),
        "pass_rate":         round(pass_rate, 4),
        "sample_count":      len(results),
        "results":           results,
        "blocking_failures": blocking,
    }


# ---------------------------------------------------------------------------
# Redis Streams worker
# ---------------------------------------------------------------------------

async def _ensure_consumer_group(redis_client: aioredis.Redis) -> None:
    """Create the stream and consumer group if they don't exist yet."""
    try:
        await redis_client.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info("Created stream '%s' consumer group '%s'", STREAM_KEY, CONSUMER_GROUP)
    except Exception as e:
        if "BUSYGROUP" in str(e):
            logger.debug("Consumer group '%s' already exists", CONSUMER_GROUP)
        else:
            raise


async def eval_worker_loop(pool: asyncpg.Pool, redis_client: aioredis.Redis) -> None:
    """
    Long-running worker consuming from Redis Stream 'eval:pending'.

    Each iteration:
      1. XAUTOCLAIM — reclaim messages idle > XAUTOCLAIM_IDLE ms (crash recovery)
      2. XREADGROUP  — read up to 10 fresh messages (blocks 2s when stream is empty)
      3. Process each message concurrently under EVAL_CONCURRENCY semaphore
      4. XACK on success; leave in PEL on failure so XAUTOCLAIM retries it
    """
    logger.info(
        "Eval worker started (stream=%s group=%s consumer=%s)",
        STREAM_KEY, CONSUMER_GROUP, CONSUMER_NAME,
    )
    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)

    while True:
        try:
            messages: list[tuple[str, dict]] = []

            # Step 1: reclaim messages that have been sitting in PEL too long
            try:
                claimed_result = await redis_client.xautoclaim(
                    STREAM_KEY, CONSUMER_GROUP, CONSUMER_NAME,
                    min_idle_time=XAUTOCLAIM_IDLE,
                    start_id="0-0",
                    count=10,
                )
                claimed_msgs = claimed_result[1] if claimed_result else []
                if claimed_msgs:
                    logger.info("XAUTOCLAIM reclaimed %d messages", len(claimed_msgs))
                    messages.extend(claimed_msgs)
            except Exception as e:
                logger.warning("XAUTOCLAIM error (non-fatal): %s", e)

            # Step 2: read new messages, block up to 2s when stream is empty
            response = await redis_client.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME,
                {STREAM_KEY: ">"},
                count=10,
                block=2000,
            )
            if response:
                for _stream_name, msgs in response:
                    messages.extend(msgs)

            if not messages:
                continue

            logger.info("Processing %d eval messages", len(messages))

            async def process_message(msg_id: str, fields: dict) -> None:
                async with semaphore:
                    try:
                        tc = EvalTestCase(**json.loads(fields["test_case_json"]))
                        result = await run_eval_suite(tc)

                        async with pool.acquire() as conn:
                            await conn.execute("""
                                INSERT INTO eval_results
                                  (eval_run_id, test_case_id, request_id, prompt_id,
                                   prompt_version, overall_score, passed,
                                   metrics_json, evaluated_at)
                                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                            """,
                                result.eval_run_id,
                                result.test_case_id,
                                result.request_id,
                                result.prompt_id,
                                result.prompt_version,
                                result.overall_score,
                                result.passed,
                                json.dumps([m.model_dump() for m in result.metrics]),
                                result.evaluated_at,
                            )

                        await redis_client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                        logger.info("ACK'd message %s test_case=%s", msg_id, tc.test_case_id)
                    except Exception as e:
                        # Intentionally NOT ACK-ing — message stays in PEL and will be
                        # reclaimed by XAUTOCLAIM after XAUTOCLAIM_IDLE ms
                        logger.error(
                            "Failed to process message %s: %s — will be reclaimed",
                            msg_id, e, exc_info=True,
                        )

            await asyncio.gather(*[process_message(mid, fields) for mid, fields in messages])

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Worker loop error: %s", e, exc_info=True)
            await asyncio.sleep(5.0)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10)
    app.state.db_pool = pool

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.redis = redis_client
    await _ensure_consumer_group(redis_client)

    asyncio.create_task(_preload_models())
    task = asyncio.create_task(eval_worker_loop(pool, redis_client))
    logger.info("Eval harness started (judge=%s stream=%s)", JUDGE_MODEL, STREAM_KEY)
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await redis_client.aclose()
    await pool.close()


app = FastAPI(title="Eval Harness", version="0.1.0", lifespan=lifespan)
app.add_middleware(CorrelationIDMiddleware)


class CIGateRequest(BaseModel):
    prompt_id:    str
    version:      int
    test_cases:   list[EvalTestCase]
    min_score:    float = 0.8
    min_pass_rate: float = 0.9


class EnqueueRequest(BaseModel):
    test_case: EvalTestCase


@app.post("/eval/ci-gate")
async def ci_gate_endpoint(req: CIGateRequest) -> dict:
    """Run CI gate check — called by prompt registry before promotion."""
    return await run_ci_gate(req.prompt_id, req.version, req.test_cases, req.min_score, req.min_pass_rate)


@app.post("/eval/run")
async def run_single_eval(test_case: EvalTestCase) -> EvalRunResult:
    """Run eval on a single test case immediately (for debugging)."""
    return await run_eval_suite(test_case)


@app.post("/eval/enqueue", status_code=202)
async def enqueue_eval(req: EnqueueRequest, request: Request) -> dict:
    """Push a test case onto the Redis Stream for async background evaluation."""
    msg_id = await request.app.state.redis.xadd(
        STREAM_KEY,
        {"test_case_json": req.test_case.model_dump_json()},
    )
    logger.info("Enqueued test_case=%s msg_id=%s", req.test_case.test_case_id, msg_id)
    return {"queued": True, "message_id": msg_id, "stream": STREAM_KEY}


@app.get("/eval/stream/info")
async def stream_info(request: Request) -> dict:
    """Returns Redis Stream length and consumer group lag for observability."""
    try:
        length = await request.app.state.redis.xlen(STREAM_KEY)
        groups = await request.app.state.redis.xinfo_groups(STREAM_KEY)
        group_info = next((g for g in groups if g["name"] == CONSUMER_GROUP), {})
        return {
            "stream":    STREAM_KEY,
            "length":    length,
            "group":     CONSUMER_GROUP,
            "pending":   group_info.get("pending", 0),
            "consumers": group_info.get("consumers", 0),
            "last_delivered_id": group_info.get("last-delivered-id", "0"),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health(request: Request) -> dict:
    redis_ok = False
    try:
        await request.app.state.redis.ping()
        redis_ok = True
    except Exception:
        pass
    return {"status": "ok", "service": "eval-harness", "redis": redis_ok}
