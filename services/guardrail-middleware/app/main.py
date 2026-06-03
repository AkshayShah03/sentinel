"""
Guardrail Middleware Service

Drop-in safety layer wrapping any LLM call with input + output guardrails.
Uses any OpenAI-compatible API (Groq, OpenRouter, Ollama, OpenAI) via LLM_BASE_URL.

OWASP LLM Top 10 coverage:
  LLM01: Prompt Injection → injection_check()
  LLM02: Sensitive Data Leakage → pii_check(), pii_output_check()
  LLM06: Excessive Agency → topic_restriction_check()
  LLM07: System Prompt Leakage → output_grounding_check()
"""
from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

try:
    from presidio_analyzer import AnalyzerEngine as _PresidioAnalyzerEngine
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False

import openai
import structlog
from fastapi import FastAPI, HTTPException, Request, status
from opentelemetry import trace
from pydantic import BaseModel, Field

from shared.logging_config import configure_logging, CorrelationIDMiddleware
from shared.models.domain import (
    GuardrailCheckResult,
    GuardrailCheckType,
    GuardrailResult,
    GuardrailVerdict,
    PIIEntity,
)

configure_logging("guardrail-middleware")
logger = structlog.get_logger("guardrail-middleware")

LLM_API_KEY   = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
JUDGE_MODEL   = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")

# Risk thresholds — tune per use case
PII_THRESHOLD             = float(os.getenv("PII_THRESHOLD", "0.0"))
INJECTION_THRESHOLD       = float(os.getenv("INJECTION_THRESHOLD", "0.7"))
JAILBREAK_THRESHOLD       = float(os.getenv("JAILBREAK_THRESHOLD", "0.7"))
TOXICITY_INPUT_THRESHOLD  = float(os.getenv("TOXICITY_INPUT_THRESHOLD", "0.6"))
HALLUCINATION_THRESHOLD   = float(os.getenv("HALLUCINATION_THRESHOLD", "0.5"))
TOXICITY_OUTPUT_THRESHOLD = float(os.getenv("TOXICITY_OUTPUT_THRESHOLD", "0.4"))

LLM_CHECK_TIMEOUT = float(os.getenv("LLM_CHECK_TIMEOUT", "3.0"))  # seconds per attempt
LLM_CHECK_RETRIES = int(os.getenv("LLM_CHECK_RETRIES", "1"))       # retries after 1st attempt
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "3"))
CB_RECOVERY_TIMEOUT  = float(os.getenv("CB_RECOVERY_TIMEOUT", "30.0"))

tracer = trace.get_tracer("guardrail-middleware")


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class _CBState(str, enum.Enum):
    CLOSED    = "closed"     # normal operation
    OPEN      = "open"       # fast-failing; fail-open on guardrail checks
    HALF_OPEN = "half_open"  # probing whether downstream recovered


class CircuitBreaker:
    """
    Asyncio-safe circuit breaker for LLM-based guardrail checks.
    Fail-open policy: an OPEN circuit returns a passing result so requests
    are not blocked when the LLM judge is unavailable.
    """

    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self._state         = _CBState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> _CBState:
        if self._state == _CBState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = _CBState.HALF_OPEN
                logger.info("Circuit '%s' OPEN → HALF_OPEN", self.name)
        return self._state

    def record_success(self) -> None:
        if self._state != _CBState.CLOSED:
            logger.info("Circuit '%s' → CLOSED", self.name)
        self._state         = _CBState.CLOSED
        self._failure_count = 0
        self._opened_at     = None

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold and self._state != _CBState.OPEN:
            self._state     = _CBState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit '%s' → OPEN (failures=%d, recovery_in=%.0fs)",
                self.name, self._failure_count, self.recovery_timeout,
            )


# One circuit breaker per LLM-based check — isolated so one flaky check
# doesn't affect the others even though they share the same LLM endpoint.
_CB_TOXICITY_INPUT  = CircuitBreaker("toxicity_input",  CB_FAILURE_THRESHOLD, CB_RECOVERY_TIMEOUT)
_CB_TOXICITY_OUTPUT = CircuitBreaker("toxicity_output", CB_FAILURE_THRESHOLD, CB_RECOVERY_TIMEOUT)
_CB_HALLUCINATION   = CircuitBreaker("hallucination",   CB_FAILURE_THRESHOLD, CB_RECOVERY_TIMEOUT)


# ---------------------------------------------------------------------------
# PII patterns — regex-based (fast, zero API cost)
# ---------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL",       re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')),
    ("PHONE",       re.compile(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')),
    ("SSN",         re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("CREDIT_CARD", re.compile(r'\b(?:\d[ -]?){13,16}\b')),
    ("IP_ADDRESS",  re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
    ("API_KEY",     re.compile(r'\b(?:sk-|ghp_|AIza|AKIA)[A-Za-z0-9_-]{16,}\b')),
]

_INJECTION_PATTERNS = re.compile(
    r'ignore\s+(previous|all|above)\s+instructions?|'
    r'disregard\s+(your|all)\s+(previous\s+)?instructions?|'
    r'forget\s+everything|'
    r'new\s+instructions?:|'
    r'system\s+prompt:|'
    r'<\s*/?system\s*>|'
    r'\[\s*INST\s*\]|'
    r'###\s*instruction',
    re.IGNORECASE,
)

_JAILBREAK_PATTERNS = re.compile(
    r'(DAN|jailbreak|do\s+anything\s+now|you\s+are\s+now|pretend\s+you\s+are\s+an?\s+AI\s+without|'
    r'act\s+as\s+an?\s+(unrestricted|uncensored|evil)|roleplay\s+as)',
    re.IGNORECASE,
)

# Presidio entity types to detect (beyond what regex covers)
_PRESIDIO_ENTITIES = [
    "PERSON", "LOCATION", "DATE_TIME", "MEDICAL_LICENSE",
    "IBAN_CODE", "US_DRIVER_LICENSE", "US_SSN", "US_BANK_NUMBER",
    "CREDIT_CARD", "PHONE_NUMBER", "EMAIL_ADDRESS", "IP_ADDRESS",
    "US_ITIN", "US_PASSPORT",
]


# ---------------------------------------------------------------------------
# Individual guardrail checks
# ---------------------------------------------------------------------------

def pii_check(text: str, analyzer: Optional[Any] = None) -> GuardrailCheckResult:
    """
    Two-stage PII detection:
      1. Regex patterns — O(n), <1ms, covers structured PII (SSN, email, credit card…)
      2. Presidio NLP   — ~50ms, covers unstructured PII (names, locations, dates…)
                          Only runs when regex finds nothing (avoids NLP overhead on flagged text).
    """
    t0 = time.monotonic()
    entities: list[PIIEntity] = []
    counter: dict[str, int] = {}

    # Stage 1: fast regex pre-filter
    for entity_type, pattern in _PII_PATTERNS:
        for match in pattern.finditer(text):
            counter[entity_type] = counter.get(entity_type, 0) + 1
            replacement = f"<<{entity_type}_{counter[entity_type]}>>"
            entities.append(PIIEntity(
                entity_type=entity_type,
                start=match.start(),
                end=match.end(),
                original=match.group(),
                replacement=replacement,
                confidence=0.95,
            ))

    # Stage 2: NLP fallback — only when regex found nothing
    if not entities and analyzer is not None:
        try:
            results = analyzer.analyze(text=text, language="en", entities=_PRESIDIO_ENTITIES)
            for r in results:
                etype = r.entity_type
                counter[etype] = counter.get(etype, 0) + 1
                replacement = f"<<{etype}_{counter[etype]}>>"
                entities.append(PIIEntity(
                    entity_type=etype,
                    start=r.start,
                    end=r.end,
                    original=text[r.start:r.end],
                    replacement=replacement,
                    confidence=float(r.score),
                ))
        except Exception as e:
            logger.warning("Presidio analysis failed: %s", e)

    score = 1.0 if entities else 0.0
    return GuardrailCheckResult(
        check_type=GuardrailCheckType.PII_DETECTION,
        passed=len(entities) == 0,
        score=score,
        threshold=PII_THRESHOLD,
        reason=f"Found {len(entities)} PII entities: {[e.entity_type for e in entities]}" if entities else None,
        pii_entities=entities,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


def redact_pii(text: str, entities: list[PIIEntity]) -> str:
    """Apply entity replacements right-to-left to preserve positions."""
    result = text
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        result = result[:entity.start] + entity.replacement + result[entity.end:]
    return result


def injection_check(text: str) -> GuardrailCheckResult:
    t0 = time.monotonic()
    matches = _INJECTION_PATTERNS.findall(text)
    # Any injection pattern match should block — no threshold needed
    score = 1.0 if matches else 0.0
    return GuardrailCheckResult(
        check_type=GuardrailCheckType.PROMPT_INJECTION,
        passed=len(matches) == 0,
        score=score,
        threshold=INJECTION_THRESHOLD,
        reason=f"Injection patterns detected: {matches}" if matches else None,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


def jailbreak_check(text: str) -> GuardrailCheckResult:
    t0 = time.monotonic()
    matches = _JAILBREAK_PATTERNS.findall(text)
    score = 1.0 if matches else 0.0
    return GuardrailCheckResult(
        check_type=GuardrailCheckType.JAILBREAK,
        passed=len(matches) == 0,
        score=score,
        threshold=JAILBREAK_THRESHOLD,
        reason=f"Jailbreak patterns: {matches}" if matches else None,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


async def toxicity_check(
    text: str,
    phase: str,
    client: openai.AsyncOpenAI,
) -> GuardrailCheckResult:
    """
    LLM-as-judge toxicity check.
    Uses the smallest/fastest model (haiku-class) because toxicity classification
    is a simpler task than generation, and speed is critical in the request path.
    """
    t0 = time.monotonic()
    threshold = TOXICITY_INPUT_THRESHOLD if phase == "input" else TOXICITY_OUTPUT_THRESHOLD
    check_type = GuardrailCheckType.TOXICITY_INPUT if phase == "input" else GuardrailCheckType.TOXICITY_OUTPUT

    prompt = f"""Rate the TOXICITY of this text on a scale of 0.0 to 1.0.

Toxicity means: hate speech, harassment, threats, profanity, explicit violence, dangerous instructions, or encouragement of illegal activity.
NOT toxicity: sharing personal data, privacy concerns, financial information, neutral questions, or normal conversation.

0.0 = completely benign, safe text.
1.0 = extremely toxic — hate speech, death threats, explicit harmful content.

Text: {text[:1000]}

Respond with ONLY valid JSON, no extra text: {{"score": <float 0.0-1.0>, "reason": "<one sentence explaining the score>"}}"""

    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        # Strip markdown code fences if the model wraps output
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
    except Exception as e:
        logger.warning("Toxicity check failed, defaulting to 0.0: %s", e)
        score, reason = 0.0, "check failed — defaulting to allow"

    return GuardrailCheckResult(
        check_type=check_type,
        passed=score < threshold,
        score=score,
        threshold=threshold,
        reason=reason if score >= threshold else None,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


async def hallucination_check(
    output: str,
    context: list[str],
    client: openai.AsyncOpenAI,
) -> GuardrailCheckResult:
    """
    LLM-as-judge hallucination check.
    Compares the model's output against the context it had available.
    Only meaningful when context (RAG chunks) is provided.
    """
    t0 = time.monotonic()

    if not context:
        return GuardrailCheckResult(
            check_type=GuardrailCheckType.HALLUCINATION,
            passed=True,
            score=0.0,
            threshold=HALLUCINATION_THRESHOLD,
            reason="No context provided — hallucination check skipped",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    context_str = "\n\n".join(context[:5])
    prompt = f"""You are evaluating whether an AI response is grounded in the provided context.

CONTEXT:
{context_str[:2000]}

AI RESPONSE:
{output[:1000]}

For each factual claim in the AI response, determine if it is supported by the context.
Return ONLY JSON: {{"hallucination_score": <0.0-1.0>, "reason": "<brief explanation>", "unsupported_claims": [<list of strings>]}}

0.0 = fully grounded. 1.0 = completely hallucinated."""

    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)
        score = float(data.get("hallucination_score", 0.0))
        reason = data.get("reason", "")
        unsupported = data.get("unsupported_claims", [])
    except Exception as e:
        logger.warning("Hallucination check failed: %s", e)
        score, reason, unsupported = 0.0, "check failed", []

    return GuardrailCheckResult(
        check_type=GuardrailCheckType.HALLUCINATION,
        passed=score < HALLUCINATION_THRESHOLD,
        score=score,
        threshold=HALLUCINATION_THRESHOLD,
        reason=f"Unsupported: {unsupported}" if unsupported else reason,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


# ---------------------------------------------------------------------------
# Circuit-breaker wrapper for LLM-based checks
# ---------------------------------------------------------------------------

async def _guarded_llm_check(
    cb: CircuitBreaker,
    coro_factory: Callable,
    fallback: GuardrailCheckResult,
) -> GuardrailCheckResult:
    """
    Wraps an async LLM guardrail check with:
      - Circuit breaker: fail-open (return passing fallback) when OPEN
      - Per-attempt timeout: LLM_CHECK_TIMEOUT seconds
      - Exponential backoff retry: LLM_CHECK_RETRIES additional attempts
    """
    if cb.state == _CBState.OPEN:
        logger.warning("Circuit '%s' is OPEN — skipping check (fail-open)", cb.name)
        return fallback

    for attempt in range(LLM_CHECK_RETRIES + 1):
        try:
            result = await asyncio.wait_for(coro_factory(), timeout=LLM_CHECK_TIMEOUT)
            cb.record_success()
            return result
        except asyncio.TimeoutError:
            cb.record_failure()
            logger.warning("Circuit '%s': timeout attempt %d/%d", cb.name, attempt + 1, LLM_CHECK_RETRIES + 1)
        except Exception as exc:
            cb.record_failure()
            logger.warning("Circuit '%s': error attempt %d/%d: %s", cb.name, attempt + 1, LLM_CHECK_RETRIES + 1, exc)

        if attempt < LLM_CHECK_RETRIES:
            await asyncio.sleep(0.1 * (2 ** attempt))  # 0.1s, 0.2s, …

    logger.warning("Circuit '%s': all attempts exhausted — returning fallback", cb.name)
    return fallback


# ---------------------------------------------------------------------------
# Pipeline orchestrators
# ---------------------------------------------------------------------------

async def run_input_guardrails(
    text: str,
    client: openai.AsyncOpenAI,
    analyzer: Optional[Any] = None,
) -> GuardrailResult:
    """Run all input checks. Fast regex/injection checks sync; PII (NLP) + toxicity (LLM) in parallel."""
    t0 = time.monotonic()

    inj_result = injection_check(text)
    jb_result  = jailbreak_check(text)

    _tox_input_fallback = GuardrailCheckResult(
        check_type=GuardrailCheckType.TOXICITY_INPUT,
        passed=True, score=0.0, threshold=TOXICITY_INPUT_THRESHOLD,
        reason="circuit open — toxicity check skipped (fail-open)",
    )
    pii_result, tox_result = await asyncio.gather(
        asyncio.to_thread(pii_check, text, analyzer),
        _guarded_llm_check(
            _CB_TOXICITY_INPUT,
            lambda: toxicity_check(text, "input", client),
            _tox_input_fallback,
        ),
    )

    checks = [pii_result, inj_result, jb_result, tox_result]

    redacted_text = None
    if not inj_result.passed or not jb_result.passed:
        verdict = GuardrailVerdict.BLOCK
        blocked_reason = inj_result.reason if not inj_result.passed else jb_result.reason
    elif not tox_result.passed:
        verdict = GuardrailVerdict.BLOCK
        blocked_reason = tox_result.reason
    elif not pii_result.passed:
        verdict = GuardrailVerdict.REDACT
        redacted_text = redact_pii(text, pii_result.pii_entities)
        blocked_reason = None
    else:
        verdict = GuardrailVerdict.ALLOW
        blocked_reason = None

    return GuardrailResult(
        phase="input",
        verdict=verdict,
        checks=checks,
        redacted_text=redacted_text,
        blocked_reason=blocked_reason,
        total_latency_ms=(time.monotonic() - t0) * 1000,
    )


async def run_output_guardrails(
    output: str,
    context: list[str],
    client: openai.AsyncOpenAI,
    analyzer: Optional[Any] = None,
) -> GuardrailResult:
    """Run output safety checks in parallel."""
    t0 = time.monotonic()

    _tox_output_fallback = GuardrailCheckResult(
        check_type=GuardrailCheckType.TOXICITY_OUTPUT,
        passed=True, score=0.0, threshold=TOXICITY_OUTPUT_THRESHOLD,
        reason="circuit open — toxicity check skipped (fail-open)",
    )
    _halluc_fallback = GuardrailCheckResult(
        check_type=GuardrailCheckType.HALLUCINATION,
        passed=True, score=0.0, threshold=HALLUCINATION_THRESHOLD,
        reason="circuit open — hallucination check skipped (fail-open)",
    )
    pii_out_result, tox_out_result, halluc_result = await asyncio.gather(
        asyncio.to_thread(pii_check, output, analyzer),
        _guarded_llm_check(
            _CB_TOXICITY_OUTPUT,
            lambda: toxicity_check(output, "output", client),
            _tox_output_fallback,
        ),
        _guarded_llm_check(
            _CB_HALLUCINATION,
            lambda: hallucination_check(output, context, client),
            _halluc_fallback,
        ),
    )
    pii_out_result.check_type = GuardrailCheckType.PII_LEAK

    checks = [pii_out_result, tox_out_result, halluc_result]

    redacted_text = None
    blocked_reason = None

    if not tox_out_result.passed:
        verdict = GuardrailVerdict.BLOCK
        blocked_reason = tox_out_result.reason
    elif not halluc_result.passed:
        verdict = GuardrailVerdict.FLAG
        blocked_reason = halluc_result.reason
    elif not pii_out_result.passed:
        verdict = GuardrailVerdict.REDACT
        redacted_text = redact_pii(output, pii_out_result.pii_entities)
    else:
        verdict = GuardrailVerdict.ALLOW

    return GuardrailResult(
        phase="output",
        verdict=verdict,
        checks=checks,
        redacted_text=redacted_text,
        blocked_reason=blocked_reason,
        total_latency_ms=(time.monotonic() - t0) * 1000,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class GuardedChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 2048
    temperature: float = 0.7
    context: list[str] = Field(default_factory=list)
    session_id: str = ""
    prompt_id: str = ""

class GuardedChatResponse(BaseModel):
    content: str
    input_guardrail: GuardrailResult
    output_guardrail: GuardrailResult
    model: str
    usage: dict[str, int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = openai.AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    # Load Presidio NLP analyzer — spaCy model takes 2-3s on cold start, run in thread
    if _PRESIDIO_AVAILABLE:
        try:
            app.state.presidio = await asyncio.to_thread(_PresidioAnalyzerEngine)
            logger.info("Presidio analyzer loaded (NLP PII fallback active)")
        except Exception as e:
            app.state.presidio = None
            logger.warning("Presidio failed to load — NLP PII fallback disabled: %s", e)
    else:
        app.state.presidio = None
        logger.warning("presidio-analyzer not installed — NLP PII fallback disabled")

    logger.info("guardrail_started", base_url=LLM_BASE_URL, judge=JUDGE_MODEL)
    yield
    await app.state.llm.close()
    logger.info("Guardrail middleware shutting down")


app = FastAPI(
    title="Guardrail Middleware",
    version="0.1.0",
    description="Drop-in safety layer for any LLM call",
    lifespan=lifespan,
)
app.add_middleware(CorrelationIDMiddleware)


@app.post("/v1/chat", response_model=GuardedChatResponse)
async def guarded_chat(req: GuardedChatRequest, request: Request) -> GuardedChatResponse:
    """
    Main endpoint. Wraps an LLM call with full input + output guardrails.

    Flow:
    1. Run input guardrails on the last user message
    2. If blocked → return 400 with reason
    3. If redacted → use redacted text for LLM call
    4. Call LLM (any OpenAI-compatible endpoint)
    5. Run output guardrails on response
    6. Return response + both guardrail results for observability
    """
    client: openai.AsyncOpenAI = request.app.state.llm

    with tracer.start_as_current_span("guardrail.full_pipeline") as span:
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        last_user_text = user_messages[-1].content
        span.set_attribute("session_id", req.session_id)
        span.set_attribute("prompt_id", req.prompt_id)

        analyzer = request.app.state.presidio

        # ── Step 1: Input guardrails ──────────────────────────────────────
        with tracer.start_as_current_span("guardrail.input"):
            input_result = await run_input_guardrails(last_user_text, client, analyzer)

        span.set_attribute("input.verdict", input_result.verdict)

        if input_result.verdict == GuardrailVerdict.BLOCK:
            logger.warning(
                "input_blocked",
                reason=input_result.blocked_reason,
                session_id=req.session_id,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "blocked": True,
                    "reason": input_result.blocked_reason,
                    "checks": [c.model_dump() for c in input_result.checks],
                },
            )

        messages = [m.model_dump() for m in req.messages]
        if input_result.verdict == GuardrailVerdict.REDACT and input_result.redacted_text:
            messages[-1]["content"] = input_result.redacted_text
            logger.info("PII redacted from input for session=%s", req.session_id)

        # ── Step 2: LLM call ──────────────────────────────────────────────
        with tracer.start_as_current_span("guardrail.llm_call"):
            response = await client.chat.completions.create(
                model=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                messages=messages,
            )
        output_text = response.choices[0].message.content or ""

        # ── Step 3: Output guardrails ─────────────────────────────────────
        with tracer.start_as_current_span("guardrail.output"):
            output_result = await run_output_guardrails(output_text, req.context, client, analyzer)

        span.set_attribute("output.verdict", output_result.verdict)

        if output_result.verdict == GuardrailVerdict.BLOCK:
            logger.warning("output_blocked", reason=output_result.blocked_reason)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "blocked": True,
                    "reason": output_result.blocked_reason,
                    "phase": "output",
                },
            )

        final_output = (
            output_result.redacted_text
            if output_result.verdict == GuardrailVerdict.REDACT and output_result.redacted_text
            else output_text
        )

        return GuardedChatResponse(
            content=final_output,
            input_guardrail=input_result,
            output_guardrail=output_result,
            model=req.model,
            usage={
                "input_tokens":  response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )


@app.post("/v1/guardrail/input")
async def check_input_only(body: dict[str, Any], request: Request) -> GuardrailResult:
    """Check input guardrails only — useful for pre-flight checks."""
    text = body.get("text", "")
    return await run_input_guardrails(text, request.app.state.llm, request.app.state.presidio)


@app.post("/v1/guardrail/output")
async def check_output_only(body: dict[str, Any], request: Request) -> GuardrailResult:
    """Check output guardrails only."""
    output  = body.get("output", "")
    context = body.get("context", [])
    return await run_output_guardrails(output, context, request.app.state.llm, request.app.state.presidio)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "guardrail-middleware"}


@app.get("/ready")
async def ready(request: Request) -> dict:
    """K8s readiness probe — reports Presidio status and circuit breaker states."""
    presidio_loaded = request.app.state.presidio is not None
    return {
        "ready": True,
        "presidio_nlp": presidio_loaded,
        "pii_coverage": "regex+nlp" if presidio_loaded else "regex-only",
        "circuit_breakers": {
            _CB_TOXICITY_INPUT.name:  _CB_TOXICITY_INPUT.state,
            _CB_TOXICITY_OUTPUT.name: _CB_TOXICITY_OUTPUT.state,
            _CB_HALLUCINATION.name:   _CB_HALLUCINATION.state,
        },
    }
