"""
Shared domain models for the AI Safety Platform.

Every service imports from here. These are the canonical data types
that flow between guardrail middleware, eval harness, prompt registry,
and monitoring pipeline.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GuardrailVerdict(str, Enum):
    ALLOW   = "allow"
    BLOCK   = "block"
    REDACT  = "redact"
    FLAG    = "flag"          # allow but mark for human review


class GuardrailCheckType(str, Enum):
    # Input checks
    PII_DETECTION       = "pii_detection"
    PROMPT_INJECTION    = "prompt_injection"
    JAILBREAK           = "jailbreak"
    TOPIC_RESTRICTION   = "topic_restriction"
    TOXICITY_INPUT      = "toxicity_input"
    # Output checks
    HALLUCINATION       = "hallucination"
    TOXICITY_OUTPUT     = "toxicity_output"
    PII_LEAK            = "pii_leak"
    FORMAT_COMPLIANCE   = "format_compliance"
    GROUNDING           = "grounding"


class EvalMetricType(str, Enum):
    # DeepEval built-ins
    ANSWER_RELEVANCY    = "answer_relevancy"
    FAITHFULNESS        = "faithfulness"
    HALLUCINATION       = "hallucination"
    TOXICITY            = "toxicity"
    BIAS                = "bias"
    SUMMARIZATION       = "summarization"
    # LLM-as-judge custom
    G_EVAL              = "g_eval"
    LLM_JUDGE           = "llm_judge"
    # Deterministic
    EXACT_MATCH         = "exact_match"
    REGEX_MATCH         = "regex_match"
    JSON_SCHEMA         = "json_schema"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    BERT_SCORE          = "bert_score"


class PromotionStatus(str, Enum):
    DRAFT      = "draft"
    TESTING    = "testing"       # in A/B test
    STAGING    = "staging"       # passed CI gate, pending prod
    PRODUCTION = "production"    # live
    DEPRECATED = "deprecated"
    ROLLED_BACK= "rolled_back"


# ---------------------------------------------------------------------------
# Guardrail models
# ---------------------------------------------------------------------------

class PIIEntity(BaseModel):
    entity_type:  str          # EMAIL, PHONE, SSN, NAME, CREDIT_CARD …
    start:        int
    end:          int
    original:     str
    replacement:  str          # <<EMAIL_1>> etc.
    confidence:   float


class GuardrailCheckResult(BaseModel):
    check_type:  GuardrailCheckType
    passed:      bool
    score:       float           # 0.0 = clean, 1.0 = max risk
    threshold:   float           # configured threshold for this check
    reason:      Optional[str] = None
    pii_entities: list[PIIEntity] = Field(default_factory=list)
    latency_ms:  float = 0.0


class GuardrailResult(BaseModel):
    """Complete result for one request (input or output phase)."""
    request_id:   str = Field(default_factory=_uid)
    phase:        str                          # "input" | "output"
    verdict:      GuardrailVerdict
    checks:       list[GuardrailCheckResult]
    redacted_text: Optional[str] = None       # populated when verdict=REDACT
    blocked_reason: Optional[str] = None
    total_latency_ms: float = 0.0
    timestamp:    datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Eval models
# ---------------------------------------------------------------------------

class EvalTestCase(BaseModel):
    """One input/output pair to be evaluated."""
    test_case_id:     str = Field(default_factory=_uid)
    request_id:       str                    # links back to a live request
    session_id:       str
    prompt_id:        str                    # which prompt version was used
    prompt_version:   int
    input_text:       str
    output_text:      str
    context:          list[str] = Field(default_factory=list)  # RAG chunks if any
    expected_output:  Optional[str] = None   # for deterministic metrics
    metadata:         dict[str, Any] = Field(default_factory=dict)
    captured_at:      datetime = Field(default_factory=_now)


class EvalMetricResult(BaseModel):
    metric_type:  EvalMetricType
    score:        float           # 0.0–1.0
    passed:       bool
    threshold:    float
    reason:       str             # DeepEval always provides a reason
    latency_ms:   float = 0.0


class EvalRunResult(BaseModel):
    """All metric scores for one test case."""
    eval_run_id:   str = Field(default_factory=_uid)
    test_case_id:  str
    request_id:    str
    prompt_id:     str
    prompt_version: int
    metrics:       list[EvalMetricResult]
    overall_score: float          # weighted average of all metrics
    passed:        bool           # all metrics above threshold
    judge_model:   str = "claude-sonnet-4-20250514"
    total_latency_ms: float = 0.0
    evaluated_at:  datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Prompt registry models
# ---------------------------------------------------------------------------

class PromptTemplate(BaseModel):
    """One versioned prompt with its metadata."""
    prompt_id:    str = Field(default_factory=_uid)
    name:         str            # human-readable slug: "customer-support-v3"
    version:      int            # monotonically incrementing
    content:      str            # the actual system prompt text
    variables:    list[str] = Field(default_factory=list)  # {{variable}} names
    model:        str = "claude-sonnet-4-20250514"
    temperature:  float = 0.7
    max_tokens:   int = 2048
    status:       PromotionStatus = PromotionStatus.DRAFT
    # Eval gate: must achieve this score to promote to production
    min_eval_score: float = 0.8
    tags:         list[str] = Field(default_factory=list)
    created_at:   datetime = Field(default_factory=_now)
    promoted_at:  Optional[datetime] = None
    created_by:   str = "system"
    metadata:     dict[str, Any] = Field(default_factory=dict)


class ABTestConfig(BaseModel):
    """Controls traffic splitting between two prompt versions."""
    test_id:      str = Field(default_factory=_uid)
    prompt_id:    str            # same name, two versions
    control_version:   int       # the current production version
    treatment_version: int       # the challenger
    traffic_split:     float = 0.5  # fraction going to treatment (0.0–1.0)
    min_samples:       int = 100    # minimum samples before analysis
    significance_level: float = 0.05
    active:       bool = True
    started_at:   datetime = Field(default_factory=_now)
    ended_at:     Optional[datetime] = None


class ABTestAssignment(BaseModel):
    """Records which version a specific request was assigned to."""
    request_id:   str
    test_id:      str
    version_assigned: int
    assigned_at:  datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Monitoring models
# ---------------------------------------------------------------------------

class QualitySnapshot(BaseModel):
    """Rolling quality metrics for a prompt version — stored in TimescaleDB."""
    snapshot_id:      str = Field(default_factory=_uid)
    prompt_id:        str
    prompt_version:   int
    window_start:     datetime
    window_end:       datetime
    sample_count:     int
    avg_overall_score: float
    avg_hallucination_score: float
    avg_toxicity_score: float
    avg_relevancy_score: float
    pass_rate:        float          # fraction of evals that passed all metrics
    p50_eval_latency_ms: float
    p99_eval_latency_ms: float
    guardrail_block_rate: float      # fraction of requests blocked by guardrails
    pii_detection_rate:  float
    created_at:       datetime = Field(default_factory=_now)


class DriftAlert(BaseModel):
    """Fired when quality drops significantly vs baseline."""
    alert_id:     str = Field(default_factory=_uid)
    prompt_id:    str
    prompt_version: int
    metric:       str
    baseline_score: float
    current_score:  float
    drop_pct:       float
    severity:       str            # "warning" | "critical"
    triggered_at:   datetime = Field(default_factory=_now)
    resolved_at:    Optional[datetime] = None
    auto_rollback:  bool = False   # whether rollback was triggered
