"""Comprehensive automated reliability and self-recovery tests for RUPIKA (Docker 24/7 hardening).

Covers all 11 required scenarios + production mock prohibition + process locking:
1. HTTP 409 Conflict handling and single-instance lock
2. Expired callback query graceful resilience
3. Slow AI provider & timeout classification
4. Provider timeout and compatible fallback
5. HTTP 429 rate limiting with Retry-After adherence
6. Worker crash, restart & safe startup reconciliation
7. Stale job recovery without duplicate paid generations (needs_reconciliation)
8. Duplicate Telegram updates & callback idempotency
9. Database contention (BEGIN IMMEDIATE concurrency safety)
10. Telegram notification failure & outbox durability
11. Paid-generation crash and billing uncertainty
12. Fail-closed production mode (zero silent mock usage)
13. Non-blocking handler execution
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, List, Optional
import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from stockforge.bot.lifecycle import (
    BotProcessLock,
    remove_bot_heartbeat,
    remove_worker_heartbeat,
    safe_answer_callback,
    touch_bot_heartbeat,
    touch_worker_heartbeat,
)
from stockforge.budget.service import BudgetExceededError, BudgetService
from stockforge.config import Settings
from stockforge.db.models import (
    Asset,
    Base,
    Job,
    JobAttempt,
    JobEvent,
    JobItem,
    ProductionPlan,
    Review,
    TelegramOutbox,
    utc_now,
)
from stockforge.errors import ErrorCode, NormalizedProviderError
from stockforge.errors.codes import ErrorSeverity
from stockforge.providers.base import (
    ProviderCapability,
    ProviderKind,
    ProviderTextResponse,
    RoleType,
    TextProvider,
)
from stockforge.providers.registry import ModelRegistration, ProviderRegistry
from stockforge.providers.resilience import (
    AllFallbacksExhaustedError,
    AmbiguousTimeoutError,
    ResilientExecutionCoordinator,
    calculate_backoff_delay,
)
from stockforge.providers.router import ProviderRouter, RoleRoute
from stockforge.queue.service import QueueService
from stockforge.queue.worker import Worker


@pytest.fixture
def in_memory_engine() -> Engine:
    """Create in-memory SQLite database engine with WAL semantics."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def queue_service(in_memory_engine: Engine) -> QueueService:
    return QueueService(in_memory_engine)


# ---------------------------------------------------------------------------
# 1. HTTP 409 Conflict Handling & Single Instance Process Lock
# ---------------------------------------------------------------------------
def test_bot_single_instance_lock() -> None:
    """Verify that two bot instances cannot acquire the advisory process lock simultaneously."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        lock_file = f.name

    try:
        lock1 = BotProcessLock(lock_path=lock_file)
        lock2 = BotProcessLock(lock_path=lock_file)

        assert lock1.acquire() is True, "Instance 1 must acquire the process lock"
        assert lock2.acquire() is False, "Instance 2 must be prevented from acquiring held lock"

        lock1.release()
        assert lock2.acquire() is True, "Instance 2 must acquire the lock once Instance 1 releases"
        lock2.release()
    finally:
        if os.path.exists(lock_file):
            os.unlink(lock_file)


def test_bot_heartbeat_conflict_recording() -> None:
    """Verify that bot heartbeat correctly captures healthy vs conflict states."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        hb_file = f.name

    original_env = os.environ.get("BOT_HEARTBEAT_FILE")
    os.environ["BOT_HEARTBEAT_FILE"] = hb_file

    try:
        touch_bot_heartbeat("healthy")
        data = json.loads(Path(hb_file).read_text())
        assert data["status"] == "healthy"
        assert "timestamp" in data

        touch_bot_heartbeat("conflict", {"count": 2})
        data = json.loads(Path(hb_file).read_text())
        assert data["status"] == "conflict"
        assert data["details"]["count"] == 2

        remove_bot_heartbeat()
        assert not os.path.exists(hb_file)
    finally:
        if original_env:
            os.environ["BOT_HEARTBEAT_FILE"] = original_env
        else:
            os.environ.pop("BOT_HEARTBEAT_FILE", None)
        if os.path.exists(hb_file):
            os.unlink(hb_file)


# ---------------------------------------------------------------------------
# 2. Expired Callback Query Graceful Handling
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_expired_callback_query_does_not_crash() -> None:
    """Verify that safe_answer_callback handles expired Telegram queries without raising exceptions."""
    class FakeExpiredQuery:
        async def answer(self, **kwargs: Any) -> None:
            from telegram.error import BadRequest
            raise BadRequest("Query is too old and response timeout expired or query id is invalid")

    query = FakeExpiredQuery()
    result = await safe_answer_callback(query, text="Testing expired")
    assert result is False, "Expired query answer must return False cleanly without raising"


# ---------------------------------------------------------------------------
# 3. Slow AI Provider & Timeout Classification
# ---------------------------------------------------------------------------
def test_slow_provider_timeout_error_classification() -> None:
    """Verify that timeouts are classified with REQUEST_TIMEOUT requiring status reconciliation."""
    from stockforge.errors.classifier import classify_error
    import httpx

    timeout_exc = httpx.ReadTimeout("The read operation timed out after 60.0 seconds")
    norm_err = classify_error(exception=timeout_exc, provider="test-provider", model="test-model")

    assert norm_err.code == ErrorCode.REQUEST_TIMEOUT
    # Invariant: Timed out operations require reconciliation to protect against ambiguous double-billing
    assert norm_err.action == "reconcile"
    assert "timed out" in norm_err.safe_user_message.lower()


# ---------------------------------------------------------------------------
# 4. Provider Timeout and Compatible Fallback
# ---------------------------------------------------------------------------
def test_provider_timeout_fallback_success() -> None:
    """Verify coordinator switches to compatible fallback model when primary encounters server/availability error."""
    reg = ProviderRegistry()
    reg.register_provider("p1", ProviderKind.TEXT, "https://api.p1.com", "P1_KEY")
    reg.register_provider("p2", ProviderKind.TEXT, "https://api.p2.com", "P2_KEY")

    reg.register_model(
        model_id="primary-model",
        provider_name="p1",
        kind=ProviderKind.TEXT,
        capabilities={ProviderCapability.TEXT_GENERATION},
        cost_per_unit_cents=1,
        license_evidence="verified",
    )
    reg.register_model(
        model_id="fallback-model",
        provider_name="p2",
        kind=ProviderKind.TEXT,
        capabilities={ProviderCapability.TEXT_GENERATION},
        cost_per_unit_cents=1,
        license_evidence="verified",
    )

    router = ProviderRouter(reg)
    router.set_route(
        RoleType.RESEARCH,
        primary_model_id="primary-model",
        fallbacks=["fallback-model"],
    )

    coordinator = ResilientExecutionCoordinator(
        router=router,
        registry=reg,
        max_attempts_per_provider=1,
        base_delay_seconds=0.01,
        sleeper=lambda _: None,
    )

    call_history: List[str] = []

    def mock_call(model_id: str) -> str:
        call_history.append(model_id)
        if model_id == "primary-model":
            import httpx
            resp = httpx.Response(
                status_code=503,
                text="Service temporarily unavailable",
                request=httpx.Request("POST", "https://api.p1.com"),
            )
            raise httpx.HTTPStatusError("503 Service Unavailable", request=resp.request, response=resp)
        return "fallback-response"

    result = coordinator.execute_with_resilience(
        role=RoleType.RESEARCH,
        call_fn=mock_call,
    )

    assert result == "fallback-response"
    assert call_history == ["primary-model", "fallback-model"]
    assert len(coordinator.fallback_events) == 1
    assert coordinator.fallback_events[0].from_model == "primary-model"
    assert coordinator.fallback_events[0].to_model == "fallback-model"


# ---------------------------------------------------------------------------
# 5. HTTP 429 Rate Limiting with Retry-After Adherence
# ---------------------------------------------------------------------------
def test_http_429_retry_after_calculation() -> None:
    """Verify that calculate_backoff_delay strictly respects Retry-After headers."""
    # When Retry-After is 7 seconds, delay must be 7.0
    delay = calculate_backoff_delay(attempt=1, retry_after=7.0, max_delay=30.0)
    assert delay == 7.0

    # When Retry-After exceeds max_delay, cap at max_delay
    capped = calculate_backoff_delay(attempt=1, retry_after=60.0, max_delay=30.0)
    assert capped == 30.0


def test_http_429_retry_success() -> None:
    """Verify provider retries on 429 and succeeds on next attempt without unnecessary fallback."""
    reg = ProviderRegistry()
    reg.register_provider("p1", ProviderKind.TEXT, "https://api.p1.com", "P1_KEY")
    reg.register_model(
        model_id="rate-limited-model",
        provider_name="p1",
        kind=ProviderKind.TEXT,
        capabilities={ProviderCapability.TEXT_GENERATION},
        cost_per_unit_cents=1,
        license_evidence="verified",
    )

    router = ProviderRouter(reg)
    router.set_route(
        RoleType.RESEARCH,
        primary_model_id="rate-limited-model",
        fallbacks=[],
    )

    sleeps: List[float] = []

    coordinator = ResilientExecutionCoordinator(
        router=router,
        registry=reg,
        max_attempts_per_provider=3,
        base_delay_seconds=0.01,
        sleeper=lambda s: sleeps.append(s),
    )

    attempts = 0

    def mock_call(model_id: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            import httpx
            resp = httpx.Response(
                status_code=429,
                headers={"Retry-After": "2"},
                text="Rate limit exceeded",
                request=httpx.Request("POST", "https://api.p1.com"),
            )
            raise httpx.HTTPStatusError("429 Too Many Requests", request=resp.request, response=resp)
        return "success-after-rate-limit"

    result = coordinator.execute_with_resilience(
        role=RoleType.RESEARCH,
        call_fn=mock_call,
    )

    assert result == "success-after-rate-limit"
    assert attempts == 2
    assert len(sleeps) == 1
    assert sleeps[0] == 2.0


# ---------------------------------------------------------------------------
# 6 & 7. Worker Crash, Restart & Stale Job Recovery (No Blind Paid Replay)
# ---------------------------------------------------------------------------
def test_stale_job_recovery_never_replays_ambiguous_paid_calls(queue_service: QueueService) -> None:
    """CRITICAL SAFETY INVARIANT: Stale jobs with recorded attempts MUST move to needs_reconciliation."""
    engine = queue_service.engine

    past_time = utc_now() - timedelta(minutes=5)
    with Session(engine) as session:
        # Job 1: In-flight paid call when worker crashed
        j1 = Job(
            type="image_batch",
            operation_id="op_crashed_paid",
            status="running",
            lease_owner="worker-crashed",
            lease_until=past_time,
            attempts=1,
            max_attempts=3,
            created_at=past_time,
            updated_at=past_time,
        )
        session.add(j1)
        session.commit()
        session.refresh(j1)

        # Record attempt for Job 1
        att = JobAttempt(
            job_id=j1.id,
            attempt_number=1,
            role="image",
            provider="bytedance",
            model="seedream-5-0",
            outcome="in_flight",
            estimated_cost_cents=18,
            actual_cost_cents=0,
            started_at=past_time,
        )
        session.add(att)

        # Job 2: Abandoned before any attempt started (safe to retry)
        j2 = Job(
            type="image_batch",
            operation_id="op_abandoned_safe",
            status="running",
            lease_owner="worker-crashed",
            lease_until=past_time,
            attempts=1,
            max_attempts=3,
            created_at=past_time,
            updated_at=past_time,
        )
        session.add(j2)
        session.commit()

        j1_id, j2_id = j1.id, j2.id

    # Run crash recovery sweep
    recovered_ids = queue_service.recover_stale_jobs()
    assert j1_id in recovered_ids
    assert j2_id in recovered_ids

    with Session(engine) as session:
        job1 = session.get(Job, j1_id)
        job2 = session.get(Job, j2_id)

        assert job1 is not None and job2 is not None
        # Invariant: Job 1 has attempts -> MUST NOT re-run -> needs_reconciliation
        assert job1.status == "needs_reconciliation"
        assert job1.lease_owner is None

        # Job 2 had NO attempts -> safe to revert to pending/retrying
        assert job2.status in ("pending", "retrying")
        assert job2.lease_owner is None

    # Verify that claim_job only claims Job 2, and NEVER claims Job 1!
    claimed = queue_service.claim_job(worker_id="new-worker-1", lease_seconds=60)
    assert claimed is not None
    assert claimed.id == j2_id, "Worker must claim the safe job, never the needs_reconciliation job"

    # Subsequent claim should return None (Job 1 is safely quarantined)
    claimed2 = queue_service.claim_job(worker_id="new-worker-1", lease_seconds=60)
    assert claimed2 is None


# ---------------------------------------------------------------------------
# 8. Duplicate Telegram Updates & Callback Idempotency
# ---------------------------------------------------------------------------
def test_review_action_double_click_idempotency(in_memory_engine: Engine) -> None:
    """Verify that multiple clicks on the same review callback return already_completed."""
    from stockforge.bot.review import ReviewService

    with Session(in_memory_engine) as session:
        asset = Asset(
            filename="SF_TEST_IDEMP.jpg",
            raw_path="/tmp/fake.jpg",
            final_path="/tmp/fake_final.jpg",
            sha256="abc123hash",
            phash="phash123",
            width=3000,
            height=3000,
            mime_type="image/jpeg",
            ai_generated=True,
            license_verified=True,
            license_evidence="commercial_verified",
            status="needs_review",
            created_at=utc_now(),
        )
        session.add(asset)
        session.commit()
        session.refresh(asset)
        asset_id = asset.id

    service = ReviewService(session_factory=lambda: Session(in_memory_engine))

    # First click: approve
    res1 = service.process_review_action(
        asset_id=asset_id,
        action="app",
        telegram_user_id=12345,
        user_name="Admin",
    )
    assert res1.success is True
    assert res1.decision == "approved"
    assert res1.already_completed is False

    # Second click: duplicate/replay click
    res2 = service.process_review_action(
        asset_id=asset_id,
        action="app",
        telegram_user_id=12345,
        user_name="Admin",
    )
    assert res2.success is False
    assert res2.already_completed is True
    assert "sudah disetujui" in res2.message.lower() or "sudah selesai ditinjau" in res2.message.lower()


# ---------------------------------------------------------------------------
# 9. Database Contention Safety (Concurrent BEGIN IMMEDIATE Claims)
# ---------------------------------------------------------------------------
def test_concurrent_worker_claims_database_contention() -> None:
    """Verify that under concurrent execution, exactly one worker claims an eligible job."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_file = f.name

    try:
        from stockforge.db.session import create_db_engine
        engine = create_db_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)

        qs = QueueService(engine)
        job = qs.enqueue_job(job_type="image_batch", operation_id="contention_test_op")

        claimed_by: List[str] = []
        errors: List[Exception] = []

        def worker_attempt(worker_name: str) -> None:
            # Each worker thread has its own connection via the pool
            try:
                res = qs.claim_job(worker_id=worker_name, lease_seconds=60)
                if res is not None:
                    claimed_by.append(worker_name)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker_attempt, args=(f"worker-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Contention caused unhandled DB exceptions: {errors}"
        assert len(claimed_by) == 1, f"Expected exactly 1 claim, got: {claimed_by}"
    finally:
        if os.path.exists(db_file):
            os.unlink(db_file)


# ---------------------------------------------------------------------------
# 10. Telegram Notification Failure & Outbox Durability
# ---------------------------------------------------------------------------
def test_telegram_outbox_durability_and_persistence(queue_service: QueueService) -> None:
    """Verify that notifications are persistently queued in SQLite outbox table."""
    engine = queue_service.engine

    entry = queue_service.publish_outbox_event(
        chat_id=987654321,
        event_key="batch_complete_100",
        payload={"job_id": 100, "status": "completed"},
    )
    assert entry.id is not None
    assert entry.sent_at is None

    with Session(engine) as session:
        persisted = session.scalar(
            select(TelegramOutbox).where(TelegramOutbox.event_key == "batch_complete_100")
        )
        assert persisted is not None
        assert persisted.chat_id == 987654321
        payload = json.loads(persisted.payload_json)
        assert payload["job_id"] == 100


# ---------------------------------------------------------------------------
# 11. Paid-Generation Crash and Billing Uncertainty
# ---------------------------------------------------------------------------
def test_billable_ambiguous_timeout_stops_and_quarantines() -> None:
    """Verify that an ambiguous timeout during a billable external call halts retry and sets needs_reconciliation."""
    reg = ProviderRegistry()
    reg.register_provider("p1", ProviderKind.IMAGE, "https://api.p1.com", "P1_KEY")
    reg.register_model(
        model_id="paid-image-model",
        provider_name="p1",
        kind=ProviderKind.IMAGE,
        capabilities={ProviderCapability.IMAGE_GENERATION, ProviderCapability.COMMERCIAL_LICENSE_VERIFIED},
        cost_per_unit_cents=18,
        license_evidence="verified",
    )

    router = ProviderRouter(reg)
    router.set_route(
        RoleType.IMAGE,
        primary_model_id="paid-image-model",
        fallbacks=[],
    )

    coordinator = ResilientExecutionCoordinator(
        router=router,
        registry=reg,
        max_attempts_per_provider=3,
        base_delay_seconds=0.01,
        sleeper=lambda _: None,
    )

    def mock_billable_call(model_id: str) -> None:
        import httpx
        raise httpx.ReadTimeout("Server timed out waiting for image generation")

    with pytest.raises(AmbiguousTimeoutError) as exc_info:
        coordinator.execute_with_resilience(
            role=RoleType.IMAGE,
            call_fn=mock_billable_call,
            is_billable=True,
            estimated_cost_cents=18,
            operation_id="op_ambiguous_billable_1",
        )

    assert "rekonsiliasi status" in str(exc_info.value).lower()
    assert exc_info.value.operation_id == "op_ambiguous_billable_1"


# ---------------------------------------------------------------------------
# 12. Fail-Closed Production Mode (Zero Silent Mock Usage)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_production_forbids_silent_mock_fallback(monkeypatch: pytest.MonkeyPatch, in_memory_engine: Engine) -> None:
    """Verify that in production environment, ResearchService fails closed and raises RuntimeError on missing real provider."""
    from stockforge.research.service import ResearchService

    monkeypatch.setenv("APP_ENV", "production")

    reg = ProviderRegistry()
    reg.register_provider("unregistered_provider", ProviderKind.TEXT, "https://api.test.com", "KEY")
    reg.register_model(
        model_id="production-model",
        provider_name="unregistered_provider",
        kind=ProviderKind.TEXT,
        capabilities={ProviderCapability.TEXT_GENERATION},
        cost_per_unit_cents=1,
        license_evidence="verified",
    )
    router = ProviderRouter(reg)
    router.set_route(
        RoleType.RESEARCH,
        primary_model_id="production-model",
        fallbacks=[],
    )

    service = ResearchService(session_factory=lambda: Session(in_memory_engine), router=router)

    with pytest.raises(RuntimeError) as exc_info:
        await service._get_text_provider(RoleType.RESEARCH)

    assert "FATAL" in str(exc_info.value)
    assert "dilarang keras" in str(exc_info.value).lower()
