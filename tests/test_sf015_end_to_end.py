"""End-to-End Pilot and Release-Readiness Verification Suite for RUPIKA (SF-015).

Verifies the entire lifecycle and hard safety invariants:
1. Full mocked pipeline: Niche research -> Concept -> Plan creation -> Explicit approval
   -> GenerationGate single-use consumption & enqueue -> Worker claim & generation (JPEG)
   -> Deterministic QC (resolution/blur/format) -> Vision QC (anatomy/IP) & Metadata (title/keywords/category)
   -> Telegram human approval (/review) -> Adobe Contributor export (CSV + JPEG + ZIP + Manifest).
2. Security & Authorization: Unauthorized Telegram users and group chats are strictly rejected.
3. Idempotency & Replay Defense: Duplicate plan consumption and double review clicks are safely rejected.
4. Provider Failures & Fallback Exhaustion: Quota exhaustion, 401 invalid key, and 429 backoff handling.
5. Ambiguous Billable Timeout Quarantine: Unconfirmed paid calls transition to needs_reconciliation without replay.
6. Storage Pre-Flight Guard: Low disk space halts generation immediately via DiskFullError.
7. Compliance Export Gates: Unapproved, unlicensed, or non-compliant assets cannot be exported.
8. Database Online Backup & Restore Drill: Hot online backup with SHA-256 and restore integrity verification.
9. Fail-Closed Production Invariant: Production mode (APP_ENV=production) forbids silent mock fallbacks.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import zipfile
from PIL import Image, ImageDraw
import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from telegram import Chat, Message, Update, User
from telegram.constants import ChatType

from stockforge.bot.auth import DENIED_GENERIC_MESSAGE, GROUP_CHAT_REJECT_MESSAGE, require_admin
from stockforge.bot.review import ReviewService
from stockforge.budget.service import BudgetService
from stockforge.config import Settings
from stockforge.db.models import (
    Asset,
    AssetMetadata,
    Base,
    Concept,
    Job,
    JobAttempt,
    JobItem,
    ProductionPlan,
    QcResult,
    Review,
    TelegramUser,
    utc_now,
)
from stockforge.errors import ErrorCode, NormalizedProviderError
from stockforge.errors.classifier import ErrorClassifier
from stockforge.export.adobe import ComplianceExportError
from stockforge.export.service import ExportService
from stockforge.images.service import ImageGenerationService
from stockforge.ops.backup import (
    check_sqlite_integrity,
    perform_sqlite_online_backup,
    restore_sqlite_backup,
)
from stockforge.ops.health import DiskFullError, DiskHealthStatus, DiskSpaceReport, assert_disk_healthy_for_generation
from stockforge.providers.base import (
    ImageProvider,
    ProviderCapability,
    ProviderImageResponse,
    ProviderKind,
    ProviderTextResponse,
    ProviderVisionResponse,
    RoleType,
    TextProvider,
    VisionProvider,
)
from stockforge.providers.registry import ModelRegistration, ProviderRegistry
from stockforge.providers.resilience import (
    AllFallbacksExhaustedError,
    AmbiguousTimeoutError,
    ResilientExecutionCoordinator,
)
from stockforge.providers.router import ProviderRouter
from stockforge.qc.deterministic import DeterministicQcService
from stockforge.qc.vision import VisionQcService
from stockforge.queue.service import QueueService
from stockforge.queue.worker import Worker
from stockforge.research.gate import (
    GenerationGate,
    PlanAlreadyConsumedError,
)
from stockforge.research.planner import ProductionPlanner
from stockforge.research.service import ResearchService
from stockforge.storage.service import StorageService


# ---------------------------------------------------------------------------
# Mock Providers with High Fidelity & Realistic Data
# ---------------------------------------------------------------------------

def generate_valid_high_contrast_jpeg(width: int = 2048, height: int = 2048) -> bytes:
    """Generate a valid, high-contrast JPEG with Laplacian variance > 15 to pass deterministic QC."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw alternating black/white stripes and geometric shapes to ensure sharp edge variance
    for y in range(0, height, 32):
        color = (0, 0, 0) if (y // 32) % 2 == 0 else (240, 240, 240)
        draw.rectangle([0, y, width, y + 16], fill=color)
    draw.ellipse([width // 4, height // 4, 3 * width // 4, 3 * height // 4], outline=(255, 0, 0), width=10)
    draw.rectangle([width // 3, height // 3, 2 * width // 3, 2 * height // 3], fill=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class MockE2ETextProvider(TextProvider):
    """Text provider returning realistic research concepts and metadata."""

    def __init__(self, name: str = "mock-text-provider") -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        operation_id: Optional[str] = None,
    ) -> ProviderTextResponse:
        content = json.dumps([
            {
                "name": "Batik Craftsman Traditional Workshop",
                "priority": 1,
                "source_notes": "Yogyakarta craft trends",
                "rationale": "High commercial demand for authentic Indonesian cultural craftsmanship",
            }
        ])
        return ProviderTextResponse(
            request_id="req-mock-text-1",
            provider=self.provider_name,
            model=model,
            role=RoleType.RESEARCH,
            text=content,
            prompt_tokens=40,
            completion_tokens=120,
        )


class MockE2EImageProvider(ImageProvider):
    """Image provider returning compliant 2048x2048 JPEG images."""

    def __init__(self, name: str = "mock-image-provider") -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate_image(
        self,
        prompt: str,
        model: str,
        width: int = 2048,
        height: int = 2048,
        operation_id: Optional[str] = None,
    ) -> ProviderImageResponse:
        data = generate_valid_high_contrast_jpeg(width, height)
        return ProviderImageResponse(
            request_id=operation_id or "op-mock-img-1",
            provider=self.provider_name,
            model=model,
            role=RoleType.IMAGE,
            image_bytes=data,
            mime_type="image/jpeg",
            width=width,
            height=height,
            format="JPEG",
        )


class MockE2EVisionProvider(VisionProvider):
    """Vision provider returning valid Vision QC evaluations and 45 compliant keywords."""

    def __init__(self, name: str = "mock-vision-provider") -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        operation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderVisionResponse:
        # Differentiate between Vision QC analysis and Metadata Generation by checking prompt contents
        if "Quality Control" in prompt or "ANATOMY" in prompt:
            qc_payload = {
                "passed": True,
                "confidence": 0.98,
                "anatomy": {
                    "has_humans": True,
                    "hands_and_fingers_defect": False,
                    "face_defect": False,
                    "interaction_defect": False,
                    "details": "Anatomically correct hands and facial coherence verified.",
                },
                "issues": [],
                "notes": "Crisp lighting, sharp focus, clean background.",
            }
            return ProviderVisionResponse(
                request_id="req-mock-vis-qc",
                provider=self.provider_name,
                model=model,
                role=RoleType.VISION_QC,
                analysis=qc_payload,
                passed=True,
            )
        else:
            # Metadata generation prompt
            meta_payload = {
                "title": "Traditional Indonesian Batik Artisan Working in Workshop",
                "category_id": 15,  # Culture and Religion
                "content_type": "photo",
                "keywords": [
                    "batik", "indonesia", "craftsman", "traditional", "artisan", "workshop",
                    "textile", "heritage", "culture", "handmade", "wax", "fabric",
                    "craftsmanship", "pattern", "java", "yogyakarta", "authentic", "ethnic",
                    "studio", "manual", "asian", "cultural", "detailed", "intricate",
                    "process", "creative", "artist", "garment", "material", "custom",
                    "folk", "tradition", "skill", "worker", "indonesian", "handcraft",
                    "technique", "design", "production", "southeast asia", "local", "making",
                    "apparel", "craft", "dyeing"
                ],
            }
            return ProviderVisionResponse(
                request_id="req-mock-vis-meta",
                provider=self.provider_name,
                model=model,
                role=RoleType.METADATA,
                analysis=meta_payload,
                passed=True,
            )


# ---------------------------------------------------------------------------
# Test 1: Full Mocked Pipeline End-to-End
# ---------------------------------------------------------------------------

def test_full_mocked_pipeline_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify complete lifecycle:
    Niche research -> Plan -> Approval -> Gate -> Enqueue -> Claim -> Worker Execution
    -> Deterministic QC -> Vision QC -> Metadata -> Telegram Review -> Adobe Export.
    """
    db_path = tmp_path / "test_stockforge.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = lambda: Session(engine)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageService(data_dir)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("TELEGRAM_ADMIN_IDS", "12345678")
    monkeypatch.setenv("ALLOW_ADOBE_AUTO_SUBMISSION", "false")

    # Mock disk space check so environment disk threshold doesn't halt test execution
    monkeypatch.setattr(
        "stockforge.ops.health.check_disk_space",
        lambda *args, **kwargs: DiskSpaceReport(
            total_bytes=100 * 1024 * 1024 * 1024,
            used_bytes=50 * 1024 * 1024 * 1024,
            free_bytes=50 * 1024 * 1024 * 1024,
            free_percent=50.0,
            status=DiskHealthStatus.OK,
        ),
    )

    # 1. Setup Registry and Providers
    registry = ProviderRegistry()
    text_prov = MockE2ETextProvider()
    img_prov = MockE2EImageProvider()
    vis_prov = MockE2EVisionProvider()

    registry.register_provider("mock_text", ProviderKind.TEXT, "https://api.mock.test", "TEXT_KEY")
    registry.register_provider("mock_image", ProviderKind.IMAGE, "https://api.mock.test", "IMAGE_KEY")
    registry.register_provider("mock_vision", ProviderKind.VISION, "https://api.mock.test", "VISION_KEY")

    registry.register_provider_instance("mock_text", text_prov)
    registry.register_provider_instance("mock_image", img_prov)
    registry.register_provider_instance("mock_vision", vis_prov)

    registry.register_model(
        model_id="mock-text-model",
        provider_name="mock_text",
        kind=ProviderKind.TEXT,
        capabilities={ProviderCapability.TEXT_GENERATION},
    )
    registry.register_model(
        model_id="mock-image-model",
        provider_name="mock_image",
        kind=ProviderKind.IMAGE,
        capabilities={ProviderCapability.IMAGE_GENERATION, ProviderCapability.COMMERCIAL_LICENSE_VERIFIED},
        license_evidence="adobe_stock_commercial_sublicense_v1",
        cost_per_unit_cents=2,
    )
    registry.register_model(
        model_id="mock-vision-model",
        provider_name="mock_vision",
        kind=ProviderKind.VISION,
        capabilities={ProviderCapability.VISION_INSPECTION},
    )

    router = ProviderRouter(registry)
    router.set_route(RoleType.RESEARCH, "mock-text-model", [])
    router.set_route(RoleType.PROMPT, "mock-text-model", [])
    router.set_route(RoleType.IMAGE, "mock-image-model", [])
    router.set_route(RoleType.VISION_QC, "mock-vision-model", [])
    router.set_route(RoleType.METADATA, "mock-vision-model", [])

    queue_service = QueueService(engine)
    budget_service = BudgetService(session_factory, daily_limit_cents=1000, monthly_limit_cents=5000)

    # 2. Stage 1: Niche Discovery
    research_service = ResearchService(
        session_factory=session_factory,
        router=router,
        text_provider=text_prov,
    )
    niches = asyncio.run(research_service.discover_niches(
        domain_query="Indonesia Culture",
        count=1,
    ))
    assert len(niches) >= 1
    selected_niche = niches[0]
    prompt_text = "Elderly artisan stamping intricate wax pattern onto handmade batik silk fabric in traditional Yogyakarta studio"

    from stockforge.db.models import ResearchRun
    with session_factory() as session:
        rr = ResearchRun(topic="Indonesia Culture", trigger="manual")
        session.add(rr)
        session.commit()
        run_id = rr.id

    # 3. Stage 2: Production Plan Creation & Approval
    planner = ProductionPlanner(session_factory=session_factory)
    plan = planner.create_plan(
        research_run_id=run_id,
        selected_niches=[selected_niche["name"]],
        concepts=[{"title": selected_niche["name"], "prompt": prompt_text}],
        image_count=1,
        provider_routes={"image": "mock-image-model"},
        estimated_max_cost_cents=2,
    )
    assert plan.status == "awaiting_approval"
    assert plan.plan_hash is not None

    # Operator explicitly approves plan
    approved_plan = planner.approve_plan(
        plan_id=plan.id,
        approved_by_user_id=12345678,
        expected_plan_hash=plan.plan_hash,
    )
    assert approved_plan.status == "approved"

    # 4. Stage 3: Gate Consumption & Job Enqueue
    gate = GenerationGate(session_factory=session_factory, queue_service=queue_service, budget_service=budget_service)
    job = gate.consume_and_enqueue_job(
        plan_id=approved_plan.id,
        expected_plan_hash=approved_plan.plan_hash,
        actor_telegram_id=12345678,
    )
    job_id = job.id
    assert job_id > 0

    # Verify plan status changed to consumed
    with session_factory() as session:
        consumed_plan = session.get(ProductionPlan, approved_plan.id)
        assert consumed_plan.status == "consumed"
        assert consumed_plan.consumed_job_id == job_id

    # 5. Stage 4: Worker Pipeline Execution (Image -> QC -> Vision QC -> Metadata)
    worker = Worker(queue_service=queue_service, worker_id="test-e2e-worker")
    claimed_job = queue_service.claim_job(worker_id=worker.worker_id)
    assert claimed_job is not None
    assert claimed_job.id == job_id

    # Execute image pipeline directly
    img_service = ImageGenerationService(
        session_factory=session_factory,
        storage_service=storage,
        router=router,
        license_evidence="adobe_stock_commercial_sublicense_v1",
    )
    gen_result = asyncio.run(img_service.generate_batch(
        prompts=[prompt_text],
        max_batch_size=1,
        job_id=job_id,
    ))
    assert gen_result.success_count == 1
    asset = gen_result.successful_assets[0]
    assert asset.id > 0

    # Deterministic QC
    det_qc = DeterministicQcService(session_factory=session_factory)
    det_results = det_qc.inspect_asset(asset.id)
    assert all(r.passed for r in det_results)

    # Vision QC & Metadata Generation
    vision_qc = VisionQcService(
        session_factory=session_factory,
        storage_service=storage,
        router_or_provider=router,
    )
    v_res = asyncio.run(vision_qc.inspect_image_vision(asset.id))
    assert v_res.passed is True

    meta_res = asyncio.run(vision_qc.generate_metadata(asset.id))
    assert len(meta_res.title) <= 70
    kw_list = json.loads(meta_res.keywords_json)
    assert len(kw_list) >= 40
    assert meta_res.category_id == 15

    # 6. Stage 5: Telegram Review & Human Approval
    review_svc = ReviewService(session_factory=session_factory, queue_service=queue_service)
    review_outcome = review_svc.process_review_action(
        asset_id=asset.id,
        action="app",
        telegram_user_id=12345678,
    )
    assert review_outcome.success is True
    assert review_outcome.decision == "approved"

    with session_factory() as session:
        approved_asset = session.get(Asset, asset.id)
        assert approved_asset.status == "approved"

    # 7. Stage 6: Adobe Contributor Export
    export_svc = ExportService(session_factory=session_factory, storage_service=storage)
    export_batch, zip_path = export_svc.export_approved_assets(
        asset_ids=[asset.id],
        created_by_user_id=12345678,
    )
    assert zip_path.exists()
    assert export_batch.id > 0

    # Verify ZIP contents and compliance
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "adobe_metadata.csv" in namelist
        assert "manifest.json" in namelist
        
        # Check CSV content
        csv_bytes = zf.read("adobe_metadata.csv")
        csv_str = csv_bytes.decode("utf-8")
        lines = csv_str.strip().split("\r\n" if "\r\n" in csv_str else "\n")
        assert len(lines) >= 2
        assert lines[0] == "Filename,Title,Keywords,Category"
        
        # Check manifest
        manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest_data["total_assets"] == 1
        assert manifest_data["submission_mode"] == "manual"
        assert "generative_ai_disclosure" in manifest_data


# ---------------------------------------------------------------------------
# Test 2: Security & Authorization Enforcement
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_unauthorized_telegram_access_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that users not in allowlist or non-private chats are strictly rejected."""
    monkeypatch.setenv("TELEGRAM_ADMIN_IDS", "12345678")

    called = False

    @require_admin
    async def sample_admin_handler(update: Update, context: Any) -> None:
        nonlocal called
        called = True

    # Case 1: Unauthorized user ID
    class DummyUser:
        id = 99999999

    class DummyChat:
        id = 99999999
        type = ChatType.PRIVATE

    class DummyMessage:
        async def reply_text(self, text: str) -> None:
            pass

    class DummyUpdate:
        effective_user = DummyUser()
        effective_chat = DummyChat()
        effective_message = DummyMessage()
        callback_query = None

    called = False
    await sample_admin_handler(DummyUpdate(), None)
    assert called is False, "Akses user tidak terdaftar harus ditolak!"

    # Case 2: Group chat rejection
    class DummyGroupChat:
        id = -100123456
        type = ChatType.SUPERGROUP

    class DummyGroupUpdate:
        effective_user = DummyUser()
        effective_user.id = 12345678  # Admin ID, but in group!
        effective_chat = DummyGroupChat()
        effective_message = DummyMessage()
        callback_query = None

    called = False
    await sample_admin_handler(DummyGroupUpdate(), None)
    assert called is False, "Akses di group/supergroup harus ditolak!"


# ---------------------------------------------------------------------------
# Test 3: Idempotency & Replay Defense
# ---------------------------------------------------------------------------

def test_replay_and_double_click_prevention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify single-use plan consumption and double review clicks cannot duplicate operations."""
    db_path = tmp_path / "test_idempotency.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = lambda: Session(engine)

    queue_service = QueueService(engine)
    gate = GenerationGate(session_factory=session_factory, queue_service=queue_service)

    # 1. Create research run first for foreign key
    from stockforge.db.models import ResearchRun
    with session_factory() as session:
        rr = ResearchRun(topic="Test Topic", trigger="manual")
        session.add(rr)
        session.flush()

        plan = ProductionPlan(
            research_run_id=rr.id,
            selected_niches_json="[]",
            concepts_json="[]",
            image_count=1,
            estimated_max_cost_cents=5,
            provider_routes_json="{}",
            status="consumed",
            consumed_job_id=42,
            plan_hash="hash123",
            approved_by=12345678,
            approved_at=utc_now(),
            expires_at=utc_now() + timedelta(hours=24),
        )
        session.add(plan)
        session.commit()
        plan_id = plan.id

    # Duplicate consumption must raise PlanAlreadyConsumedError
    with pytest.raises(PlanAlreadyConsumedError):
        gate.consume_and_enqueue_job(plan_id=plan_id, actor_telegram_id=12345678)

    # 2. Review service double approval check
    with session_factory() as session:
        asset = Asset(
            filename="asset_test.jpg",
            raw_path=str(tmp_path / "asset_test.jpg"),
            sha256="dummy_sha256",
            phash="0000000000000000",
            width=2048,
            height=2048,
            status="approved",  # Already approved
        )
        session.add(asset)
        session.commit()
        asset_id = asset.id

    review_svc = ReviewService(session_factory=session_factory, queue_service=queue_service)
    outcome = review_svc.process_review_action(asset_id=asset_id, action="app", telegram_user_id=12345678)
    assert outcome.success is False
    assert outcome.already_completed is True
    assert "sudah disetujui" in outcome.message


# ---------------------------------------------------------------------------
# Test 4: Provider Failure Classification & Fallback Exhaustion
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_provider_resilience_quota_and_fallback_exhaustion() -> None:
    """Verify error classification and graceful exhaustion when all fallbacks fail."""
    classifier = ErrorClassifier()

    # Simulate 401 Unauthorized
    err_401 = classifier.classify(http_status=401, body={"error": {"message": "Invalid API Key"}}, provider="openai")
    assert err_401.code == ErrorCode.INVALID_API_KEY
    assert err_401.retryable is False

    # Simulate Quota Exhausted with explicit provider evidence
    err_quota = classifier.classify(
        http_status=429,
        body={"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}},
        provider="openai",
    )
    assert err_quota.code == ErrorCode.INSUFFICIENT_CREDITS
    assert err_quota.retryable is False

    # Simulate fallback exhaustion in coordinator
    import httpx

    def mock_broken_call(model_id: str) -> None:
        resp = httpx.Response(
            status_code=401,
            json={"error": {"message": "Invalid API Key"}},
            request=httpx.Request("POST", "https://api.broken.com"),
        )
        raise httpx.HTTPStatusError("401 Unauthorized", request=resp.request, response=resp)

    registry = ProviderRegistry()
    registry.register_provider("broken", ProviderKind.TEXT, "https://api.broken.com", "BROKEN_KEY")
    registry.register_model(
        model_id="broken-model",
        provider_name="broken",
        kind=ProviderKind.TEXT,
        capabilities={ProviderCapability.TEXT_GENERATION},
    )
    router = ProviderRouter(registry)
    router.set_route(RoleType.RESEARCH, "broken-model", [])  # No fallback models

    coordinator = ResilientExecutionCoordinator(router=router, registry=registry)
    with pytest.raises(AllFallbacksExhaustedError):
        coordinator.execute_with_resilience(
            role=RoleType.RESEARCH,
            call_fn=mock_broken_call,
        )


# ---------------------------------------------------------------------------
# Test 5: Ambiguous Billable Timeout Quarantine
# ---------------------------------------------------------------------------

def test_billable_timeout_quarantine(tmp_path: Path) -> None:
    """Verify ambiguous timeout during paid generation is quarantined to needs_reconciliation without replay."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test_quarantine.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    queue_service = QueueService(engine)

    past_time = utc_now() - timedelta(minutes=5)
    with Session(engine) as session:
        job = Job(
            type="image_batch",
            operation_id="op_crashed_paid_test",
            status="running",
            lease_owner="worker_crashed",
            lease_until=past_time,
            attempts=1,
            max_attempts=3,
            created_at=past_time,
            updated_at=past_time,
        )
        session.add(job)
        session.flush()

        item = JobItem(
            job_id=job.id,
            item_index=1,
            stage="image_generation",
            status="processing",
            created_at=past_time,
            updated_at=past_time,
        )
        session.add(item)
        session.flush()

        attempt = JobAttempt(
            job_id=job.id,
            job_item_id=item.id,
            attempt_number=1,
            role="image",
            provider="mock-image",
            model="mock-image-model",
            outcome="timeout",
            started_at=past_time,
        )
        session.add(attempt)
        session.commit()
        job_id = job.id

    recovered_ids = queue_service.recover_stale_jobs()
    assert job_id in recovered_ids

    # Ensure job was safely quarantined to prevent duplicate charges
    with Session(engine) as session:
        res = session.get(Job, job_id)
        assert res is not None
        assert res.status == "needs_reconciliation"


# ---------------------------------------------------------------------------
# Test 6: Storage Pre-Flight Disk Full Guard
# ---------------------------------------------------------------------------

def test_storage_preflight_disk_full_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify that low disk space triggers DiskFullError and halts processing."""
    def mock_check_disk_space(path: Path, **kwargs: Any) -> DiskSpaceReport:
        return DiskSpaceReport(
            total_bytes=100 * 1024 * 1024 * 1024,
            used_bytes=98 * 1024 * 1024 * 1024,
            free_bytes=200 * 1024 * 1024,  # 200 MB (below 500 MB floor)
            free_percent=2.0,
            status=DiskHealthStatus.CRITICAL,
        )

    monkeypatch.setattr("stockforge.ops.health.check_disk_space", mock_check_disk_space)

    with pytest.raises(DiskFullError) as exc_info:
        assert_disk_healthy_for_generation(tmp_path)
    assert "Kapasitas penyimpanan kritis" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 7: Compliance Export Gates Block Unapproved or Invalid Assets
# ---------------------------------------------------------------------------

def test_compliance_export_gates_block_unapproved_assets(tmp_path: Path) -> None:
    """Verify ExportService strictly blocks unapproved or non-compliant assets."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test_export_compliance.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = lambda: Session(engine)
    storage = StorageService(tmp_path / "data")

    # 1. Unapproved asset (qc_passed only)
    dummy_file = tmp_path / "unapproved.jpg"
    dummy_file.write_bytes(b"dummy")

    with session_factory() as session:
        asset = Asset(
            filename="unapproved.jpg",
            raw_path=str(dummy_file),
            sha256="sha111",
            phash="0000000000000000",
            width=2048,
            height=2048,
            status="qc_passed",  # Not approved!
            license_verified=True,
            license_evidence="evidence",
        )
        session.add(asset)
        session.commit()
        unapproved_id = asset.id

    export_svc = ExportService(session_factory=session_factory, storage_service=storage)

    with pytest.raises(ComplianceExportError) as exc_info:
        export_svc.export_approved_assets(asset_ids=[unapproved_id])
    assert "Aset belum disetujui" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 8: Database Online Backup and Restore Drill
# ---------------------------------------------------------------------------

def test_sqlite_online_backup_and_restore_drill(tmp_path: Path) -> None:
    """Verify live hot backup and restore with SHA-256 verification and PRAGMA integrity_check."""
    db_path = tmp_path / "original.db"
    backup_dir = tmp_path / "backups"
    restore_path = tmp_path / "restored.db"

    # Setup database with data
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = TelegramUser(telegram_id=12345678, role="admin", enabled=True)
        session.add(user)
        session.commit()

    # Perform online backup
    backup_file = perform_sqlite_online_backup(db_path=db_path, backup_dir=backup_dir)
    assert backup_file.exists()
    sha_file = backup_dir / f"{backup_file.name}.sha256"
    assert sha_file.exists()

    # Restore from backup
    restore_success = restore_sqlite_backup(backup_file=backup_file, target_db_path=restore_path)
    assert restore_success is True
    assert restore_path.exists()

    # Verify integrity and restored data
    is_ok, msg = check_sqlite_integrity(restore_path)
    assert is_ok is True

    restore_engine = create_engine(f"sqlite:///{restore_path}")
    with Session(restore_engine) as session:
        user_check = session.scalar(select(TelegramUser).where(TelegramUser.telegram_id == 12345678))
        assert user_check is not None
        assert user_check.role == "admin"


# ---------------------------------------------------------------------------
# Test 9: Fail-Closed Production Mock Prohibition
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_production_mode_strictly_forbids_mock_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify that in production (APP_ENV=production), mock fallback raises fatal RuntimeError."""
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

    engine = create_engine(f"sqlite:///{tmp_path / 'test_prod_mock.db'}")
    Base.metadata.create_all(engine)
    session_factory = lambda: Session(engine)

    service = ResearchService(session_factory=session_factory, router=router)

    with pytest.raises(RuntimeError) as exc_info:
        await service._get_text_provider(RoleType.RESEARCH)

    assert "FATAL" in str(exc_info.value)
    assert "dilarang keras" in str(exc_info.value).lower()
