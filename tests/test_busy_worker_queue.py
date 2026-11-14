"""Tests for busy worker detection, queuing, and real-time user notification (anti-stuck UX)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from stockforge.bot.progress_handlers import jobs_command
from stockforge.bot.research_handlers import research_callback_handler
from stockforge.bot.review import get_review_service, review_callback_handler
from stockforge.db.models import (
    Asset,
    Base,
    Job,
    JobItem,
    ProductionPlan,
    ResearchRun,
    TelegramUser,
    utc_now,
)
from stockforge.queue.service import QueueService
from stockforge.research.gate import GenerationGate
from stockforge.research.planner import ProductionPlanner


@pytest.fixture
def db_engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_busy.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def test_queue_service_busy_detection_and_position(db_engine):
    """Verify QueueService accurately reports worker busy status and queue positions."""
    queue_svc = QueueService(db_engine)

    # Initially queue is empty and worker is idle
    summary = queue_svc.get_queue_summary()
    assert summary["is_busy"] is False
    assert summary["running_job_id"] is None
    assert summary["pending_count"] == 0

    # Add active running job
    now = utc_now()
    with Session(db_engine) as session:
        running_job = Job(
            type="image_batch",
            operation_id="op_running_1",
            status="running",
            lease_owner="worker_1",
            lease_until=now + timedelta(minutes=5),
            attempts=1,
            created_at=now,
            updated_at=now,
        )
        session.add(running_job)
        session.flush()

        item = JobItem(
            job_id=running_job.id,
            item_index=1,
            stage="qc",
            status="processing",
            created_at=now,
            updated_at=now,
        )
        session.add(item)

        # Add two pending jobs in queue
        pending_job_1 = Job(
            type="image_batch",
            operation_id="op_pending_1",
            status="pending",
            attempts=0,
            created_at=now,
            updated_at=now,
        )
        pending_job_2 = Job(
            type="image_batch",
            operation_id="op_pending_2",
            status="pending",
            attempts=0,
            created_at=now,
            updated_at=now,
        )
        session.add_all([pending_job_1, pending_job_2])
        session.commit()

        running_id = running_job.id
        p1_id = pending_job_1.id
        p2_id = pending_job_2.id

    # Now verify summary reports worker is busy
    summary = queue_svc.get_queue_summary()
    assert summary["is_busy"] is True
    assert summary["running_job_id"] == running_id
    assert summary["pending_count"] == 2

    # Verify detail of running job item
    detail = queue_svc.get_running_job_item_detail(running_id)
    assert detail is not None
    assert detail["item_index"] == 1
    assert detail["stage"] == "qc"

    # Verify queue positions:
    # Running job exists -> p1 is queue position #2 (running + 1)
    # p2 is queue position #3 (running + p1 + 1)
    assert queue_svc.get_queue_position(p1_id) == 2
    assert queue_svc.get_queue_position(p2_id) == 3


@pytest.mark.anyio
async def test_plan_approval_notifies_busy_worker_and_queues(db_engine, monkeypatch):
    """Verify that approving a production plan while worker is busy informs user of busy status and queue position."""
    monkeypatch.setenv("TELEGRAM_ADMIN_IDS", "12345678")
    session_factory = lambda: Session(db_engine)
    queue_svc = QueueService(db_engine)
    gate = GenerationGate(session_factory, queue_svc)
    planner = ProductionPlanner(session_factory)

    now = utc_now()
    with session_factory() as session:
        # Create Telegram user
        user = TelegramUser(telegram_id=12345678, role="admin", enabled=True)
        session.add(user)
        session.flush()

        # Create active running job simulating a busy worker
        running_job = Job(
            type="image_batch",
            operation_id="op_active_batch",
            status="running",
            lease_owner="worker_busy_test",
            lease_until=now + timedelta(minutes=5),
            attempts=1,
            created_at=now,
            updated_at=now,
        )
        session.add(running_job)
        session.flush()

        item = JobItem(
            job_id=running_job.id,
            item_index=1,
            stage="generating",
            status="processing",
            created_at=now,
            updated_at=now,
        )
        session.add(item)

        # Create draft plan ready for approval
        rr = ResearchRun(topic="Technology", trigger="manual")
        session.add(rr)
        session.flush()

        plan = ProductionPlan(
            research_run_id=rr.id,
            selected_niches_json='["AI Gadgets"]',
            concepts_json='[{"title": "Futuristic Device", "prompt": "Minimalist smart gadget on sleek wooden desk"}]',
            image_count=1,
            estimated_max_cost_cents=2,
            provider_routes_json='{"image": "mock-model"}',
            status="awaiting_approval",
            plan_hash="plan_hash_busy_123",
            expires_at=now + timedelta(hours=24),
        )
        session.add(plan)
        session.commit()
        plan_id = plan.id
        running_id = running_job.id

    # Mock Telegram Update and CallbackQuery
    class MockQuery:
        data = f"rsh:appr:{plan_id}:plan_hash_busy_123"
        edited_text = ""

        async def answer(self, *args, **kwargs):
            pass

        async def edit_message_text(self, text, parse_mode=None):
            self.edited_text = text

    class MockUser:
        id = 12345678

    class MockChat:
        id = 12345678
        type = "private"

    class MockUpdate:
        callback_query = MockQuery()
        effective_user = MockUser()
        effective_chat = MockChat()
        effective_message = None

    class MockContext:
        bot = None

    update = MockUpdate()
    await research_callback_handler(
        update=update,
        context=MockContext(),
        session_factory=session_factory,
        queue_service=queue_svc,
        gate=gate,
        planner=planner,
    )

    # Check response text
    response = update.callback_query.edited_text
    assert "Worker Sedang Sibuk Mengerjakan Task Lain!" in response
    assert f"Batch #{running_id}" in response
    assert "PENDING" in response
    assert "Mohon menunggu sebentar" in response


@pytest.mark.anyio
async def test_jobs_command_displays_worker_status_banner(db_engine, monkeypatch):
    """Verify /jobs displays real-time worker busy/standby status banners."""
    monkeypatch.setenv("TELEGRAM_ADMIN_IDS", "12345678")
    session_factory = lambda: Session(db_engine)

    # 1. Test when worker is idle
    class MockMessage:
        replied_text = ""

        async def reply_text(self, text, parse_mode=None):
            self.replied_text = text

    class MockChat:
        id = 12345678
        type = "private"

    class MockUpdate:
        effective_message = MockMessage()
        effective_chat = MockChat()
        effective_user = MockChat()
        callback_query = None

    class MockContext:
        bot = None

    # Empty queue
    update = MockUpdate()
    await jobs_command(update, MockContext(), session_factory)
    assert "Belum ada antrean pekerjaan" in update.effective_message.replied_text

    # 2. Add running job and pending job
    now = utc_now()
    with session_factory() as session:
        running = Job(
            type="image_batch",
            operation_id="op_job_cmd_1",
            status="running",
            lease_owner="worker_busy",
            lease_until=now + timedelta(minutes=5),
            attempts=1,
            created_at=now,
            updated_at=now,
        )
        session.add(running)
        session.flush()

        item = JobItem(
            job_id=running.id,
            item_index=1,
            stage="generating",
            status="processing",
            created_at=now,
            updated_at=now,
        )
        session.add(item)

        pending = Job(
            type="image_batch",
            operation_id="op_job_cmd_2",
            status="pending",
            attempts=0,
            created_at=now,
            updated_at=now,
        )
        session.add(pending)
        session.commit()
        run_id = running.id
        pen_id = pending.id

    update = MockUpdate()
    await jobs_command(update, MockContext(), session_factory)
    res_text = update.effective_message.replied_text

    assert "Status Worker:</b> ⏳ <b>SIBUK (BUSY)</b>" in res_text
    assert f"Batch #{run_id}" in res_text
    assert "SEDANG BERJALAN" in res_text
    assert f"Batch #{pen_id}" in res_text
    assert "DALAM ANTREAN" in res_text
