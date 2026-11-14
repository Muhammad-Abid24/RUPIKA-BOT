"""Tests for human anatomy hand & finger count guardrails (SF-017 / Human Anatomy Guardrails)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from stockforge.images.guardrails import (
    enforce_human_anatomy_guardrail,
    is_human_subject_prompt,
    sanitize_and_enforce_plagiarism_free_prompt,
)
from stockforge.providers.base import ProviderVisionResponse


def test_is_human_subject_prompt_detection():
    """Verify accurate detection of human subjects vs non-human scenes."""
    # Positive human cases
    assert is_human_subject_prompt("Professional doctor holding a medical tablet in a hospital")
    assert is_human_subject_prompt("Portrait of an Asian businesswoman in modern office")
    assert is_human_subject_prompt("Two engineers examining blueprint on construction site")
    assert is_human_subject_prompt("Chef garnishing a gourmet plate with fresh herbs")
    assert is_human_subject_prompt("Close up of hands typing on a mechanical keyboard")
    assert is_human_subject_prompt("Smiling customer receiving coffee from barista")

    # Negative non-human cases
    assert not is_human_subject_prompt("Serene mountain landscape at golden hour sunrise")
    assert not is_human_subject_prompt("Modern clay concept car prototype in automotive design studio")
    assert not is_human_subject_prompt("Minimalist wooden desk setup with potted succulent plant")
    assert not is_human_subject_prompt("Abstract 3D geometric glass prisms with soft illumination")


def test_enforce_human_anatomy_guardrail_appends_exact_finger_rule():
    """Ensure prompt containing humans receives the mandatory 5-finger/10-finger guardrail."""
    original_prompt = "A female entrepreneur presenting quarterly statistics"
    guarded_prompt = enforce_human_anatomy_guardrail(original_prompt)

    assert "exactly 5 distinct fingers per visible hand" in guarded_prompt
    assert "10 fingers for two hands, 5 fingers for one hand" in guarded_prompt
    assert "no extra fingers" in guarded_prompt
    assert "no missing fingers" in guarded_prompt

    # Idempotent: should not append twice
    guarded_twice = enforce_human_anatomy_guardrail(guarded_prompt)
    assert guarded_twice == guarded_prompt


def test_sanitize_and_enforce_plagiarism_free_prompt_includes_anatomy_rule():
    """Verify end-to-end prompt sanitization adds both commercial stock and anatomy guardrails."""
    raw_prompt = "Worker holding a coffee cup in modern kitchen"
    sanitized = sanitize_and_enforce_plagiarism_free_prompt(raw_prompt)

    # Both commercial safety and anatomy rules must be present
    assert "100% original composition" in sanitized
    assert "5 distinct fingers per visible hand" in sanitized
    assert "10 fingers for two hands" in sanitized

    # Non-human prompt should only receive commercial safety, not anatomy guardrails
    non_human_raw = "Futuristic electric sports car parked in luxury garage"
    non_human_sanitized = sanitize_and_enforce_plagiarism_free_prompt(non_human_raw)
    assert "100% original composition" in non_human_sanitized
    assert "5 distinct fingers per visible hand" not in non_human_sanitized


@pytest.mark.anyio
async def test_vision_qc_strictly_fails_on_finger_count_mismatch(tmp_path):
    """Vision QC must fail closed when finger count is incorrect (e.g. 6 fingers or polydactyly)."""
    from stockforge.qc.vision import VisionQcService
    from stockforge.db.models import Asset
    from stockforge.providers.base import RoleType

    from unittest.mock import AsyncMock, MagicMock
    import json

    test_img = tmp_path / "test_human_asset.jpg"
    test_img.write_bytes(b"\xff\xd8\xff\xe0testjpegdata")

    # Mock provider returning polydactyly / incorrect finger count
    mock_provider = MagicMock()
    mock_response = ProviderVisionResponse(
        request_id="req_test_123",
        provider="mock_vision",
        role=RoleType.VISION_QC,
        model="rupika-qc",
        passed=False,
        analysis={
            "passed": False,
            "confidence": 0.95,
            "anatomy": {
                "has_humans": True,
                "visible_hands_count": 2,
                "fingers_per_hand_correct": False,
                "hands_and_fingers_defect": True,
                "face_defect": False,
                "interaction_defect": False,
                "details": "Left hand exhibits polydactyly with 6 distinct fingers; right hand has 5 fingers.",
            },
            "issues": ["polydactyly on left hand: 6 fingers instead of 5"],
            "notes": "Fatal anatomical defect: hand does not conform to the 5 fingers per hand rule.",
        },
    )
    mock_provider.analyze_image = AsyncMock(return_value=mock_response)

    # Test asset
    mock_asset = Asset(
        id=101,
        job_id=1,
        filename="test_human_asset.jpg",
        raw_path=str(test_img),
        status="raw_validated",
        sha256="abc12345",
    )

    mock_session = MagicMock()
    mock_session.get.return_value = mock_asset
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__enter__.return_value = mock_session

    mock_storage = MagicMock()
    mock_storage.read_bytes.return_value = b"\xff\xd8\xff\xe0testjpegdata"

    mock_router = MagicMock()
    mock_router.resolve_model.return_value = "rupika-qc"
    mock_router.registry.get_model.return_value = MagicMock(provider_name="test_vision_provider")
    mock_router.registry._provider_instances = {"test_vision_provider": mock_provider}

    qc_service = VisionQcService(
        session_factory=mock_session_factory,
        storage_service=mock_storage,
        router_or_provider=mock_provider,
    )

    evaluation = await qc_service.inspect_image_vision(asset_id=101)

    assert evaluation.passed is False
    det = json.loads(evaluation.details_json)
    assert any("anatomical_defect_hands_or_fingers" in issue for issue in det.get("issues", []))
    assert any("polydactyly" in issue for issue in det.get("issues", []))


def test_review_regeneration_prompt_improvement_targets_hand_finger_rule():
    """Verify prompt regeneration explicitly improves prompt with 5/10 finger count specification."""
    from stockforge.bot.review import ReviewService
    from stockforge.db.models import Asset, QcResult

    mock_sf = MagicMock()
    mock_qs = MagicMock()
    review_svc = ReviewService(session_factory=mock_sf, queue_service=mock_qs)

    asset = Asset(
        id=202,
        status="qc_failed",
        filename="asset_202.jpg",
    )
    qc_result = QcResult(
        asset_id=202,
        check_name="vision_qc",
        passed=False,
        score=0.4,
        details_json="""{
            "issues": ["anatomical_defect_hands_or_fingers"],
            "notes": "Left hand has an extra 6th finger.",
            "anatomy": {"hands_and_fingers_defect": true, "fingers_per_hand_correct": false}
        }""",
    )
    asset.qc_results = [qc_result]

    improved = review_svc._improve_prompt_for_regeneration(
        session=MagicMock(),
        asset=asset,
        base_prompt="A barista pouring latte art",
    )

    assert "exactly 5 distinct fingers per visible hand" in improved
    assert "10 fingers for two hands, 5 fingers for one hand" in improved
    assert "no extra fingers" in improved
