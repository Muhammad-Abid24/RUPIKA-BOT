"""Dynamic image model pricing resolver and cost estimator (SF-008/SF-021)."""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Dict, Optional, Tuple

logger = logging.getLogger("stockforge.providers.pricing")

# Fallback catalog for verified image generation models (USD per generated image)
# Based on OpenRouter and direct provider pricing specifications
KNOWN_IMAGE_MODEL_PRICING: Dict[str, float] = {
    "meta/muse-image": 0.010,
    "bytedance-seed/seedream-5-0-flash": 0.018,
    "bytedance-seed/seedream-5-0-lite": 0.035,
    "bytedance-seed/seedream-5-0-pro": 0.045,
    "bytedance-seed/seedream-4.5": 0.040,
    "openai/gpt-5-image-mini": 0.008,
    "google/gemini-2.5-flash-image": 0.030,
    "google/gemini-3.1-flash-lite-image": 0.030,
    "openai/gpt-5.4-image-2": 0.030,
    "openai/gpt-5-image": 0.040,
    "openai/dall-e-3": 0.040,
    "dall-e-3": 0.040,
    "recraft/recraft-v4": 0.040,
    "recraft/recraft-v4.1": 0.040,
    "google/gemini-3.1-flash-image": 0.060,
    "google/gemini-3-pro-image": 0.120,
}

# In-memory cache: model_id -> (price_usd, cached_at_timestamp)
_PRICING_CACHE: Dict[str, Tuple[float, float]] = {}
_CACHE_TTL_SECONDS = 900  # 15 minutes


def get_active_image_model() -> str:
    """Retrieve the configured image model identifier from .env or environment with hot-reload."""
    from pathlib import Path
    for env_path in [
        Path(".env"),
        Path("/app/.env"),
        Path("/data/.env"),
        Path("/host_mnt/Users/muhammadabid/script-bot/Rupika/RUPIKA-BOT/.env"),
        Path("/Users/muhammadabid/script-bot/Rupika/RUPIKA-BOT/.env"),
    ]:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("IMAGE_MODEL=") and not line.startswith("#"):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
    return os.environ.get("IMAGE_MODEL", "meta/muse-image").strip()


def resolve_image_unit_cost_usd(model_id: Optional[str] = None) -> float:
    """
    Resolve the valid unit price in USD per generated image for the given model.
    Tries live endpoint lookup from OpenRouter, falling back to verified model pricing catalog.
    """
    target_model = (model_id or get_active_image_model()).strip()

    # 1. Check in-memory cache
    now = time.time()
    if target_model in _PRICING_CACHE:
        price, ts = _PRICING_CACHE[target_model]
        if now - ts < _CACHE_TTL_SECONDS:
            return price

    # 2. Try fetching live pricing from OpenRouter API
    api_key = os.environ.get("IMAGE_API_KEY", "").strip()
    try:
        url = f"https://openrouter.ai/api/v1/models/{target_model}/endpoints"
        headers = {"User-Agent": "RupikaBot/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode())
            endpoints = data.get("data", {}).get("endpoints", [])
            if endpoints:
                pricing = endpoints[0].get("pricing", {})
                img_out = float(pricing.get("image_output") or 0)
                img_tok = float(pricing.get("image_token") or 0)
                resolved_price: Optional[float] = None

                # OpenRouter diffusion/image generation token billing (standard 1024x1024 uses 4,175 tokens)
                if img_tok > 0:
                    resolved_price = round(img_tok * 4175.0, 4)
                elif img_out >= 0.001:
                    resolved_price = round(img_out, 4)
                elif img_out > 0:
                    # Token/unit multiplier
                    resolved_price = round(img_out * 1000.0, 4)

                if resolved_price is not None and resolved_price > 0:
                    _PRICING_CACHE[target_model] = (resolved_price, now)
                    logger.info("Live image pricing resolved for %s: $%.4f/img", target_model, resolved_price)
                    return resolved_price
    except Exception as ex:
        logger.debug("Live pricing lookup for %s failed (%s), using catalog fallback", target_model, ex)

    # 3. Fallback to verified catalog
    if target_model in KNOWN_IMAGE_MODEL_PRICING:
        cat_price = KNOWN_IMAGE_MODEL_PRICING[target_model]
        _PRICING_CACHE[target_model] = (cat_price, now)
        return cat_price

    # Fallback to standard 4 cents
    default_price = 0.04
    _PRICING_CACHE[target_model] = (default_price, now)
    return default_price


def estimate_production_cost(image_count: int, model_id: Optional[str] = None) -> Tuple[int, float, str, str]:
    """
    Calculate valid production cost for a batch of images.
    Returns:
        (cost_cents: int, total_cost_usd: float, model_display_name: str, formatted_cost_usd: str)
    """
    model = model_id or get_active_image_model()
    unit_cost_usd = resolve_image_unit_cost_usd(model)
    total_cost_usd = round(image_count * unit_cost_usd, 4)
    cost_cents = max(1, int(round(total_cost_usd * 100.0)))

    # Format USD display: if fractional cents exist (e.g. 0.018, 0.035), show up to 3 decimals
    cents_val = total_cost_usd * 100.0
    if abs(cents_val - round(cents_val)) > 0.005:
        formatted_usd = f"{total_cost_usd:.3f}"
    else:
        formatted_usd = f"{total_cost_usd:.2f}"

    return cost_cents, total_cost_usd, model, formatted_usd
