"""Centralized error classification engine (SF-017)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from stockforge.errors.codes import (
    ErrorCode,
    NormalizedProviderError,
)
from stockforge.errors.provider_adapters.anthropic import AnthropicErrorAdapter
from stockforge.errors.provider_adapters.base import (
    BaseProviderErrorAdapter,
    sanitize_sensitive_details,
)
from stockforge.errors.provider_adapters.gemini import GeminiErrorAdapter
from stockforge.errors.provider_adapters.generic import GenericErrorAdapter
from stockforge.errors.provider_adapters.openai import OpenAIErrorAdapter
from stockforge.errors.provider_adapters.stability import StabilityErrorAdapter

logger = logging.getLogger("stockforge.errors.classifier")


class ErrorClassifier:
    """Classifies provider and transport errors into normalized domain errors."""

    def __init__(self, adapters: Optional[List[BaseProviderErrorAdapter]] = None) -> None:
        self.adapters = adapters or [
            OpenAIErrorAdapter(),
            AnthropicErrorAdapter(),
            GeminiErrorAdapter(),
            StabilityErrorAdapter(),
            GenericErrorAdapter(),  # Fallback must be last
        ]

    def get_adapter(self, provider: Optional[str]) -> BaseProviderErrorAdapter:
        if provider:
            for adapter in self.adapters:
                if adapter.can_handle(provider):
                    return adapter
        return self.adapters[-1]  # Generic fallback

    def classify(
        self,
        http_status: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        request_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        exception: Optional[Exception] = None,
    ) -> NormalizedProviderError:
        """Classify HTTP status, response body, headers, or Python exception into NormalizedProviderError."""
        if isinstance(http_status, Exception):
            if exception is None:
                exception = http_status
            http_status = None

        if isinstance(exception, NormalizedProviderError):
            if provider and (not exception.provider or exception.provider == "generic"):
                exception.provider = provider
            if model and not exception.model:
                exception.model = model
            return exception

        status = http_status
        content_body = body
        req_headers = headers

        # Extract information from common exception types if provided
        if exception is not None:
            resp_obj = getattr(exception, "response", None)

            # Check for HTTP status attributes (direct or from response)
            if status is None:
                if resp_obj is not None and hasattr(resp_obj, "status_code"):
                    status = resp_obj.status_code
                else:
                    status = getattr(exception, "status_code", None) or getattr(exception, "status", None)
            
            # Check for headers (direct or from response)
            if req_headers is None:
                if resp_obj is not None and hasattr(resp_obj, "headers"):
                    req_headers = dict(resp_obj.headers)
                else:
                    exc_headers = getattr(exception, "headers", None)
                    if isinstance(exc_headers, (dict, list)):
                        req_headers = dict(exc_headers)

            # Check for response body or message
            if content_body is None:
                resp_obj = getattr(exception, "response", None)
                if resp_obj is not None and hasattr(resp_obj, "text"):
                    content_body = getattr(resp_obj, "text", "")
                elif hasattr(exception, "body"):
                    content_body = getattr(exception, "body", "")
                else:
                    content_body = str(exception)

        adapter = self.get_adapter(provider)
        normalized = adapter.classify(
            http_status=status,
            headers=req_headers,
            body=content_body,
            request_id=request_id,
            provider=provider or "generic",
            model=model,
        )

        logger.debug(
            "Classified error: provider=%s model=%s status=%s code=%s action=%s",
            normalized.provider,
            normalized.model,
            normalized.http_status,
            normalized.code.value,
            normalized.action,
        )
        return normalized


# Global singleton instance
default_classifier = ErrorClassifier()


def classify_error(
    http_status: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    request_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    exception: Optional[Exception] = None,
) -> NormalizedProviderError:
    """Convenience functional wrapper for error classification."""
    return default_classifier.classify(
        http_status=http_status,
        headers=headers,
        body=body,
        request_id=request_id,
        provider=provider,
        model=model,
        exception=exception,
    )

