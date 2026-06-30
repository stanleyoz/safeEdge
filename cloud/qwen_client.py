"""
Base Qwen Cloud client — DashScope via OpenAI-compatible endpoint.
All cloud skills inherit from this.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# International (intl) endpoint is the hackathon default. Override with
# DASHSCOPE_BASE_URL for the mainland-China endpoint if needed.
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenCloudClient:
    def __init__(
        self,
        model: str = "qwen-plus",
        max_tokens: int = 1024,
        rpm_limit: int = 30,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._min_interval = 60.0 / rpm_limit
        self._last_call = 0.0
        self._client = self._init_client()

    def chat(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        retries: int = 3,
    ) -> Optional[str]:
        if self._client is None:
            logger.warning("Qwen Cloud client not initialised — check DASHSCOPE_API_KEY")
            return None

        self._rate_limit()

        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=self._max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                self._last_call = time.monotonic()
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("Qwen Cloud attempt %d failed: %s", attempt + 1, exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def chat_multimodal(
        self,
        system: str,
        user_content: list,   # list of {"type": "text"|"image_url", ...} dicts
        retries: int = 3,
    ) -> Optional[str]:
        """Send a multimodal (vision) request to a Qwen-VL model."""
        if self._client is None:
            logger.warning("Qwen Cloud client not initialised — check DASHSCOPE_API_KEY")
            return None

        self._rate_limit()

        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user_content},
                    ],
                    max_tokens=self._max_tokens,
                )
                self._last_call = time.monotonic()
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("Qwen-VL attempt %d failed: %s", attempt + 1, exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None

    async def chat_async(self, system: str, user: str, **kwargs) -> Optional[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.chat(system, user, **kwargs)
        )

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    @staticmethod
    def _init_client():
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("DASHSCOPE_API_KEY not set")
            return None
        base_url = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        try:
            from openai import OpenAI
            logger.info("Qwen Cloud client → %s", base_url)
            return OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            logger.error("openai package not installed")
            return None
