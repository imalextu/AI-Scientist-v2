from __future__ import annotations

import os
from typing import Any, Callable, Dict, Tuple

from openai import OpenAI

from .config import LLMConfig

DeltaCallback = Callable[[str], None]
CancelChecker = Callable[[], bool]


class RequestCancelledError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = config.api_key.strip() or os.getenv(config.api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"未提供 API Key，请设置配置 llm.api_key 或环境变量 {config.api_key_env}"
            )

        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def complete_with_meta(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: DeltaCallback | None = None,
        cancel_checker: CancelChecker | None = None,
    ) -> Tuple[str, Dict[str, int], str]:
        try:
            return self._stream_complete_with_meta(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_delta=on_delta,
                cancel_checker=cancel_checker,
            )
        except RequestCancelledError:
            raise
        except Exception as exc:
            if not self._should_fallback_to_non_stream(exc):
                raise
            # Fallback for providers that do not fully support stream responses.
            return self._single_complete_with_meta(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_delta=on_delta,
                cancel_checker=cancel_checker,
            )

    def _stream_complete_with_meta(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: DeltaCallback | None = None,
        cancel_checker: CancelChecker | None = None,
    ) -> Tuple[str, Dict[str, int], str]:
        self._raise_if_cancelled(cancel_checker)
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        text_chunks: list[str] = []
        finish_reason = ""
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            for chunk in response:
                self._raise_if_cancelled(cancel_checker, response=response)

                usage_chunk = getattr(chunk, "usage", None)
                if usage_chunk:
                    usage = {
                        "prompt_tokens": int(
                            getattr(usage_chunk, "prompt_tokens", 0) or 0
                        ),
                        "completion_tokens": int(
                            getattr(usage_chunk, "completion_tokens", 0) or 0
                        ),
                        "total_tokens": int(
                            getattr(usage_chunk, "total_tokens", 0) or 0
                        ),
                    }

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                choice = choices[0]
                delta = getattr(choice, "delta", None)
                piece = self._normalize_delta_content(getattr(delta, "content", None))
                if piece:
                    text_chunks.append(piece)
                    if on_delta:
                        on_delta(piece)

                reason = getattr(choice, "finish_reason", None)
                if isinstance(reason, str) and reason.strip():
                    finish_reason = reason.strip()
        finally:
            self._close_response(response)

        text = "".join(text_chunks).strip()
        return text, usage, finish_reason

    def _single_complete_with_meta(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: DeltaCallback | None = None,
        cancel_checker: CancelChecker | None = None,
    ) -> Tuple[str, Dict[str, int], str]:
        self._raise_if_cancelled(cancel_checker)
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )

        self._raise_if_cancelled(cancel_checker)
        text = (response.choices[0].message.content or "").strip()
        if text and on_delta:
            on_delta(text)
        finish_reason = (response.choices[0].finish_reason or "").strip()
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return text, usage, finish_reason

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: DeltaCallback | None = None,
        cancel_checker: CancelChecker | None = None,
    ) -> Tuple[str, Dict[str, int]]:
        text, usage, _ = self.complete_with_meta(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            on_delta=on_delta,
            cancel_checker=cancel_checker,
        )
        return text, usage

    def _normalize_delta_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
            return "".join(parts)

        text_attr = getattr(content, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        return ""

    def _should_fallback_to_non_stream(self, exc: Exception) -> bool:
        message = str(exc).lower()
        if not message:
            return False
        unsupported_hints = (
            "stream_options",
            "unknown parameter",
            "does not support stream",
            "does not support streaming",
            "streaming is not supported",
            "unsupported",
        )
        if "stream" not in message:
            return False
        return any(hint in message for hint in unsupported_hints)

    def _raise_if_cancelled(
        self,
        cancel_checker: CancelChecker | None,
        *,
        response: Any | None = None,
    ) -> None:
        if not cancel_checker:
            return
        if not cancel_checker():
            return
        if response is not None:
            self._close_response(response)
        raise RequestCancelledError("任务已取消")

    def _close_response(self, response: Any) -> None:
        close_fn = getattr(response, "close", None)
        if not callable(close_fn):
            return
        try:
            close_fn()
        except Exception:
            return
