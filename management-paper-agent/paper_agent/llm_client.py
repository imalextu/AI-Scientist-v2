from __future__ import annotations

import os
from typing import Dict, Tuple

from openai import OpenAI

from .config import LLMConfig


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
    ) -> Tuple[str, Dict[str, int], str]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )

        text = (response.choices[0].message.content or "").strip()
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
    ) -> Tuple[str, Dict[str, int]]:
        text, usage, _ = self.complete_with_meta(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text, usage
