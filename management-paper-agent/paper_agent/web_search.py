from __future__ import annotations

import html
import os
import re
import time
from typing import Any, Dict, List

import requests

from .config import WebSearchConfig


class WebSearchClient:
    def __init__(self, config: WebSearchConfig):
        self.config = config

    def search(self, query: str) -> List[Dict[str, str]]:
        if not self.config.enabled:
            return []

        clean_query = self._collapse_whitespace(str(query or ""))
        if not clean_query:
            return []

        provider = self.config.provider.lower()
        if provider != "bocha":
            return []

        return self._search_bocha(clean_query)

    def _search_bocha(self, query: str) -> List[Dict[str, str]]:
        api_key = self.config.api_key.strip() or os.getenv(
            self.config.api_key_env, ""
        ).strip()
        if not api_key:
            return []

        payload = self._request_json_post(
            self.config.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            body={
                "query": query,
                "summary": bool(self.config.summary),
                "count": min(50, max(1, int(self.config.max_results))),
                "freshness": self.config.freshness,
            },
            retries=1,
        )
        if payload is None:
            return []

        data_block = payload.get("data")
        if isinstance(data_block, dict):
            response_payload = data_block
        else:
            response_payload = payload

        pages = response_payload.get("webPages", {})
        raw_items = pages.get("value", []) if isinstance(pages, dict) else []
        if not isinstance(raw_items, list):
            return []

        items: List[Dict[str, str]] = []
        seen: set[str] = set()
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            normalized = self._normalize_bocha_item(raw_item)
            if not normalized:
                continue
            key = self._item_key(normalized)
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(normalized)
            if len(items) >= max(1, int(self.config.max_results)):
                break
        return items

    def _normalize_bocha_item(self, item: Dict[str, Any]) -> Dict[str, str] | None:
        title = self._clean_text(item.get("name", ""))
        if not title:
            return None

        url = self._clean_text(item.get("url", ""))
        summary = self._clean_text(item.get("summary", "")) or self._clean_text(
            item.get("snippet", "")
        )
        site_name = self._clean_text(item.get("siteName", ""))
        date_published = self._clean_text(item.get("datePublished", ""))

        return {
            "title": title,
            "url": url,
            "summary": summary,
            "site_name": site_name,
            "date_published": date_published,
            "source": "Bocha Web Search",
        }

    def _item_key(self, item: Dict[str, str]) -> str:
        url = self._clean_text(item.get("url", "")).lower()
        if url:
            return f"url:{url}"
        title = self._clean_text(item.get("title", "")).lower()
        if not title:
            return ""
        normalized_title = re.sub(r"\W+", "", title)
        return f"title:{normalized_title}" if normalized_title else ""

    def _request_json_post(
        self,
        url: str,
        *,
        headers: Dict[str, str] | None = None,
        body: Dict[str, Any] | None = None,
        retries: int = 1,
        timeout: int = 20,
    ) -> Dict[str, Any] | None:
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, dict):
                    return payload
                return None
            except (requests.RequestException, ValueError):
                if attempt >= retries:
                    return None
                time.sleep(0.3 * (attempt + 1))
        return None

    def _clean_text(self, value: Any) -> str:
        text = str(value or "")
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        return self._collapse_whitespace(text)

    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


def format_web_search_context(items: List[Dict[str, str]]) -> str:
    if not items:
        return "暂无可用网页检索结果。"

    lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        title = item.get("title", "").strip()
        if not title:
            continue
        url = item.get("url", "").strip() or "URL 未提供"
        site_name = item.get("site_name", "").strip() or "站点未知"
        date_published = item.get("date_published", "").strip() or "时间未知"
        summary = item.get("summary", "").strip() or "无摘要"
        lines.append(
            f"{idx}. {title}\n"
            f"   - 站点: {site_name}\n"
            f"   - 发布时间: {date_published}\n"
            f"   - 链接: {url}\n"
            f"   - 摘要: {summary[:400]}"
        )
    return "\n".join(lines) if lines else "暂无可用网页检索结果。"
