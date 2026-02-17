from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests

from .config import RetrievalConfig


class LiteratureRetriever:
    def __init__(self, config: RetrievalConfig):
        self.config = config

    def search(self, query: str) -> List[Dict[str, str]]:
        if not self.config.enabled:
            return []
        clean_query = query.strip()
        if not clean_query:
            return []

        provider = self.config.provider.lower()
        try:
            if provider == "semantic_scholar":
                return self._search_semantic_scholar(clean_query)
            if provider == "arxiv":
                return self._search_arxiv(clean_query)
            return self._search_crossref(clean_query)
        except requests.RequestException:
            return []

    def _search_crossref(self, query: str) -> List[Dict[str, str]]:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": query,
                "rows": self.config.crossref.max_results,
                "sort": "relevance",
                "order": "desc",
            },
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        results: List[Dict[str, str]] = []
        for item in items:
            title = ""
            title_list = item.get("title", [])
            if title_list:
                title = title_list[0]
            authors = ", ".join(
                [
                    " ".join(
                        [
                            author.get("given", "").strip(),
                            author.get("family", "").strip(),
                        ]
                    ).strip()
                    for author in item.get("author", [])[:8]
                ]
            )
            year = ""
            date_parts = (
                item.get("issued", {})
                .get("date-parts", [[None]])[0]
            )
            if date_parts and date_parts[0]:
                year = str(date_parts[0])
            venue = ""
            container = item.get("container-title", [])
            if container:
                venue = container[0]
            abstract = (item.get("abstract") or "").strip()
            doi = item.get("DOI", "")

            if title:
                results.append(
                    {
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "venue": venue,
                        "abstract": abstract,
                        "source": f"Crossref (DOI: {doi})" if doi else "Crossref",
                    }
                )
        return results

    def _search_semantic_scholar(self, query: str) -> List[Dict[str, str]]:
        headers = {}
        api_key = os.getenv(self.config.semantic_scholar.api_key_env, "").strip()
        if api_key:
            headers["X-API-KEY"] = api_key

        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": query,
                "limit": self.config.semantic_scholar.max_results,
                "fields": "title,authors,venue,year,abstract,citationCount,url",
            },
            timeout=20,
        )
        resp.raise_for_status()
        items = resp.json().get("data", [])
        results: List[Dict[str, str]] = []
        for item in items:
            authors = ", ".join(
                [author.get("name", "").strip() for author in item.get("authors", [])[:8]]
            )
            title = item.get("title", "").strip()
            if title:
                results.append(
                    {
                        "title": title,
                        "authors": authors,
                        "year": str(item.get("year", "")),
                        "venue": item.get("venue", "").strip(),
                        "abstract": (item.get("abstract") or "").strip(),
                        "source": "Semantic Scholar",
                    }
                )
        return results

    def _search_arxiv(self, query: str) -> List[Dict[str, str]]:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": self.config.arxiv.max_results,
            },
            timeout=20,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        results: List[Dict[str, str]] = []
        for entry in entries:
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (
                entry.findtext("atom:summary", default="", namespaces=ns) or ""
            ).strip()
            published = (
                entry.findtext("atom:published", default="", namespaces=ns) or ""
            ).strip()
            year = published[:4] if len(published) >= 4 else ""
            authors = ", ".join(
                [
                    (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
                    for author in entry.findall("atom:author", ns)
                ]
            )
            if title:
                results.append(
                    {
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "venue": "arXiv",
                        "abstract": summary,
                        "source": "arXiv",
                    }
                )
        return results


def format_literature_context(items: List[Dict[str, str]]) -> str:
    if not items:
        return "暂无可用文献线索。"

    lines = []
    for idx, item in enumerate(items, start=1):
        title = item.get("title", "").strip()
        if not title:
            continue
        authors = item.get("authors", "").strip() or "未知作者"
        year = item.get("year", "").strip() or "年份未知"
        venue = item.get("venue", "").strip() or "来源未知"
        abstract = item.get("abstract", "").strip() or "无摘要"
        source = item.get("source", "").strip() or "未知来源"
        lines.append(
            f"{idx}. {title}\n"
            f"   - 作者: {authors}\n"
            f"   - 年份: {year}\n"
            f"   - 来源: {venue} | {source}\n"
            f"   - 摘要: {abstract[:400]}"
        )

    return "\n".join(lines) if lines else "暂无可用文献线索。"
