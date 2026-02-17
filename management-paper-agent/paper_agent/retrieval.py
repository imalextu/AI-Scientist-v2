from __future__ import annotations

import html
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List

import requests

from .config import RetrievalConfig

OPENALEX_MAILTO_ENV = "OPENALEX_MAILTO"
SEMANTIC_SCHOLAR_API_KEY_ENV = "S2_API_KEY"
RECALL_MULTIPLIER = 4
MAX_RECALL_CAP = 48


class LiteratureRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        *,
        query_expander: Callable[[str], List[str]] | None = None,
    ):
        self.config = config
        self.query_expander = query_expander

    def search(self, query: str) -> List[Dict[str, str]]:
        if not self.config.enabled:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        query_variants = self._build_query_variants(clean_query)
        provider = self.config.provider.lower()
        if provider == "semantic_scholar":
            search_fn = self._search_semantic_scholar_candidates
        elif provider == "arxiv":
            search_fn = self._search_arxiv_candidates
        elif provider == "crossref":
            search_fn = self._search_crossref_candidates
        else:
            search_fn = self._search_openalex_candidates

        candidates = self._collect_candidates(query_variants, search_fn)
        deduped = self._dedupe_candidates(candidates)
        deduped.sort(
            key=lambda item: (
                self._safe_int(item.get("citation_count")),
                self._safe_int(item.get("year")),
                len((item.get("abstract") or "").strip()),
            ),
            reverse=True,
        )

        pool_size = self._recall_pool_size()
        return [self._to_public_item(item) for item in deduped[:pool_size]]

    def _build_query_variants(self, topic: str) -> List[str]:
        variants: List[str] = [topic]
        if self.query_expander:
            try:
                expanded = self.query_expander(topic)
            except Exception:
                expanded = []
            for query in expanded:
                text = self._collapse_whitespace(str(query or ""))
                if text:
                    variants.append(text)

        deduped: List[str] = []
        seen: set[str] = set()
        for query in variants:
            cleaned = self._collapse_whitespace(query)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
            if len(deduped) >= 2:
                break

        return deduped if deduped else [topic]

    def _collect_candidates(
        self,
        topics: List[str],
        search_fn: Callable[[str], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for topic in topics:
            candidates.extend(search_fn(topic))
        return candidates

    def _recall_pool_size(self) -> int:
        return min(
            MAX_RECALL_CAP,
            max(self.config.max_results, self.config.max_results * RECALL_MULTIPLIER),
        )

    def _search_openalex_candidates(self, query: str) -> List[Dict[str, Any]]:
        per_page = min(200, self._recall_pool_size())
        mailto = os.getenv(OPENALEX_MAILTO_ENV, "").strip()

        params: Dict[str, Any] = {
            "search": query,
            "per-page": per_page,
            "sort": "relevance_score:desc",
        }
        if mailto:
            params["mailto"] = mailto

        payload = self._request_json(
            "https://api.openalex.org/works",
            params=params,
            retries=2,
        )
        if payload is None:
            return []

        candidates: List[Dict[str, Any]] = []
        for item in payload.get("results", []):
            normalized = self._normalize_openalex_item(item)
            if normalized:
                candidates.append(normalized)
        return candidates

    def _search_crossref_candidates(self, query: str) -> List[Dict[str, Any]]:
        payload = self._request_json(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": query,
                "rows": self._recall_pool_size(),
                "sort": "relevance",
                "order": "desc",
            },
            retries=1,
        )
        if payload is None:
            return []

        candidates: List[Dict[str, Any]] = []
        for item in payload.get("message", {}).get("items", []):
            normalized = self._normalize_crossref_item(item)
            if normalized:
                candidates.append(normalized)
        return candidates

    def _search_semantic_scholar_candidates(self, query: str) -> List[Dict[str, Any]]:
        headers: Dict[str, str] = {}
        api_key = os.getenv(SEMANTIC_SCHOLAR_API_KEY_ENV, "").strip()
        if api_key:
            headers["X-API-KEY"] = api_key

        payload = self._request_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": query,
                "limit": min(100, self._recall_pool_size()),
                "fields": (
                    "title,authors,venue,year,abstract,citationCount,url,externalIds"
                ),
            },
            retries=1,
        )
        if payload is None:
            return []

        candidates: List[Dict[str, Any]] = []
        for item in payload.get("data", []):
            normalized = self._normalize_semantic_item(item)
            if normalized:
                candidates.append(normalized)
        return candidates

    def _search_arxiv_candidates(self, query: str) -> List[Dict[str, Any]]:
        text = self._request_text(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": min(30, self._recall_pool_size()),
            },
            retries=1,
        )
        if not text:
            return []

        root = ET.fromstring(text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        candidates: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            normalized = self._normalize_arxiv_item(entry, ns)
            if normalized:
                candidates.append(normalized)
        return candidates

    def _dedupe_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_key: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            key = self._candidate_key(candidate)
            if not key:
                continue
            existing = best_by_key.get(key)
            if existing is None or self._is_better_raw_candidate(candidate, existing):
                best_by_key[key] = candidate
        return list(best_by_key.values())

    def _candidate_key(self, candidate: Dict[str, Any]) -> str:
        doi = self._clean_text(candidate.get("doi", "")).lower()
        if doi:
            return f"doi:{doi}"

        title = self._clean_text(candidate.get("title", "")).lower()
        normalized_title = re.sub(r"\W+", "", title)
        if normalized_title:
            return f"title:{normalized_title}"
        return ""

    def _is_better_raw_candidate(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        left_quality = (
            self._safe_int(left.get("citation_count")),
            self._safe_int(left.get("year")),
            len((left.get("abstract") or "").strip()),
        )
        right_quality = (
            self._safe_int(right.get("citation_count")),
            self._safe_int(right.get("year")),
            len((right.get("abstract") or "").strip()),
        )
        return left_quality > right_quality

    def _normalize_crossref_item(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        title = self._clean_text(self._first_list_value(item.get("title")))
        if not title:
            return None

        author_names: List[str] = []
        for author in item.get("author", [])[:8]:
            given = (author.get("given") or "").strip()
            family = (author.get("family") or "").strip()
            name = (author.get("name") or "").strip()
            display = " ".join(part for part in [given, family] if part).strip() or name
            if display:
                author_names.append(self._collapse_whitespace(display))

        doi = self._normalize_doi(item.get("DOI", ""))
        source = f"Crossref (DOI: {doi})" if doi else "Crossref"
        return {
            "title": title,
            "authors": ", ".join(author_names),
            "year": self._extract_crossref_year(item),
            "venue": self._clean_text(self._first_list_value(item.get("container-title"))),
            "abstract": self._clean_text(item.get("abstract", "")),
            "source": source,
            "doi": doi,
            "citation_count": self._safe_int(item.get("is-referenced-by-count")),
        }

    def _normalize_semantic_item(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        title = self._clean_text(item.get("title", ""))
        if not title:
            return None

        authors = ", ".join(
            self._collapse_whitespace(author.get("name", "").strip())
            for author in item.get("authors", [])[:8]
            if author.get("name", "").strip()
        )

        doi = ""
        external_ids = item.get("externalIds")
        if isinstance(external_ids, dict):
            doi = self._normalize_doi(external_ids.get("DOI", ""))

        source = f"Semantic Scholar (DOI: {doi})" if doi else "Semantic Scholar"
        return {
            "title": title,
            "authors": authors,
            "year": str(item.get("year") or "").strip(),
            "venue": self._clean_text(item.get("venue", "")),
            "abstract": self._clean_text(item.get("abstract", "")),
            "source": source,
            "doi": doi,
            "citation_count": self._safe_int(item.get("citationCount")),
        }

    def _normalize_openalex_item(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        title = self._clean_text(item.get("display_name") or item.get("title", ""))
        if not title:
            return None

        author_names: List[str] = []
        for authorship in item.get("authorships", [])[:8]:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author")
            if not isinstance(author, dict):
                continue
            display_name = self._clean_text(author.get("display_name", ""))
            if display_name:
                author_names.append(display_name)

        venue = ""
        primary_location = item.get("primary_location")
        if isinstance(primary_location, dict):
            source = primary_location.get("source")
            if isinstance(source, dict):
                venue = self._clean_text(source.get("display_name", ""))
            if not venue:
                venue = self._clean_text(primary_location.get("raw_source_name", ""))

        doi = self._normalize_doi(item.get("doi", ""))
        if not doi:
            ids = item.get("ids")
            if isinstance(ids, dict):
                doi = self._normalize_doi(ids.get("doi", ""))

        openalex_id = self._clean_text(item.get("id", ""))
        source = f"OpenAlex (DOI: {doi})" if doi else "OpenAlex"
        if not doi and openalex_id:
            source = f"OpenAlex ({openalex_id})"

        return {
            "title": title,
            "authors": ", ".join(author_names),
            "year": str(item.get("publication_year") or "").strip(),
            "venue": venue,
            "abstract": self._extract_openalex_abstract(item.get("abstract_inverted_index")),
            "source": source,
            "doi": doi,
            "citation_count": self._safe_int(item.get("cited_by_count")),
        }

    def _normalize_arxiv_item(
        self,
        entry: ET.Element,
        ns: Dict[str, str],
    ) -> Dict[str, Any] | None:
        title = self._clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        if not title:
            return None

        summary = self._clean_text(
            entry.findtext("atom:summary", default="", namespaces=ns)
        )
        published = (
            entry.findtext("atom:published", default="", namespaces=ns) or ""
        ).strip()
        year = published[:4] if len(published) >= 4 else ""
        entry_id = self._clean_text(entry.findtext("atom:id", default="", namespaces=ns))
        source = f"arXiv ({entry_id})" if entry_id else "arXiv"
        authors = ", ".join(
            self._collapse_whitespace(
                author.findtext("atom:name", default="", namespaces=ns).strip()
            )
            for author in entry.findall("atom:author", ns)
            if author.findtext("atom:name", default="", namespaces=ns).strip()
        )
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "venue": "arXiv",
            "abstract": summary,
            "source": source,
            "doi": "",
            "citation_count": 0,
        }

    def _to_public_item(self, candidate: Dict[str, Any]) -> Dict[str, str]:
        return {
            "title": self._clean_text(candidate.get("title", "")),
            "authors": self._clean_text(candidate.get("authors", "")),
            "year": self._clean_text(str(candidate.get("year", ""))),
            "venue": self._clean_text(candidate.get("venue", "")),
            "abstract": self._clean_text(candidate.get("abstract", "")),
            "source": self._clean_text(candidate.get("source", "")),
        }

    def _extract_crossref_year(self, item: Dict[str, Any]) -> str:
        for key in ("issued", "published-print", "published-online", "created"):
            date_parts = item.get(key, {}).get("date-parts", [[None]])
            if date_parts and date_parts[0] and date_parts[0][0]:
                return str(date_parts[0][0])
        return ""

    def _first_list_value(self, value: Any) -> str:
        if isinstance(value, list) and value:
            return str(value[0] or "")
        if isinstance(value, str):
            return value
        return ""

    def _normalize_doi(self, value: Any) -> str:
        doi = self._clean_text(value).lower()
        if not doi:
            return ""
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
        doi = re.sub(r"^doi:\s*", "", doi)
        return doi.strip()

    def _extract_openalex_abstract(self, inverted_index: Any) -> str:
        if not isinstance(inverted_index, dict):
            return ""

        tokens_by_position: Dict[int, str] = {}
        for token, positions in inverted_index.items():
            if not isinstance(positions, list):
                continue
            clean_token = self._clean_text(token)
            if not clean_token:
                continue
            for pos in positions:
                if isinstance(pos, int) and pos >= 0 and pos not in tokens_by_position:
                    tokens_by_position[pos] = clean_token

        if not tokens_by_position:
            return ""

        ordered_tokens = [
            tokens_by_position[idx] for idx in sorted(tokens_by_position.keys())
        ]
        return self._clean_text(" ".join(ordered_tokens))

    def _request_json(
        self,
        url: str,
        *,
        headers: Dict[str, str] | None = None,
        params: Dict[str, Any] | None = None,
        retries: int = 1,
        timeout: int = 20,
    ) -> Dict[str, Any] | None:
        for attempt in range(retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=params,
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

    def _request_text(
        self,
        url: str,
        *,
        headers: Dict[str, str] | None = None,
        params: Dict[str, Any] | None = None,
        retries: int = 1,
        timeout: int = 20,
    ) -> str:
        for attempt in range(retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.text
            except requests.RequestException:
                if attempt >= retries:
                    return ""
                time.sleep(0.3 * (attempt + 1))
        return ""

    def _clean_text(self, value: Any) -> str:
        text = str(value or "")
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        return self._collapse_whitespace(text)

    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


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
