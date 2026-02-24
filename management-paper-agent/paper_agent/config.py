from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class LLMConfig:
    api_key_env: str = "OPENROUTER_API_KEY"
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-3.5-sonnet"
    temperature: float = 0.3
    max_tokens: int = 7000


@dataclass
class PaperConfig:
    domain: str = "management"
    audience: str = "undergraduate"
    language: str = "zh-CN"
    min_words: int = 10000
    max_words: int = 14000
    citation_style: str = "GB/T 7714-2015"
    output_dir: str = "outputs"


@dataclass
class RetrievalConfig:
    enabled: bool = True
    provider: str = "openalex"
    max_results: int = 8


@dataclass
class WebSearchConfig:
    enabled: bool = False
    provider: str = "bocha"
    max_results: int = 8
    api_key_env: str = "BOCHA_API_KEY"
    api_key: str = ""
    base_url: str = "https://api.bochaai.com/v1/web-search"
    freshness: str = "noLimit"
    summary: bool = True


@dataclass
class WorkflowConfig:
    temperature: float = 0.3
    max_section_tokens: int = 2000
    review_prompt: str = "prompts/literature_review.md"
    review_max_tokens: int = 6500
    idea_prompt: str = "prompts/idea_generation.md"
    outline_prompt: str = "prompts/outline.md"
    paper_prompt: str = "prompts/paper_writer.md"
    outline_max_rounds: int = 4
    paper_max_rounds: int = 6
    paper_audit_enabled: bool = True
    paper_audit_model: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    paper_audit_temperature: float = 0.2
    paper_audit_max_tokens: int = 4200
    paper_revision_temperature: float = 0.2
    paper_revision_max_tokens: int = 7000
    paper_revision_max_rounds: int = 4
    continuation_tail_chars: int = 12000
    run_until_stage: str = "paper"


@dataclass
class RuntimeConfig:
    save_raw_responses: bool = True


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    project_root: Path = Path(".")


def _normalize_run_until_stage(raw_value: Any) -> str:
    stage = str(raw_value or "paper").strip().lower()
    if stage not in {"literature", "review", "idea", "outline", "paper"}:
        return "paper"
    return stage


def _normalize_bocha_freshness(raw_value: Any) -> str:
    value = str(raw_value or "noLimit").strip()
    allowed = {"oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"}
    if value not in allowed:
        return "noLimit"
    return value


def _legacy_llm_block(raw: Dict[str, Any]) -> Dict[str, Any]:
    legacy = raw.get("openrouter", {})
    if not isinstance(legacy, dict):
        return {}
    return {
        "api_key_env": legacy.get("api_key_env", "OPENROUTER_API_KEY"),
        "api_key": legacy.get("api_key", ""),
        "base_url": legacy.get("base_url", "https://openrouter.ai/api/v1"),
        "model": legacy.get("model", "anthropic/claude-3.5-sonnet"),
    }


def _build_app_config(raw: Dict[str, Any], *, project_root: Path) -> AppConfig:
    if not isinstance(raw, dict):
        raise ValueError("配置文件格式错误，根节点必须是对象")

    llm_raw = raw.get("llm") or _legacy_llm_block(raw)
    llm_cfg = LLMConfig(
        api_key_env=str(llm_raw.get("api_key_env", "OPENROUTER_API_KEY")),
        api_key=str(llm_raw.get("api_key", "")),
        base_url=str(llm_raw.get("base_url", "https://openrouter.ai/api/v1")),
        model=str(llm_raw.get("model", "anthropic/claude-3.5-sonnet")),
        temperature=_as_float(llm_raw.get("temperature"), 0.3),
        max_tokens=_as_int(llm_raw.get("max_tokens"), 7000),
    )

    paper_raw = raw.get("paper", {})
    paper_cfg = PaperConfig(
        domain=str(paper_raw.get("domain", "management")),
        audience=str(paper_raw.get("audience", "undergraduate")),
        language=str(paper_raw.get("language", "zh-CN")),
        min_words=_as_int(paper_raw.get("min_words"), 10000),
        max_words=_as_int(paper_raw.get("max_words"), 14000),
        citation_style=str(paper_raw.get("citation_style", "GB/T 7714-2015")),
        output_dir=str(paper_raw.get("output_dir", "outputs")),
    )

    retrieval_raw = raw.get("retrieval", {})
    provider_raw = str(retrieval_raw.get("provider", "openalex")).lower()
    provider_block = retrieval_raw.get(provider_raw, {})
    provider_max_results = (
        provider_block.get("max_results")
        if isinstance(provider_block, dict)
        else None
    )
    retrieval_cfg = RetrievalConfig(
        enabled=_as_bool(retrieval_raw.get("enabled"), True),
        provider=provider_raw,
        max_results=max(
            1,
            _as_int(
                retrieval_raw.get(
                    "max_results",
                    provider_max_results,
                ),
                8,
            ),
        ),
    )
    if retrieval_cfg.provider not in {
        "crossref",
        "semantic_scholar",
        "arxiv",
        "openalex",
    }:
        retrieval_cfg.provider = "openalex"

    web_search_raw = raw.get("web_search", {})
    if not isinstance(web_search_raw, dict):
        web_search_raw = {}
    web_search_provider = str(web_search_raw.get("provider", "bocha")).strip().lower()
    if web_search_provider not in {"bocha"}:
        web_search_provider = "bocha"
    web_search_cfg = WebSearchConfig(
        enabled=_as_bool(web_search_raw.get("enabled"), False),
        provider=web_search_provider,
        max_results=max(1, _as_int(web_search_raw.get("max_results"), 8)),
        api_key_env=str(web_search_raw.get("api_key_env", "BOCHA_API_KEY")),
        api_key=str(web_search_raw.get("api_key", "")),
        base_url=str(
            web_search_raw.get("base_url", "https://api.bochaai.com/v1/web-search")
        ),
        freshness=_normalize_bocha_freshness(
            web_search_raw.get("freshness", "noLimit")
        ),
        summary=_as_bool(web_search_raw.get("summary"), True),
    )

    workflow_raw = raw.get("workflow", {})
    workflow_cfg = WorkflowConfig(
        temperature=_as_float(workflow_raw.get("temperature"), 0.3),
        max_section_tokens=_as_int(workflow_raw.get("max_section_tokens"), 2000),
        review_prompt=str(workflow_raw.get("review_prompt", "prompts/literature_review.md")),
        review_max_tokens=max(
            1800,
            _as_int(workflow_raw.get("review_max_tokens"), 6500),
        ),
        idea_prompt=str(workflow_raw.get("idea_prompt", "prompts/idea_generation.md")),
        outline_prompt=str(workflow_raw.get("outline_prompt", "prompts/outline.md")),
        paper_prompt=str(workflow_raw.get("paper_prompt", "prompts/paper_writer.md")),
        outline_max_rounds=_as_int(workflow_raw.get("outline_max_rounds"), 4),
        paper_max_rounds=_as_int(workflow_raw.get("paper_max_rounds"), 6),
        paper_audit_enabled=_as_bool(
            workflow_raw.get("paper_audit_enabled"),
            True,
        ),
        paper_audit_model=(
            str(
                workflow_raw.get(
                    "paper_audit_model",
                    "Qwen/Qwen3-VL-235B-A22B-Instruct",
                )
            ).strip()
            or "Qwen/Qwen3-VL-235B-A22B-Instruct"
        ),
        paper_audit_temperature=_as_float(
            workflow_raw.get("paper_audit_temperature"),
            0.2,
        ),
        paper_audit_max_tokens=max(
            1200,
            _as_int(workflow_raw.get("paper_audit_max_tokens"), 4200),
        ),
        paper_revision_temperature=_as_float(
            workflow_raw.get("paper_revision_temperature"),
            0.2,
        ),
        paper_revision_max_tokens=max(
            1800,
            _as_int(workflow_raw.get("paper_revision_max_tokens"), 7000),
        ),
        paper_revision_max_rounds=max(
            1,
            _as_int(workflow_raw.get("paper_revision_max_rounds"), 4),
        ),
        continuation_tail_chars=_as_int(
            workflow_raw.get("continuation_tail_chars"), 12000
        ),
        run_until_stage=_normalize_run_until_stage(
            workflow_raw.get("run_until_stage", "paper")
        ),
    )

    runtime_raw = raw.get("runtime", {})
    runtime_cfg = RuntimeConfig(
        save_raw_responses=_as_bool(runtime_raw.get("save_raw_responses"), True),
    )

    return AppConfig(
        llm=llm_cfg,
        paper=paper_cfg,
        retrieval=retrieval_cfg,
        web_search=web_search_cfg,
        workflow=workflow_cfg,
        runtime=runtime_cfg,
        project_root=project_root.resolve(),
    )


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _build_app_config(raw, project_root=path.parent)


def load_config_from_text(yaml_text: str, *, project_root: str | Path) -> AppConfig:
    raw = yaml.safe_load(yaml_text) or {}
    base = Path(project_root).expanduser().resolve()
    return _build_app_config(raw, project_root=base)
