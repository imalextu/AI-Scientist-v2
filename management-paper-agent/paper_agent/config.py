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
class SemanticScholarConfig:
    api_key_env: str = "S2_API_KEY"
    max_results: int = 8


@dataclass
class CrossrefConfig:
    max_results: int = 8


@dataclass
class ArxivConfig:
    max_results: int = 8


@dataclass
class RetrievalConfig:
    enabled: bool = True
    provider: str = "crossref"
    semantic_scholar: SemanticScholarConfig = field(
        default_factory=SemanticScholarConfig
    )
    crossref: CrossrefConfig = field(default_factory=CrossrefConfig)
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)


@dataclass
class WorkflowConfig:
    temperature: float = 0.3
    max_section_tokens: int = 2000
    idea_prompt: str = "prompts/idea_generation.md"
    outline_prompt: str = "prompts/outline.md"
    paper_prompt: str = "prompts/paper_writer.md"
    outline_max_rounds: int = 4
    paper_max_rounds: int = 6
    continuation_tail_chars: int = 12000


@dataclass
class RuntimeConfig:
    save_raw_responses: bool = True


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    project_root: Path = Path(".")


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


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
    ss_raw = retrieval_raw.get("semantic_scholar", {})
    crossref_raw = retrieval_raw.get("crossref", {})
    arxiv_raw = retrieval_raw.get("arxiv", {})
    retrieval_cfg = RetrievalConfig(
        enabled=_as_bool(retrieval_raw.get("enabled"), True),
        provider=str(retrieval_raw.get("provider", "crossref")).lower(),
        semantic_scholar=SemanticScholarConfig(
            api_key_env=str(ss_raw.get("api_key_env", "S2_API_KEY")),
            max_results=_as_int(ss_raw.get("max_results"), 8),
        ),
        crossref=CrossrefConfig(
            max_results=_as_int(crossref_raw.get("max_results"), 8),
        ),
        arxiv=ArxivConfig(
            max_results=_as_int(arxiv_raw.get("max_results"), 8),
        ),
    )
    if retrieval_cfg.provider not in {"crossref", "semantic_scholar", "arxiv"}:
        retrieval_cfg.provider = "crossref"

    workflow_raw = raw.get("workflow", {})
    workflow_cfg = WorkflowConfig(
        temperature=_as_float(workflow_raw.get("temperature"), 0.3),
        max_section_tokens=_as_int(workflow_raw.get("max_section_tokens"), 2000),
        idea_prompt=str(workflow_raw.get("idea_prompt", "prompts/idea_generation.md")),
        outline_prompt=str(workflow_raw.get("outline_prompt", "prompts/outline.md")),
        paper_prompt=str(workflow_raw.get("paper_prompt", "prompts/paper_writer.md")),
        outline_max_rounds=_as_int(workflow_raw.get("outline_max_rounds"), 4),
        paper_max_rounds=_as_int(workflow_raw.get("paper_max_rounds"), 6),
        continuation_tail_chars=_as_int(
            workflow_raw.get("continuation_tail_chars"), 12000
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
        workflow=workflow_cfg,
        runtime=runtime_cfg,
        project_root=path.parent,
    )
