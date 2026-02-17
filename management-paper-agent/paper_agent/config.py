from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
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
class WorkflowConfig:
    temperature: float = 0.3
    max_section_tokens: int = 2000
    idea_prompt: str = "prompts/idea_generation.md"
    outline_prompt: str = "prompts/outline.md"
    paper_prompt: str = "prompts/paper_writer.md"
    outline_max_rounds: int = 4
    paper_max_rounds: int = 6
    continuation_tail_chars: int = 12000
    research_enabled: bool = True
    research_prompt: str = "prompts/research_iteration.md"
    research_eval_prompt: str = "prompts/research_evaluator.md"
    research_max_tokens: int = 2200
    research_stages: list["ResearchStageConfig"] = field(
        default_factory=lambda: _default_research_stages()
    )
    run_until_stage: str = "paper"


@dataclass
class ResearchStageConfig:
    key: str
    name: str
    goal: str
    max_iterations: int = 2
    branching_factor: int = 2
    keep_top_k: int = 2


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


def _default_research_stages() -> list[ResearchStageConfig]:
    return [
        ResearchStageConfig(
            key="problem_framing",
            name="问题界定",
            goal=(
                "明确研究问题、研究对象、边界条件与核心变量，确保题目可落地且适合本科论文。"
            ),
            max_iterations=2,
            branching_factor=2,
            keep_top_k=2,
        ),
        ResearchStageConfig(
            key="method_design",
            name="方法设计",
            goal=(
                "形成可执行的方法路线，明确样本与数据来源，保证在本科条件下可实施。"
            ),
            max_iterations=2,
            branching_factor=2,
            keep_top_k=2,
        ),
        ResearchStageConfig(
            key="analysis_and_risk",
            name="分析与风险",
            goal=(
                "完善分析步骤、预期发现与潜在风险控制，形成可直接用于写作的研究方案。"
            ),
            max_iterations=2,
            branching_factor=2,
            keep_top_k=2,
        ),
    ]


def _sanitize_stage_key(raw_key: str, fallback_idx: int) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", raw_key.strip().lower()).strip("_")
    if key:
        return key
    return f"stage_{fallback_idx}"


def _parse_research_stages(raw_stages: Any) -> list[ResearchStageConfig]:
    if not isinstance(raw_stages, list) or not raw_stages:
        return _default_research_stages()

    parsed: list[ResearchStageConfig] = []
    for idx, item in enumerate(raw_stages, start=1):
        if not isinstance(item, dict):
            continue
        key_raw = str(item.get("key") or item.get("name") or f"stage_{idx}")
        key = _sanitize_stage_key(key_raw, idx)
        name = str(item.get("name") or key)
        goal = str(item.get("goal") or item.get("goals") or "").strip()
        if not goal:
            goal = f"完成{name}相关研究阶段。"

        parsed.append(
            ResearchStageConfig(
                key=key,
                name=name,
                goal=goal,
                max_iterations=max(1, _as_int(item.get("max_iterations"), 2)),
                branching_factor=max(1, _as_int(item.get("branching_factor"), 2)),
                keep_top_k=max(1, _as_int(item.get("keep_top_k"), 2)),
            )
        )

    if not parsed:
        return _default_research_stages()
    return parsed


def _normalize_run_until_stage(raw_value: Any) -> str:
    stage = str(raw_value or "paper").strip().lower()
    if stage not in {"literature", "research", "idea", "outline", "paper"}:
        return "paper"
    return stage


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

    workflow_raw = raw.get("workflow", {})
    research_stages = _parse_research_stages(workflow_raw.get("research_stages"))
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
        research_enabled=_as_bool(workflow_raw.get("research_enabled"), True),
        research_prompt=str(
            workflow_raw.get("research_prompt", "prompts/research_iteration.md")
        ),
        research_eval_prompt=str(
            workflow_raw.get("research_eval_prompt", "prompts/research_evaluator.md")
        ),
        research_max_tokens=max(
            1200, _as_int(workflow_raw.get("research_max_tokens"), 2200)
        ),
        research_stages=research_stages,
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
