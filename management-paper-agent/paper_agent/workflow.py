from __future__ import annotations

from dataclasses import dataclass
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from .config import AppConfig, ResearchStageConfig
from .llm_client import LLMClient, RequestCancelledError
from .retrieval import LiteratureRetriever, format_literature_context
from .utils import (
    ensure_dir,
    extract_json_payload,
    now_stamp,
    read_text,
    render_template,
    slugify,
    write_json,
    write_text,
)

IDEA_SYSTEM_PROMPT = """
你是管理学本科论文导师。你的任务是帮助学生生成“可执行、可完成、符合本科规范”的研究选题设计。
要求：
1. 研究问题明确，避免空泛。
2. 方法可落地（问卷、访谈、案例、二手数据等）。
3. 给出章节设计与潜在风险。
4. 按要求输出严格 JSON，不要额外解释。
""".strip()

OUTLINE_SYSTEM_PROMPT = """
你是管理学论文写作导师。请根据选题 JSON 产出本科毕业论文大纲。
要求：
1. 每章结构清晰，逻辑完整，内容不过度学术化。
2. 每节说明核心论点与证据需求。
3. 输出严格 JSON，不要附加说明文字。
""".strip()

PAPER_SYSTEM_PROMPT = """
你是管理学本科论文代写助手（学术规范模式）。
要求：
1. 用中文输出完整论文初稿。
2. 语言正式、结构完整、论证清晰。
3. 引文与参考文献格式尽量贴近 GB/T 7714-2015。
4. 严禁捏造具体统计结果；如无真实数据，请明确写“示例性分析/待实证验证”。
""".strip()

RESEARCH_SYSTEM_PROMPT = """
你是管理学研究方法导师，负责在写作前进行“研究方案树搜索”。
你每次只生成一个候选研究节点，要求可执行、边界清晰、可用于后续写作。
输出必须是严格 JSON，不要输出解释。
""".strip()

RESEARCH_EVAL_SYSTEM_PROMPT = """
你是研究方案评审人。你将根据阶段目标评估候选研究节点质量，并给出量化评分。
评分范围 0-100，输出必须是严格 JSON，不要输出解释。
""".strip()

WorkflowEventHandler = Callable[[str, Dict[str, Any]], None]


class WorkflowCancelledError(RuntimeError):
    pass


@dataclass
class ResearchNode:
    node_id: str
    parent_id: str | None
    stage_key: str
    stage_name: str
    stage_index: int
    iteration: int
    branch_index: int
    score: float
    proposal: Dict[str, Any]
    evaluation: Dict[str, Any]


class ThesisWorkflow:
    def __init__(
        self,
        config: AppConfig,
        *,
        event_handler: WorkflowEventHandler | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ):
        self.config = config
        self.project_root = Path(config.project_root).resolve()
        self.client = LLMClient(config.llm)
        self.retriever = LiteratureRetriever(config.retrieval)
        self.event_handler = event_handler
        self.cancel_checker = cancel_checker

    def run(self, topic_text: str, forced_title: str | None = None) -> Path:
        stage_order = {
            "literature": 1,
            "research": 2,
            "idea": 3,
            "outline": 4,
            "paper": 5,
        }
        target_stage = str(self.config.workflow.run_until_stage).strip().lower() or "paper"
        if target_stage not in stage_order:
            target_stage = "paper"
        if target_stage == "research" and not self.config.workflow.research_enabled:
            raise ValueError(
                "workflow.run_until_stage=research 但 workflow.research_enabled=false。"
            )

        self._ensure_not_cancelled()
        output_root = ensure_dir(self.project_root / self.config.paper.output_dir)
        run_id = now_stamp()
        run_dir = output_root / f"{run_id}_management_thesis"
        ensure_dir(run_dir)
        self._ensure_not_cancelled()
        self._emit(
            "workflow_started",
            run_id=run_id,
            output_root=str(output_root),
            model=self.config.llm.model,
        )

        write_text(run_dir / "00_topic.md", topic_text.strip())
        self._ensure_not_cancelled()
        self._emit(
            "topic_saved",
            path=str(run_dir / "00_topic.md"),
            topic=topic_text.strip(),
        )

        literature_items: list[dict[str, str]] = []
        literature_context = "暂无可用文献线索。"

        research_data: Dict[str, Any] = {
            "enabled": False,
            "reason": "research_enabled=false",
            "stages": [],
            "best_path": [],
            "best_score": None,
        }
        research_context = "研究树搜索阶段已关闭。"
        research_raw = ""
        research_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }
        idea_data: Dict[str, Any] = {}
        idea_raw = ""
        idea_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }
        outline_data: Dict[str, Any] = {}
        outline_raw = ""
        outline_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }
        thesis_md = ""
        thesis_raw = ""
        paper_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }
        executed_stages: list[str] = []

        if stage_order["literature"] <= stage_order[target_stage]:
            self._emit(
                "literature_started",
                enabled=self.config.retrieval.enabled,
                provider=self.config.retrieval.provider,
            )
            self._ensure_not_cancelled()
            literature_items, literature_context = self._collect_literature(topic_text)
            write_json(run_dir / "00_literature.json", literature_items)
            self._ensure_not_cancelled()
            self._emit(
                "literature_completed",
                count=len(literature_items),
                path=str(run_dir / "00_literature.json"),
                items=literature_items,
            )
            executed_stages.append("literature")

        if (
            self.config.workflow.research_enabled
            and stage_order["research"] <= stage_order[target_stage]
        ):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="research")
            (
                research_data,
                research_context,
                research_raw,
                research_usage,
            ) = self._run_research_tree_stage(
                topic_text=topic_text,
                literature_context=literature_context,
            )
            write_json(run_dir / "00_research_tree.json", research_data)
            self._ensure_not_cancelled()
            self._emit(
                "stage_completed",
                stage="research",
                path=str(run_dir / "00_research_tree.json"),
                content=json.dumps(research_data, ensure_ascii=False, indent=2),
                usage=research_usage,
            )
            executed_stages.append("research")

        if stage_order["idea"] <= stage_order[target_stage]:
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="idea")
            idea_data, idea_raw, idea_usage = self._run_idea_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                research_context=research_context,
                forced_title=forced_title,
            )
            write_json(run_dir / "01_idea.json", idea_data)
            self._ensure_not_cancelled()
            self._emit(
                "stage_completed",
                stage="idea",
                path=str(run_dir / "01_idea.json"),
                content=json.dumps(idea_data, ensure_ascii=False, indent=2),
                usage=idea_usage,
            )
            executed_stages.append("idea")

        if stage_order["outline"] <= stage_order[target_stage]:
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="outline")
            outline_data, outline_raw, outline_usage = self._run_outline_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                research_context=research_context,
                idea_data=idea_data,
            )
            write_json(run_dir / "02_outline.json", outline_data)
            self._ensure_not_cancelled()
            self._emit(
                "stage_completed",
                stage="outline",
                path=str(run_dir / "02_outline.json"),
                content=json.dumps(outline_data, ensure_ascii=False, indent=2),
                usage=outline_usage,
            )
            executed_stages.append("outline")

        if stage_order["paper"] <= stage_order[target_stage]:
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="paper")
            thesis_md, thesis_raw, paper_usage = self._run_paper_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                research_context=research_context,
                idea_data=idea_data,
                outline_data=outline_data,
            )
            write_text(run_dir / "03_thesis.md", thesis_md)
            self._ensure_not_cancelled()
            self._emit(
                "stage_completed",
                stage="paper",
                path=str(run_dir / "03_thesis.md"),
                content=thesis_md,
                usage=paper_usage,
            )
            executed_stages.append("paper")

        if self.config.runtime.save_raw_responses:
            self._ensure_not_cancelled()
            if "research" in executed_stages:
                write_text(run_dir / "raw_00_research.txt", research_raw)
            if "idea" in executed_stages:
                write_text(run_dir / "raw_01_idea.txt", idea_raw)
            if "outline" in executed_stages:
                write_text(run_dir / "raw_02_outline.txt", outline_raw)
            if "paper" in executed_stages:
                write_text(run_dir / "raw_03_paper.txt", thesis_raw)

        self._ensure_not_cancelled()
        metadata = self._build_metadata(
            run_id=run_id,
            topic_text=topic_text,
            run_until_stage=target_stage,
            executed_stages=executed_stages,
            research_data=research_data,
            idea_data=idea_data,
            outline_data=outline_data,
            usage={
                "research": research_usage,
                "idea": idea_usage,
                "outline": outline_usage,
                "paper": paper_usage,
            },
        )
        write_json(run_dir / "run_metadata.json", metadata)
        self._emit("metadata_saved", path=str(run_dir / "run_metadata.json"))

        self._ensure_not_cancelled()
        final_title = ""
        if "idea" in executed_stages:
            final_title = forced_title or self._get_title(idea_data)
        if final_title:
            better_dir = self._resolve_unique_dir(
                output_root / f"{run_id}_{slugify(final_title)}"
            )
            if better_dir != run_dir:
                run_dir.rename(better_dir)
                run_dir = better_dir
        self._emit(
            "workflow_completed",
            run_dir=str(run_dir),
            title=final_title,
            usage=metadata.get("usage", {}),
        )
        return run_dir

    def _collect_literature(self, topic_text: str) -> Tuple[list[dict[str, str]], str]:
        self._ensure_not_cancelled()
        if not self.config.retrieval.enabled:
            return [], "文献检索已关闭。"
        items = self.retriever.search(topic_text)
        self._ensure_not_cancelled()
        return items, format_literature_context(items)

    def _run_research_tree_stage(
        self,
        *,
        topic_text: str,
        literature_context: str,
    ) -> Tuple[Dict[str, Any], str, str, Dict[str, int]]:
        prompt_template = read_text(
            self.project_root / self.config.workflow.research_prompt
        )
        eval_template = read_text(
            self.project_root / self.config.workflow.research_eval_prompt
        )
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }
        raw_chunks: list[str] = []
        all_nodes: dict[str, ResearchNode] = {}
        parent_pool: list[ResearchNode] = []
        stage_summaries: list[dict[str, Any]] = []
        node_counter = 0
        global_round = 0

        for stage_idx, stage_cfg in enumerate(
            self.config.workflow.research_stages, start=1
        ):
            self._ensure_not_cancelled()
            stage_tag = f"research:{stage_cfg.key}"
            self._emit("stage_started", stage=stage_tag)

            stage_nodes: list[ResearchNode] = []
            stage_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            iteration_parents = parent_pool[:] if parent_pool else [None]

            for iteration in range(1, stage_cfg.max_iterations + 1):
                self._ensure_not_cancelled()
                global_round += 1
                self._emit(
                    "stage_round_started",
                    stage=stage_tag,
                    round=iteration,
                )

                for branch_idx in range(1, stage_cfg.branching_factor + 1):
                    parent = iteration_parents[
                        (branch_idx - 1) % len(iteration_parents)
                    ]
                    proposal, proposal_raw, proposal_usage = (
                        self._run_research_iteration(
                            prompt_template=prompt_template,
                            topic_text=topic_text,
                            literature_context=literature_context,
                            stage_cfg=stage_cfg,
                            stage_idx=stage_idx,
                            iteration=iteration,
                            branch_idx=branch_idx,
                            parent=parent,
                            stage_nodes=stage_nodes,
                            emit_round=global_round,
                        )
                    )
                    self._accumulate_usage(usage_total, proposal_usage)
                    self._accumulate_usage(stage_usage, proposal_usage)
                    raw_chunks.append(
                        f"[{stage_tag}][iter={iteration}][branch={branch_idx}]"
                        f"\n{proposal_raw.strip()}"
                    )

                    evaluation, eval_raw, eval_usage = self._evaluate_research_iteration(
                        eval_template=eval_template,
                        topic_text=topic_text,
                        literature_context=literature_context,
                        stage_cfg=stage_cfg,
                        stage_idx=stage_idx,
                        iteration=iteration,
                        proposal=proposal,
                        parent=parent,
                    )
                    self._accumulate_usage(usage_total, eval_usage)
                    self._accumulate_usage(stage_usage, eval_usage)
                    raw_chunks.append(
                        f"[{stage_tag}][iter={iteration}][branch={branch_idx}][evaluation]"
                        f"\n{eval_raw.strip()}"
                    )

                    node_counter += 1
                    node = ResearchNode(
                        node_id=f"{stage_cfg.key}_n{node_counter}",
                        parent_id=parent.node_id if parent else None,
                        stage_key=stage_cfg.key,
                        stage_name=stage_cfg.name,
                        stage_index=stage_idx,
                        iteration=iteration,
                        branch_index=branch_idx,
                        score=self._normalize_score(evaluation.get("score")),
                        proposal=proposal,
                        evaluation=evaluation,
                    )
                    stage_nodes.append(node)
                    all_nodes[node.node_id] = node

                stage_ranked = sorted(stage_nodes, key=lambda item: item.score, reverse=True)
                iteration_parents = stage_ranked[: stage_cfg.keep_top_k] or [None]
                self._emit(
                    "stage_round_completed",
                    stage=stage_tag,
                    round=iteration,
                    finish_reason="completed",
                )

            parent_pool = sorted(stage_nodes, key=lambda item: item.score, reverse=True)[
                : stage_cfg.keep_top_k
            ]
            stage_usage["rounds"] = stage_cfg.max_iterations
            stage_summary = self._summarize_research_stage(
                stage_cfg=stage_cfg,
                stage_idx=stage_idx,
                stage_nodes=stage_nodes,
                stage_usage=stage_usage,
            )
            stage_summaries.append(stage_summary)
            self._emit(
                "stage_completed",
                stage=stage_tag,
                content=json.dumps(stage_summary, ensure_ascii=False, indent=2),
                usage=stage_usage,
            )

        best_node: ResearchNode | None = None
        if parent_pool:
            best_node = max(parent_pool, key=lambda item: item.score)
        elif all_nodes:
            best_node = max(all_nodes.values(), key=lambda item: item.score)

        best_path = self._build_best_research_path(
            all_nodes=all_nodes,
            best_node_id=best_node.node_id if best_node else None,
        )
        report: Dict[str, Any] = {
            "enabled": True,
            "stages": stage_summaries,
            "best_score": best_node.score if best_node else None,
            "best_node_id": best_node.node_id if best_node else None,
            "best_path": best_path,
            "total_nodes": len(all_nodes),
        }
        usage_total["rounds"] = global_round
        research_context = self._format_research_context(report)
        raw_joined = "\n\n\n<!-- RESEARCH ROUND -->\n\n".join(raw_chunks).strip()
        return report, research_context, raw_joined, usage_total

    def _run_research_iteration(
        self,
        *,
        prompt_template: str,
        topic_text: str,
        literature_context: str,
        stage_cfg: ResearchStageConfig,
        stage_idx: int,
        iteration: int,
        branch_idx: int,
        parent: ResearchNode | None,
        stage_nodes: list[ResearchNode],
        emit_round: int,
    ) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
        parent_context = (
            json.dumps(self._serialize_research_node(parent), ensure_ascii=False, indent=2)
            if parent
            else "无父节点（当前阶段起始分支）"
        )
        local_best = sorted(stage_nodes, key=lambda item: item.score, reverse=True)[:3]
        tree_snapshot = (
            json.dumps(
                [self._serialize_research_node(node) for node in local_best],
                ensure_ascii=False,
                indent=2,
            )
            if local_best
            else "当前阶段暂无已评分节点。"
        )
        user_prompt = render_template(
            prompt_template,
            {
                "topic": topic_text,
                "stage_key": stage_cfg.key,
                "stage_name": stage_cfg.name,
                "stage_goal": stage_cfg.goal,
                "stage_index": stage_idx,
                "stage_total": len(self.config.workflow.research_stages),
                "iteration": iteration,
                "branch_index": branch_idx,
                "literature_context": literature_context,
                "parent_node": parent_context,
                "current_tree_snapshot": tree_snapshot,
            },
        )
        raw_text, usage = self._client_complete(
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.workflow.temperature,
            max_tokens=min(
                self.config.llm.max_tokens, self.config.workflow.research_max_tokens
            ),
            on_delta=lambda piece: self._emit_stage_delta("research", emit_round, piece),
        )
        parsed = extract_json_payload(raw_text)
        if not isinstance(parsed, dict):
            parsed = {"raw_response": raw_text}
        return parsed, raw_text, usage

    def _evaluate_research_iteration(
        self,
        *,
        eval_template: str,
        topic_text: str,
        literature_context: str,
        stage_cfg: ResearchStageConfig,
        stage_idx: int,
        iteration: int,
        proposal: Dict[str, Any],
        parent: ResearchNode | None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
        parent_context = (
            json.dumps(self._serialize_research_node(parent), ensure_ascii=False, indent=2)
            if parent
            else "无父节点"
        )
        user_prompt = render_template(
            eval_template,
            {
                "topic": topic_text,
                "stage_key": stage_cfg.key,
                "stage_name": stage_cfg.name,
                "stage_goal": stage_cfg.goal,
                "stage_index": stage_idx,
                "iteration": iteration,
                "literature_context": literature_context,
                "proposal_json": json.dumps(proposal, ensure_ascii=False, indent=2),
                "parent_node": parent_context,
            },
        )
        raw_text, usage = self._client_complete(
            system_prompt=RESEARCH_EVAL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=1200,
            on_delta=None,
        )
        parsed = extract_json_payload(raw_text)
        if not isinstance(parsed, dict):
            parsed = {"raw_response": raw_text}
        parsed["score"] = self._normalize_score(parsed.get("score"))
        return parsed, raw_text, usage

    def _summarize_research_stage(
        self,
        *,
        stage_cfg: ResearchStageConfig,
        stage_idx: int,
        stage_nodes: list[ResearchNode],
        stage_usage: Dict[str, int],
    ) -> Dict[str, Any]:
        ranked = sorted(stage_nodes, key=lambda item: item.score, reverse=True)
        top_nodes = ranked[: stage_cfg.keep_top_k]
        return {
            "stage_index": stage_idx,
            "stage_key": stage_cfg.key,
            "stage_name": stage_cfg.name,
            "stage_goal": stage_cfg.goal,
            "max_iterations": stage_cfg.max_iterations,
            "branching_factor": stage_cfg.branching_factor,
            "keep_top_k": stage_cfg.keep_top_k,
            "nodes_explored": len(stage_nodes),
            "best_score": top_nodes[0].score if top_nodes else None,
            "top_nodes": [self._serialize_research_node(node) for node in top_nodes],
            "usage": dict(stage_usage),
        }

    def _serialize_research_node(self, node: ResearchNode | None) -> Dict[str, Any]:
        if node is None:
            return {}
        return {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "stage_key": node.stage_key,
            "stage_name": node.stage_name,
            "stage_index": node.stage_index,
            "iteration": node.iteration,
            "branch_index": node.branch_index,
            "score": round(node.score, 2),
            "proposal": node.proposal,
            "evaluation": node.evaluation,
        }

    def _build_best_research_path(
        self,
        *,
        all_nodes: Dict[str, ResearchNode],
        best_node_id: str | None,
    ) -> list[Dict[str, Any]]:
        if not best_node_id:
            return []
        path: list[Dict[str, Any]] = []
        cursor = best_node_id
        visited: set[str] = set()
        while cursor and cursor not in visited:
            visited.add(cursor)
            node = all_nodes.get(cursor)
            if not node:
                break
            path.append(self._serialize_research_node(node))
            cursor = node.parent_id
        path.reverse()
        return path

    def _format_research_context(self, research_data: Dict[str, Any]) -> str:
        if not research_data.get("enabled"):
            return "研究树搜索未启用。"

        lines: list[str] = [
            "以下为研究树搜索阶段形成的写作前研究结论：",
        ]
        for stage in research_data.get("stages", []):
            if not isinstance(stage, dict):
                continue
            stage_name = str(stage.get("stage_name") or stage.get("stage_key") or "阶段")
            goal = str(stage.get("stage_goal") or "")
            best_score = stage.get("best_score")
            lines.append(f"[{stage_name}] 目标：{goal}")
            lines.append(f"[{stage_name}] 最佳评分：{best_score}")
            top_nodes = stage.get("top_nodes") or []
            if isinstance(top_nodes, list) and top_nodes:
                best_node = top_nodes[0]
                if isinstance(best_node, dict):
                    proposal = best_node.get("proposal")
                    if isinstance(proposal, dict):
                        candidate_title = proposal.get("candidate_title", "")
                        question = proposal.get("research_question", "")
                        method_design = proposal.get("method_design")
                        if isinstance(method_design, dict):
                            approach = method_design.get("approach", "")
                            sample = method_design.get("sample_plan", "")
                        else:
                            approach = ""
                            sample = ""
                        lines.append(f"[{stage_name}] 最优方案标题：{candidate_title}")
                        lines.append(f"[{stage_name}] 核心问题：{question}")
                        lines.append(f"[{stage_name}] 方法路径：{approach}；样本：{sample}")

        best_path = research_data.get("best_path") or []
        if isinstance(best_path, list) and best_path:
            lines.append("最佳路径摘要：")
            for idx, node in enumerate(best_path, start=1):
                if not isinstance(node, dict):
                    continue
                stage_name = str(node.get("stage_name") or node.get("stage_key") or "阶段")
                score = node.get("score")
                proposal = node.get("proposal")
                if isinstance(proposal, dict):
                    candidate_title = str(proposal.get("candidate_title") or "")
                    question = str(proposal.get("research_question") or "")
                    proposal_text = f"标题={candidate_title}; 问题={question}"
                else:
                    proposal_text = str(proposal or "")
                lines.append(
                    f"路径节点{idx}（{stage_name}, score={score}）：{proposal_text}"
                )
        return "\n".join(lines)

    def _normalize_score(self, raw_score: Any) -> float:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return 50.0
        return max(0.0, min(100.0, score))

    def _run_idea_stage(
        self,
        *,
        topic_text: str,
        literature_context: str,
        research_context: str,
        forced_title: str | None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
        prompt_template = read_text(self.project_root / self.config.workflow.idea_prompt)
        user_prompt = render_template(
            prompt_template,
            {
                "topic": topic_text,
                "domain": self.config.paper.domain,
                "audience": self.config.paper.audience,
                "language": self.config.paper.language,
                "min_words": self.config.paper.min_words,
                "max_words": self.config.paper.max_words,
                "citation_style": self.config.paper.citation_style,
                "literature_context": literature_context,
                "research_context": research_context,
            },
        )
        raw_text, usage = self._client_complete(
            system_prompt=IDEA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.workflow.temperature,
            max_tokens=max(self.config.llm.max_tokens // 2, 1800),
            on_delta=lambda piece: self._emit_stage_delta("idea", 1, piece),
        )
        parsed = extract_json_payload(raw_text)
        if not isinstance(parsed, dict):
            parsed = {"raw_response": raw_text}

        if forced_title:
            parsed["thesis_title_cn"] = forced_title
        return parsed, raw_text, usage

    def _run_outline_stage(
        self,
        *,
        topic_text: str,
        literature_context: str,
        research_context: str,
        idea_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
        prompt_template = read_text(
            self.project_root / self.config.workflow.outline_prompt
        )
        user_prompt = render_template(
            prompt_template,
            {
                "topic": topic_text,
                "idea_json": json.dumps(idea_data, ensure_ascii=False, indent=2),
                "literature_context": literature_context,
                "research_context": research_context,
            },
        )
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        raw_chunks: list[str] = []
        merged_text = ""

        max_rounds = max(1, self.config.workflow.outline_max_rounds)
        current_prompt = user_prompt
        parsed_json: Any = None

        for round_idx in range(max_rounds):
            self._ensure_not_cancelled()
            self._emit("stage_round_started", stage="outline", round=round_idx + 1)
            raw_text, usage, finish_reason = self._client_complete_with_meta(
                system_prompt=OUTLINE_SYSTEM_PROMPT,
                user_prompt=current_prompt,
                temperature=self.config.workflow.temperature,
                max_tokens=max(self.config.llm.max_tokens // 2, 2200),
                on_delta=lambda piece, current_round=round_idx + 1: self._emit_stage_delta(
                    "outline",
                    current_round,
                    piece,
                ),
            )
            raw_chunks.append(raw_text)
            self._accumulate_usage(usage_total, usage)
            self._emit(
                "stage_round_completed",
                stage="outline",
                round=round_idx + 1,
                finish_reason=finish_reason,
            )

            clean_chunk = self._strip_markdown_fence(raw_text)
            merged_text = self._append_with_overlap(merged_text, clean_chunk)
            parsed_json = extract_json_payload(merged_text)

            if self._outline_is_complete(parsed_json) and finish_reason != "length":
                break

            if round_idx >= max_rounds - 1:
                break

            current_prompt = self._build_outline_continuation_prompt(merged_text)

        final_parsed = extract_json_payload(merged_text)
        if not self._outline_is_complete(final_parsed):
            repaired_text, repaired_usage = self._repair_json_object(
                merged_text,
                purpose="论文大纲 JSON",
                on_delta=lambda piece: self._emit(
                    "stage_delta",
                    stage="outline",
                    round=max_rounds + 1,
                    text=piece,
                ),
            )
            if repaired_text:
                raw_chunks.append(repaired_text)
                self._accumulate_usage(usage_total, repaired_usage)
                repaired_parsed = extract_json_payload(repaired_text)
                if self._outline_is_complete(repaired_parsed):
                    final_parsed = repaired_parsed

        if not isinstance(final_parsed, dict):
            final_parsed = {"raw_response": merged_text}

        usage_total["rounds"] = len(raw_chunks)
        raw_text_joined = "\n\n\n<!-- CONTINUATION ROUND -->\n\n".join(raw_chunks)
        return final_parsed, raw_text_joined.strip(), usage_total

    def _run_paper_stage(
        self,
        *,
        topic_text: str,
        literature_context: str,
        research_context: str,
        idea_data: Dict[str, Any],
        outline_data: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, int]]:
        prompt_template = read_text(self.project_root / self.config.workflow.paper_prompt)
        user_prompt = render_template(
            prompt_template,
            {
                "topic": topic_text,
                "language": self.config.paper.language,
                "min_words": self.config.paper.min_words,
                "max_words": self.config.paper.max_words,
                "citation_style": self.config.paper.citation_style,
                "idea_json": json.dumps(idea_data, ensure_ascii=False, indent=2),
                "outline_json": json.dumps(outline_data, ensure_ascii=False, indent=2),
                "literature_context": literature_context,
                "research_context": research_context,
            },
        )

        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        raw_chunks: list[str] = []
        merged_paper = ""

        max_rounds = max(1, self.config.workflow.paper_max_rounds)
        current_prompt = user_prompt

        for round_idx in range(max_rounds):
            self._ensure_not_cancelled()
            self._emit("stage_round_started", stage="paper", round=round_idx + 1)
            raw_text, usage, finish_reason = self._client_complete_with_meta(
                system_prompt=PAPER_SYSTEM_PROMPT,
                user_prompt=current_prompt,
                temperature=self.config.workflow.temperature,
                max_tokens=self.config.llm.max_tokens,
                on_delta=lambda piece, current_round=round_idx + 1: self._emit_stage_delta(
                    "paper",
                    current_round,
                    piece,
                ),
            )
            raw_chunks.append(raw_text)
            self._accumulate_usage(usage_total, usage)
            self._emit(
                "stage_round_completed",
                stage="paper",
                round=round_idx + 1,
                finish_reason=finish_reason,
            )

            clean_chunk = self._strip_markdown_fence(raw_text)
            merged_paper = self._append_with_overlap(merged_paper, clean_chunk)

            if not self._needs_continuation(
                full_text=merged_paper,
                finish_reason=finish_reason,
                round_idx=round_idx,
                max_rounds=max_rounds,
            ):
                break

            current_prompt = self._build_continuation_prompt(merged_paper)

        usage_total["rounds"] = len(raw_chunks)
        raw_text_joined = "\n\n\n<!-- CONTINUATION ROUND -->\n\n".join(raw_chunks)
        return merged_paper.strip(), raw_text_joined.strip(), usage_total

    def _build_metadata(
        self,
        *,
        run_id: str,
        topic_text: str,
        run_until_stage: str,
        executed_stages: list[str],
        research_data: Dict[str, Any],
        idea_data: Dict[str, Any],
        outline_data: Dict[str, Any],
        usage: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "topic": topic_text,
            "model": self.config.llm.model,
            "paper_constraints": {
                "language": self.config.paper.language,
                "min_words": self.config.paper.min_words,
                "max_words": self.config.paper.max_words,
                "citation_style": self.config.paper.citation_style,
            },
            "retrieval": {
                "enabled": self.config.retrieval.enabled,
                "provider": self.config.retrieval.provider,
            },
            "execution": {
                "run_until_stage": run_until_stage,
                "executed_stages": executed_stages,
            },
            "research": {
                "enabled": self.config.workflow.research_enabled,
                "stage_count": len(self.config.workflow.research_stages),
                "best_score": research_data.get("best_score"),
                "best_node_id": research_data.get("best_node_id"),
                "total_nodes": research_data.get("total_nodes"),
            },
            "usage": usage,
            "idea_title": self._get_title(idea_data),
            "outline_has_chapters": "chapters" in outline_data,
        }

    def _get_title(self, idea_data: Dict[str, Any]) -> str:
        title = idea_data.get("thesis_title_cn")
        if isinstance(title, str) and title.strip():
            return title.strip()
        return ""

    def _strip_markdown_fence(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        pattern = r"^```(?:markdown|md)?\s*(.*?)\s*```$"
        match = re.match(pattern, stripped, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return stripped

    def _accumulate_usage(self, total: Dict[str, int], usage: Dict[str, int]) -> None:
        total["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        total["completion_tokens"] += int(usage.get("completion_tokens", 0))
        total["total_tokens"] += int(usage.get("total_tokens", 0))

    def _needs_continuation(
        self,
        *,
        full_text: str,
        finish_reason: str,
        round_idx: int,
        max_rounds: int,
    ) -> bool:
        if round_idx >= max_rounds - 1:
            return False

        if finish_reason == "length":
            return True

        # If key ending sections are missing, treat it as incomplete and continue.
        if not self._has_tail_sections(full_text):
            return True

        if self._looks_cutoff(full_text):
            return True

        return False

    def _build_continuation_prompt(self, current_paper: str) -> str:
        tail = self._tail_text(current_paper, self.config.workflow.continuation_tail_chars)
        missing_sections = []
        if "参考文献" not in current_paper:
            missing_sections.append("参考文献")
        if "致谢" not in current_paper:
            missing_sections.append("致谢")

        missing_hint = "、".join(missing_sections) if missing_sections else "后续未完成章节"
        return (
            "你正在续写同一篇管理学本科论文。\n"
            "请基于已生成内容继续写完，不要重写前文。\n\n"
            "已生成内容末尾（仅用于衔接）：\n"
            "```markdown\n"
            f"{tail}\n"
            "```\n\n"
            "要求：\n"
            "1. 只输出新增内容，不要重复已写文本。\n"
            "2. 从中断位置自然衔接，保持同一题目与结构。\n"
            f"3. 优先补齐：{missing_hint}。\n"
            "4. 输出到论文完整结束为止。\n"
        )

    def _build_outline_continuation_prompt(self, current_outline_text: str) -> str:
        tail = self._tail_text(
            current_outline_text, self.config.workflow.continuation_tail_chars
        )
        return (
            "你正在续写同一个论文大纲 JSON，上一轮输出因为长度限制可能被截断。\n"
            "请从中断位置继续，仅输出剩余 JSON 内容。\n\n"
            "已生成内容末尾（用于衔接）：\n"
            "```json\n"
            f"{tail}\n"
            "```\n\n"
            "要求：\n"
            "1. 不要重复已输出内容。\n"
            "2. 不要从头重写，不要加解释文字。\n"
            "3. 不要使用 Markdown 代码块包裹输出。\n"
            "4. 输出完成后确保整体可以拼接为合法 JSON。\n"
        )

    def _repair_json_object(
        self,
        broken_text: str,
        *,
        purpose: str,
        on_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, Dict[str, int]]:
        repair_system_prompt = (
            "你是一个严格的 JSON 修复器。"
            "你的任务是将输入的损坏/截断文本修复为一个合法 JSON 对象。"
            "只输出 JSON 对象，不要输出任何解释。"
        )
        repair_user_prompt = (
            f"请将下面的{purpose}文本修复为合法 JSON 对象。\n"
            "要求：\n"
            "1. 尽量保留已有内容，不要无依据扩写。\n"
            "2. 若存在截断，优先做结构补全（如括号、逗号、引号）。\n"
            "3. 输出必须是可被 json.loads 解析的单个 JSON 对象。\n\n"
            "原始文本：\n"
            "```text\n"
            f"{broken_text}\n"
            "```\n"
        )
        raw_text, usage = self._client_complete(
            system_prompt=repair_system_prompt,
            user_prompt=repair_user_prompt,
            temperature=0.0,
            max_tokens=max(self.config.llm.max_tokens // 2, 1500),
            on_delta=on_delta,
        )
        return self._strip_markdown_fence(raw_text), usage

    def _emit(self, event: str, **payload: Any) -> None:
        if not self.event_handler:
            return
        try:
            self.event_handler(event, payload)
        except Exception:
            # Event reporting should never break the workflow.
            return

    def _emit_stage_delta(self, stage: str, round_idx: int, text: str) -> None:
        self._ensure_not_cancelled()
        self._emit(
            "stage_delta",
            stage=stage,
            round=round_idx,
            text=text,
        )

    def _is_cancelled(self) -> bool:
        if not self.cancel_checker:
            return False
        try:
            return bool(self.cancel_checker())
        except Exception:
            return False

    def _ensure_not_cancelled(self) -> None:
        if not self._is_cancelled():
            return
        raise WorkflowCancelledError("任务已取消")

    def _client_complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Tuple[str, Dict[str, int]]:
        self._ensure_not_cancelled()
        try:
            return self.client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_delta=on_delta,
                cancel_checker=self._is_cancelled,
            )
        except RequestCancelledError as exc:
            raise WorkflowCancelledError(str(exc) or "任务已取消") from exc

    def _client_complete_with_meta(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Tuple[str, Dict[str, int], str]:
        self._ensure_not_cancelled()
        try:
            return self.client.complete_with_meta(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_delta=on_delta,
                cancel_checker=self._is_cancelled,
            )
        except RequestCancelledError as exc:
            raise WorkflowCancelledError(str(exc) or "任务已取消") from exc

    def _tail_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return text
        return text[-max_chars:]

    def _append_with_overlap(self, existing: str, new_chunk: str) -> str:
        left = existing.strip()
        right = new_chunk.strip()
        if not left:
            return right
        if not right:
            return left

        max_overlap = min(len(left), len(right), 1200)
        overlap = 0
        for size in range(max_overlap, 79, -1):
            if left[-size:] == right[:size]:
                overlap = size
                break

        if overlap:
            right = right[overlap:].lstrip()
        if not right:
            return left

        # If the continuation is fully duplicated content, skip it.
        if right in left:
            return left

        return f"{left}\n\n{right}"

    def _has_tail_sections(self, text: str) -> bool:
        has_ref = "参考文献" in text
        has_ack = "致谢" in text
        return has_ref and has_ack

    def _outline_is_complete(self, parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False
        chapters = parsed.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            return False
        return True

    def _looks_cutoff(self, text: str) -> bool:
        stripped = text.rstrip()
        if not stripped:
            return True
        tail = stripped[-1]
        if tail in {"，", "、", ",", "：", ":", "（", "(", "“", '"'}:
            return True
        if stripped.endswith("##") or stripped.endswith("###"):
            return True
        return False

    def _resolve_unique_dir(self, candidate: Path) -> Path:
        if not candidate.exists():
            return candidate
        suffix = 2
        while True:
            updated = candidate.with_name(f"{candidate.name}_{suffix}")
            if not updated.exists():
                return updated
            suffix += 1
