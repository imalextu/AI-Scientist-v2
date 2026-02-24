from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from .config import AppConfig
from .llm_client import LLMClient, RequestCancelledError
from .retrieval import LiteratureRetriever, format_literature_context
from .web_search import WebSearchClient, format_web_search_context
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
1. 先做可研究性评估：可证伪性、变量可操作化、数据可获得性。
2. 生成 3-5 个候选研究问题（RQ），每个包含理论切入点、预期贡献、难度评级。
3. 推荐最优 RQ 并说明理由，明确最终研究问题与初步理论锚点。
4. 必须输出“论文类型判定”字段，并在 `定量实证型` / `案例研究型` / `对策建议型` 三选一。
5. 论文类型判定依据必须覆盖：题目特征、数据条件、学生偏好；信息不足时默认 `对策建议型`。
6. 方法建议要适配本科阶段可执行条件（问卷、访谈、案例、公开数据等），并与论文类型一致。
7. 按要求输出严格 JSON，不要额外解释。
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

LITERATURE_REVIEW_SYSTEM_PROMPT = """
你是管理学文献综述写作导师。请基于给定主题、题目、选题设计结果与文献证据，产出高质量中文文献综述。
要求：
1. 严格围绕输入文献，不得捏造不存在的来源。
2. 结构化呈现研究问题、机制、变量、方法与结论。
3. 正文需与已确定研究问题和理论锚点保持一致，强调比较、综合与批判，不是逐篇摘抄。
4. 按模板输出 Markdown，不要输出 JSON。
""".strip()

RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT = """
你是学术检索查询改写助手。你的任务是把用户给出的论文题目改写为中英文两个检索标题。
输出必须是严格 JSON，格式：
{
  "zh_query": "...",
  "en_query": "..."
}
要求：
1. `zh_query` 保持原标题语义（若输入是中文，可直接使用原标题）。
2. `en_query` 给出原标题的英文翻译，保持语义等价，不做扩展。
3. 只返回这两个字段，不要增加其他字段。
4. 不要输出除 JSON 之外的内容。
""".strip()

RETRIEVAL_RERANK_SYSTEM_PROMPT = """
你是学术文献检索精排助手。请从候选文献中选出最相关的结果并排序。
输出必须是严格 JSON，格式：
{
  "selected_ids": [1, 4, 2]
}
要求：
1. 只输出 `selected_ids`，按相关性从高到低排列。
2. id 必须来自候选列表中的 `id` 字段。
3. 返回数量不超过请求的 `top_k`。
4. 不要输出 JSON 之外的内容。
""".strip()

POLICY_DIAGNOSIS_SYSTEM_PROMPT = """
你是管理学本科论文问题诊断助手。请基于输入信息生成“现状-问题-原因”三层分析。
输出必须是严格 JSON，不要输出解释。
要求：
1. 仅依据给定输入推断，不得编造可验证事实。
2. 每个问题需对应至少一个原因，并给出证据线索来源。
3. 分析粒度保持本科论文可执行深度。
""".strip()

POLICY_STRATEGY_SYSTEM_PROMPT = """
你是管理学本科论文对策设计助手。请基于三层分析结果设计可执行的对策体系。
输出必须是严格 JSON，不要输出解释。
要求：
1. 对策需与问题-原因逐项对齐。
2. 给出实施路径、优先级、资源与风险控制。
3. 避免空泛口号，保证本科写作可展开。
""".strip()

LLM_LOG_PROMPT_CHAR_LIMIT = 12000
IDEA_PAPER_TYPE_FIELD = "论文类型判定"
DEFAULT_PAPER_TYPE = "对策建议型"
PAPER_TYPE_BRANCH_PATHS = {
    "定量实证型": ["理论框架+假设", "研究设计(问卷)", "数据分析"],
    "案例研究型": ["分析框架构建", "案例数据收集", "案例分析"],
    "对策建议型": ["轻量理论+现状调研", "问题诊断", "对策设计"],
}
PAPER_TYPE_ALIASES = {
    "定量": "定量实证型",
    "实证": "定量实证型",
    "quantitative": "定量实证型",
    "empirical": "定量实证型",
    "案例": "案例研究型",
    "case": "案例研究型",
    "case study": "案例研究型",
    "对策": "对策建议型",
    "建议": "对策建议型",
    "policy": "对策建议型",
    "strategy": "对策建议型",
}

WorkflowEventHandler = Callable[[str, Dict[str, Any]], None]


class WorkflowCancelledError(RuntimeError):
    pass


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
        self.retriever = LiteratureRetriever(
            config.retrieval,
            query_expander=self._expand_literature_queries_with_llm,
        )
        self.web_search = WebSearchClient(config.web_search)
        self.event_handler = event_handler
        self.cancel_checker = cancel_checker

    def run(
        self,
        topic_text: str,
        forced_title: str | None = None,
        *,
        resume_from_stage: str | None = None,
        resume_cache: Dict[str, Any] | None = None,
    ) -> Path:
        stage_order = {
            "literature": 1,
            "idea": 2,
            "review": 3,
            "outline": 4,
            "paper": 5,
        }
        target_stage = str(self.config.workflow.run_until_stage).strip().lower() or "paper"
        if target_stage not in stage_order:
            target_stage = "paper"
        if resume_from_stage is None:
            start_stage = "literature"
        else:
            start_stage = str(resume_from_stage).strip().lower()
            if start_stage not in stage_order:
                raise ValueError(f"无效的 resume_from_stage: {resume_from_stage}")
        if stage_order[start_stage] > stage_order[target_stage]:
            raise ValueError(
                f"resume_from_stage={start_stage} 不能晚于 run_until_stage={target_stage}。"
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
            resume_from_stage=start_stage if resume_from_stage else "",
            run_until_stage=target_stage,
        )

        write_text(run_dir / "00_topic.md", topic_text.strip())
        self._ensure_not_cancelled()
        self._emit(
            "topic_saved",
            path=str(run_dir / "00_topic.md"),
            topic=topic_text.strip(),
        )

        resolved_title = (forced_title or "").strip()
        if not resolved_title:
            topic_lines = [
                re.sub(r"\s+", " ", line).strip()
                for line in topic_text.splitlines()
                if line.strip()
            ]
            resolved_title = topic_lines[0] if topic_lines else "未命名题目"

        literature_items: list[dict[str, str]] = []
        literature_context = "暂无可用文献线索。"

        idea_data: Dict[str, Any] = {}
        idea_raw = ""
        idea_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }
        review_text = ""
        review_context = "文献综述阶段尚未执行。"
        review_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }
        policy_support_data: Dict[str, Any] = {}
        policy_support_context = "对策建议型增强输入未启用。"
        policy_support_usage: Dict[str, int] = {
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
        restored_stages: list[str] = []
        cache_payload = resume_cache or {}

        def _should_execute(stage: str) -> bool:
            order = stage_order[stage]
            return stage_order[start_stage] <= order <= stage_order[target_stage]

        if stage_order[start_stage] > stage_order["literature"]:
            cached_literature = cache_payload.get("literature_items")
            if not isinstance(cached_literature, list):
                raise ValueError("恢复运行缺少 00_literature.json 缓存内容。")
            if not all(isinstance(item, dict) for item in cached_literature):
                raise ValueError("缓存的 00_literature.json 格式错误。")
            literature_items = cached_literature
            literature_context = format_literature_context(literature_items)
            write_json(run_dir / "00_literature.json", literature_items)
            self._emit(
                "stage_restored",
                stage="literature",
                path=str(run_dir / "00_literature.json"),
                content=json.dumps(literature_items, ensure_ascii=False, indent=2),
            )
            restored_stages.append("literature")

        if stage_order[start_stage] > stage_order["idea"]:
            cached_idea = cache_payload.get("idea_data")
            if not isinstance(cached_idea, dict):
                raise ValueError("恢复运行缺少 01_idea.json 缓存内容。")
            idea_data = self._normalize_idea_paper_type(cached_idea)
            write_json(run_dir / "01_idea.json", idea_data)
            self._emit(
                "stage_restored",
                stage="idea",
                path=str(run_dir / "01_idea.json"),
                content=json.dumps(idea_data, ensure_ascii=False, indent=2),
            )
            restored_stages.append("idea")

        if stage_order[start_stage] > stage_order["review"]:
            cached_review = cache_payload.get("review_text")
            if not isinstance(cached_review, str):
                raise ValueError("恢复运行缺少 00_literature_review.md 缓存内容。")
            review_text = cached_review
            review_context = review_text.strip() or "文献综述阶段无有效输出。"
            write_text(run_dir / "00_literature_review.md", review_text)
            self._emit(
                "stage_restored",
                stage="review",
                path=str(run_dir / "00_literature_review.md"),
                content=review_text,
            )
            restored_stages.append("review")

        if stage_order[start_stage] > stage_order["outline"]:
            cached_outline = cache_payload.get("outline_data")
            if not isinstance(cached_outline, dict):
                raise ValueError("恢复运行缺少 02_outline.json 缓存内容。")
            outline_data = cached_outline
            if self._is_policy_paper(idea_data):
                policy_raw = outline_data.get("policy_support")
                if isinstance(policy_raw, dict):
                    policy_support_data = policy_raw
                    policy_support_context = self._format_policy_support_context(
                        policy_support_data
                    )
            write_json(run_dir / "02_outline.json", outline_data)
            self._emit(
                "stage_restored",
                stage="outline",
                path=str(run_dir / "02_outline.json"),
                content=json.dumps(outline_data, ensure_ascii=False, indent=2),
            )
            restored_stages.append("outline")

        if _should_execute("literature"):
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

        if _should_execute("idea"):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="idea")
            idea_data, idea_raw, idea_usage = self._run_idea_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                paper_title=resolved_title,
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

        if _should_execute("review"):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="review")
            review_text, review_usage = self._run_literature_review_stage(
                topic_text=topic_text,
                paper_title=resolved_title,
                idea_data=idea_data,
                literature_items=literature_items,
                literature_context=literature_context,
            )
            review_context = review_text.strip() or "文献综述阶段无有效输出。"
            write_text(run_dir / "00_literature_review.md", review_text)
            self._ensure_not_cancelled()
            self._emit(
                "stage_completed",
                stage="review",
                path=str(run_dir / "00_literature_review.md"),
                content=review_text,
                usage=review_usage,
            )
            executed_stages.append("review")

        needs_policy_support = self._is_policy_paper(idea_data) and (
            _should_execute("outline") or _should_execute("paper")
        )
        if needs_policy_support and not policy_support_data:
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="policy_support")
            policy_support_data, policy_support_usage = self._run_policy_support_stage(
                topic_text=topic_text,
                idea_data=idea_data,
                literature_context=literature_context,
                review_context=review_context,
            )
            policy_support_context = self._format_policy_support_context(
                policy_support_data
            )
            write_json(run_dir / "02a_policy_support.json", policy_support_data)
            self._ensure_not_cancelled()
            self._emit(
                "stage_completed",
                stage="policy_support",
                path=str(run_dir / "02a_policy_support.json"),
                content=json.dumps(policy_support_data, ensure_ascii=False, indent=2),
                usage=policy_support_usage,
            )
            executed_stages.append("policy_support")
        elif policy_support_data:
            policy_support_context = self._format_policy_support_context(
                policy_support_data
            )
        if self._is_policy_paper(idea_data) and policy_support_data and isinstance(
            outline_data, dict
        ):
            outline_data["policy_support"] = policy_support_data

        if _should_execute("outline"):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="outline")
            outline_data, outline_raw, outline_usage = self._run_outline_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                review_context=review_context,
                idea_data=idea_data,
                policy_support_data=policy_support_data,
                policy_support_context=policy_support_context,
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

        if _should_execute("paper"):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="paper")
            thesis_md, thesis_raw, paper_usage = self._run_paper_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                review_context=review_context,
                idea_data=idea_data,
                outline_data=outline_data,
                policy_support_data=policy_support_data,
                policy_support_context=policy_support_context,
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
            if "review" in executed_stages:
                write_text(run_dir / "raw_00_review.txt", review_text)
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
            restored_stages=restored_stages,
            resume_from_stage=start_stage if resume_from_stage else "",
            review_text=review_text,
            idea_data=idea_data,
            outline_data=outline_data,
            usage={
                "review": review_usage,
                "idea": idea_usage,
                "outline": outline_usage,
                "policy_support": policy_support_usage,
                "paper": paper_usage,
            },
            policy_support_data=policy_support_data,
        )
        write_json(run_dir / "run_metadata.json", metadata)
        self._emit("metadata_saved", path=str(run_dir / "run_metadata.json"))

        self._ensure_not_cancelled()
        final_title = ""
        if idea_data:
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
        candidates = self.retriever.search(topic_text)
        items = self._rerank_literature_with_llm(topic_text, candidates)
        self._ensure_not_cancelled()
        return items, format_literature_context(items)

    def _run_policy_support_stage(
        self,
        *,
        topic_text: str,
        idea_data: Dict[str, Any],
        literature_context: str,
        review_context: str,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }
        web_items, web_context, web_queries = self._collect_policy_web_research(
            topic_text=topic_text,
            idea_data=idea_data,
        )
        analysis_data, analysis_usage = self._run_policy_three_layer_analysis(
            topic_text=topic_text,
            idea_data=idea_data,
            literature_context=literature_context,
            review_context=review_context,
            web_context=web_context,
        )
        strategy_data, strategy_usage = self._run_policy_countermeasure_design(
            topic_text=topic_text,
            idea_data=idea_data,
            three_layer_analysis=analysis_data,
            web_context=web_context,
        )
        self._accumulate_usage(usage_total, analysis_usage)
        self._accumulate_usage(usage_total, strategy_usage)
        usage_total["rounds"] = 2
        return (
            {
                "pipeline_steps": [
                    "企业/行业信息采集",
                    "现状-问题-原因三层分析",
                    "对策体系设计",
                ],
                "web_research": {
                    "queries": web_queries,
                    "items": web_items,
                    "context": web_context,
                },
                "three_layer_analysis": analysis_data,
                "countermeasure_system": strategy_data,
            },
            usage_total,
        )

    def _collect_policy_web_research(
        self,
        *,
        topic_text: str,
        idea_data: Dict[str, Any],
    ) -> Tuple[list[dict[str, str]], str, list[str]]:
        queries = self._build_policy_web_queries(topic_text=topic_text, idea_data=idea_data)
        self._emit(
            "web_search_started",
            enabled=self.config.web_search.enabled,
            provider=self.config.web_search.provider,
            query_count=len(queries),
            queries=queries,
        )

        raw_items: list[dict[str, str]] = []
        for query in queries:
            self._ensure_not_cancelled()
            raw_items.extend(self.web_search.search(query))

        deduped_items = self._dedupe_web_items(raw_items)
        if self.config.web_search.max_results > 0:
            deduped_items = deduped_items[: self.config.web_search.max_results]

        context = format_web_search_context(deduped_items)
        if queries:
            query_line = " | ".join(queries)
            context = f"检索词：{query_line}\n\n{context}"

        self._emit(
            "web_search_completed",
            enabled=self.config.web_search.enabled,
            provider=self.config.web_search.provider,
            count=len(deduped_items),
            queries=queries,
            items=deduped_items,
        )
        return deduped_items, context, queries

    def _build_policy_web_queries(
        self,
        *,
        topic_text: str,
        idea_data: Dict[str, Any],
    ) -> list[str]:
        title = self._get_title(idea_data)
        final_question = str(idea_data.get("final_research_question") or "").strip()
        candidates = [
            f"{title} 企业 行业 现状 问题" if title else "",
            f"{final_question} 现状 原因 对策" if final_question else "",
            re.sub(r"\s+", " ", topic_text).strip(),
        ]
        queries: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            text = re.sub(r"\s+", " ", str(candidate or "").strip())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(text)
            if len(queries) >= 3:
                break
        return queries

    def _dedupe_web_items(self, items: list[dict[str, str]]) -> list[dict[str, str]]:
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip().lower()
            title = re.sub(r"\W+", "", str(item.get("title") or "").strip().lower())
            key = f"url:{url}" if url else f"title:{title}"
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _run_policy_three_layer_analysis(
        self,
        *,
        topic_text: str,
        idea_data: Dict[str, Any],
        literature_context: str,
        review_context: str,
        web_context: str,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        user_prompt = (
            "请基于输入材料完成“现状-问题-原因”三层分析，并输出 JSON。\n\n"
            f"主题：{topic_text}\n\n"
            "选题设计 JSON：\n"
            f"{json.dumps(idea_data, ensure_ascii=False, indent=2)}\n\n"
            "文献线索：\n"
            f"{literature_context}\n\n"
            "文献综述结论：\n"
            f"{review_context}\n\n"
            "企业/行业信息采集：\n"
            f"{web_context}\n\n"
            "输出 JSON 建议结构：\n"
            "{\n"
            '  "current_situation": ["..."],\n'
            '  "key_problems": [{"problem":"...","evidence":"..."}],\n'
            '  "root_causes": [{"cause":"...","linked_problem":"...","evidence":"..."}],\n'
            '  "analysis_summary": "..."\n'
            "}\n"
        )
        self._emit_llm_request(
            stage="policy_support",
            operation="three_layer_analysis",
            system_prompt=POLICY_DIAGNOSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.workflow.temperature,
            max_tokens=max(self.config.llm.max_tokens // 2, 1800),
            round_idx=1,
        )
        raw_text, usage = self._client_complete(
            system_prompt=POLICY_DIAGNOSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.workflow.temperature,
            max_tokens=max(self.config.llm.max_tokens // 2, 1800),
            on_delta=lambda piece: self._emit(
                "stage_delta",
                stage="policy_support",
                round=1,
                text=piece,
            ),
        )
        parsed = extract_json_payload(raw_text)
        if not isinstance(parsed, dict):
            parsed = {"raw_response": raw_text}
        parsed.setdefault("current_situation", [])
        parsed.setdefault("key_problems", [])
        parsed.setdefault("root_causes", [])
        parsed.setdefault("analysis_summary", "")
        return parsed, usage

    def _run_policy_countermeasure_design(
        self,
        *,
        topic_text: str,
        idea_data: Dict[str, Any],
        three_layer_analysis: Dict[str, Any],
        web_context: str,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        user_prompt = (
            "请基于输入材料设计“对策体系”，并输出 JSON。\n\n"
            f"主题：{topic_text}\n\n"
            "选题设计 JSON：\n"
            f"{json.dumps(idea_data, ensure_ascii=False, indent=2)}\n\n"
            "现状-问题-原因三层分析 JSON：\n"
            f"{json.dumps(three_layer_analysis, ensure_ascii=False, indent=2)}\n\n"
            "企业/行业信息采集：\n"
            f"{web_context}\n\n"
            "输出 JSON 建议结构：\n"
            "{\n"
            '  "overall_goal": "...",\n'
            '  "strategy_system": [{"dimension":"...","actions":["..."],"priority":"高/中/低","timeline":"短期/中期/长期"}],\n'
            '  "implementation_roadmap": [{"phase":"...","tasks":["..."],"owner":"..."}],\n'
            '  "safeguard_mechanisms": ["..."]\n'
            "}\n"
        )
        self._emit_llm_request(
            stage="policy_support",
            operation="countermeasure_design",
            system_prompt=POLICY_STRATEGY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.workflow.temperature,
            max_tokens=max(self.config.llm.max_tokens // 2, 1800),
            round_idx=2,
        )
        raw_text, usage = self._client_complete(
            system_prompt=POLICY_STRATEGY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.workflow.temperature,
            max_tokens=max(self.config.llm.max_tokens // 2, 1800),
            on_delta=lambda piece: self._emit(
                "stage_delta",
                stage="policy_support",
                round=2,
                text=piece,
            ),
        )
        parsed = extract_json_payload(raw_text)
        if not isinstance(parsed, dict):
            parsed = {"raw_response": raw_text}
        parsed.setdefault("overall_goal", "")
        parsed.setdefault("strategy_system", [])
        parsed.setdefault("implementation_roadmap", [])
        parsed.setdefault("safeguard_mechanisms", [])
        return parsed, usage

    def _format_policy_support_context(self, policy_support_data: Dict[str, Any]) -> str:
        if not isinstance(policy_support_data, dict) or not policy_support_data:
            return "对策建议型增强输入未生成。"

        web_part = policy_support_data.get("web_research", {})
        analysis_part = policy_support_data.get("three_layer_analysis", {})
        strategy_part = policy_support_data.get("countermeasure_system", {})
        return (
            "【企业/行业信息采集】\n"
            f"{json.dumps(web_part, ensure_ascii=False, indent=2)}\n\n"
            "【现状-问题-原因 三层分析】\n"
            f"{json.dumps(analysis_part, ensure_ascii=False, indent=2)}\n\n"
            "【对策体系设计】\n"
            f"{json.dumps(strategy_part, ensure_ascii=False, indent=2)}"
        )

    def _expand_literature_queries_with_llm(self, topic_text: str) -> list[str]:
        self._ensure_not_cancelled()
        user_prompt = (
            "请将以下论文题目改写为中英文检索标题。\n\n"
            f"题目：{topic_text.strip()}\n\n"
            "请按要求返回 JSON。"
        )
        self._emit_llm_request(
            stage="literature",
            operation="query_rewrite",
            system_prompt=RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=500,
        )
        try:
            response_text, _, _ = self.client.complete_with_meta(
                system_prompt=RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=500,
                cancel_checker=self.cancel_checker,
            )
        except RequestCancelledError:
            raise WorkflowCancelledError("任务已取消")
        except Exception:
            return []

        payload = extract_json_payload(response_text)
        if not isinstance(payload, dict):
            return []

        def _clean(value: Any) -> str:
            return re.sub(r"\s+", " ", str(value or "").strip())

        def _first_text(value: Any) -> str:
            if isinstance(value, str):
                return _clean(value)
            if isinstance(value, list):
                for item in value:
                    text = _clean(item)
                    if text:
                        return text
            return ""

        zh_query = _first_text(payload.get("zh_query"))
        en_query = _first_text(payload.get("en_query"))

        # 兼容旧格式：若仍返回数组字段，取首条
        if not zh_query:
            zh_query = _first_text(payload.get("zh_queries"))
        if not en_query:
            en_query = _first_text(payload.get("en_queries"))

        if not zh_query:
            zh_query = _clean(topic_text)

        queries: list[str] = []
        seen: set[str] = set()
        for query in (zh_query, en_query):
            key = query.lower()
            if not query or key in seen:
                continue
            seen.add(key)
            queries.append(query)

        if queries:
            self._emit(
                "literature_query_expanded",
                count=len(queries),
                queries=queries,
            )
        return queries

    def _rerank_literature_with_llm(
        self,
        topic_text: str,
        candidates: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        top_k = max(1, self.config.retrieval.max_results)
        if not candidates:
            return []

        pool = candidates[: max(top_k, min(len(candidates), top_k * 4))]
        payload_items = []
        for idx, item in enumerate(pool, start=1):
            payload_items.append(
                {
                    "id": idx,
                    "title": item.get("title", ""),
                    "authors": item.get("authors", ""),
                    "year": item.get("year", ""),
                    "venue": item.get("venue", ""),
                    "source": item.get("source", ""),
                    "abstract": (item.get("abstract", "") or "")[:260],
                }
            )

        user_prompt = (
            "请根据题目从候选文献中选出最相关结果并排序。\n\n"
            f"题目：{topic_text.strip()}\n"
            f"top_k：{top_k}\n\n"
            "候选文献（JSON）：\n"
            f"{json.dumps(payload_items, ensure_ascii=False)}\n\n"
            "请只返回 JSON。"
        )

        self._emit_llm_request(
            stage="literature",
            operation="rerank",
            system_prompt=RETRIEVAL_RERANK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=800,
        )
        try:
            response_text, _, _ = self.client.complete_with_meta(
                system_prompt=RETRIEVAL_RERANK_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=800,
                cancel_checker=self.cancel_checker,
            )
        except RequestCancelledError:
            raise WorkflowCancelledError("任务已取消")
        except Exception:
            return pool[:top_k]

        payload = extract_json_payload(response_text)
        if not isinstance(payload, dict):
            return pool[:top_k]

        raw_ids = payload.get("selected_ids")
        if not isinstance(raw_ids, list):
            return pool[:top_k]

        selected_ids: list[int] = []
        seen_ids: set[int] = set()
        for value in raw_ids:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if idx < 1 or idx > len(pool) or idx in seen_ids:
                continue
            seen_ids.add(idx)
            selected_ids.append(idx)
            if len(selected_ids) >= top_k:
                break

        ranked: list[dict[str, str]] = [pool[idx - 1] for idx in selected_ids]
        if len(ranked) < top_k:
            for idx, item in enumerate(pool, start=1):
                if idx in seen_ids:
                    continue
                ranked.append(item)
                if len(ranked) >= top_k:
                    break
        return ranked

    def _run_literature_review_stage(
        self,
        *,
        topic_text: str,
        paper_title: str,
        idea_data: Dict[str, Any],
        literature_items: list[dict[str, str]],
        literature_context: str,
    ) -> Tuple[str, Dict[str, int]]:
        # This stage starts with one generation round and auto-continues only when truncated.
        prompt_template = read_text(self.project_root / self.config.workflow.review_prompt)
        target_refs = min(12, max(8, len(literature_items) if literature_items else 8))
        base_prompt = render_template(
            prompt_template,
            {
                "topic": topic_text,
                "paper_title": paper_title,
                "target_references": target_refs,
                "literature_context": literature_context,
                "idea_json": json.dumps(
                    idea_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                "literature_json": json.dumps(
                    literature_items,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        )
        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        raw_chunks: list[str] = []
        merged_review = ""
        max_rounds = 4
        current_prompt = base_prompt

        for round_idx in range(max_rounds):
            self._ensure_not_cancelled()
            current_round = round_idx + 1
            self._emit("stage_round_started", stage="review", round=current_round)

            delta_emitted = False

            def emit_review_delta(piece: str) -> None:
                nonlocal delta_emitted
                if not piece:
                    return
                delta_emitted = True
                start = 0
                chunk_size = 180
                while start < len(piece):
                    self._emit_stage_delta(
                        "review",
                        current_round,
                        piece[start : start + chunk_size],
                    )
                    start += chunk_size

            self._emit_llm_request(
                stage="review",
                operation="literature_review",
                system_prompt=LITERATURE_REVIEW_SYSTEM_PROMPT,
                user_prompt=current_prompt,
                temperature=self.config.workflow.temperature,
                max_tokens=max(
                    self.config.workflow.review_max_tokens,
                    self.config.llm.max_tokens,
                ),
                round_idx=current_round,
            )
            raw_text, usage, finish_reason = self._client_complete_with_meta(
                system_prompt=LITERATURE_REVIEW_SYSTEM_PROMPT,
                user_prompt=current_prompt,
                temperature=self.config.workflow.temperature,
                max_tokens=max(
                    self.config.workflow.review_max_tokens,
                    self.config.llm.max_tokens,
                ),
                on_delta=emit_review_delta,
            )
            clean_chunk = self._strip_markdown_fence(raw_text).strip()
            if clean_chunk and not delta_emitted:
                emit_review_delta(clean_chunk)

            raw_chunks.append(raw_text)
            self._accumulate_usage(usage_total, usage)
            self._emit(
                "stage_round_completed",
                stage="review",
                round=current_round,
                finish_reason=finish_reason,
            )

            merged_review = self._append_with_overlap(merged_review, clean_chunk)
            if not self._review_needs_continuation(
                full_text=merged_review,
                finish_reason=finish_reason,
                round_idx=round_idx,
                max_rounds=max_rounds,
            ):
                break

            if round_idx < max_rounds - 1:
                current_prompt = self._build_review_continuation_prompt(merged_review)

        usage_total["rounds"] = len(raw_chunks)
        return merged_review.strip(), usage_total

    def _run_idea_stage(
        self,
        *,
        topic_text: str,
        literature_context: str,
        paper_title: str,
        forced_title: str | None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
        prompt_template = read_text(self.project_root / self.config.workflow.idea_prompt)
        user_prompt = render_template(
            prompt_template,
            {
                "topic": topic_text,
                "paper_title": paper_title,
                "domain": self.config.paper.domain,
                "audience": self.config.paper.audience,
                "language": self.config.paper.language,
                "min_words": self.config.paper.min_words,
                "max_words": self.config.paper.max_words,
                "citation_style": self.config.paper.citation_style,
                "literature_context": literature_context,
            },
        )
        self._emit_llm_request(
            stage="idea",
            operation="idea_generation",
            system_prompt=IDEA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.workflow.temperature,
            max_tokens=max(self.config.llm.max_tokens // 2, 1800),
            round_idx=1,
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
        parsed = self._normalize_idea_paper_type(parsed)
        return parsed, raw_text, usage

    def _run_outline_stage(
        self,
        *,
        topic_text: str,
        literature_context: str,
        review_context: str,
        idea_data: Dict[str, Any],
        policy_support_data: Dict[str, Any],
        policy_support_context: str,
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
                "review_context": review_context,
                "policy_support_json": json.dumps(
                    policy_support_data, ensure_ascii=False, indent=2
                ),
                "policy_support_context": policy_support_context,
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
        if self._is_policy_paper(idea_data) and policy_support_data:
            final_parsed["policy_support"] = policy_support_data

        usage_total["rounds"] = len(raw_chunks)
        raw_text_joined = "\n\n\n<!-- CONTINUATION ROUND -->\n\n".join(raw_chunks)
        return final_parsed, raw_text_joined.strip(), usage_total

    def _run_paper_stage(
        self,
        *,
        topic_text: str,
        literature_context: str,
        review_context: str,
        idea_data: Dict[str, Any],
        outline_data: Dict[str, Any],
        policy_support_data: Dict[str, Any],
        policy_support_context: str,
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
                "review_context": review_context,
                "policy_support_json": json.dumps(
                    policy_support_data, ensure_ascii=False, indent=2
                ),
                "policy_support_context": policy_support_context,
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
        restored_stages: list[str],
        resume_from_stage: str,
        review_text: str,
        idea_data: Dict[str, Any],
        outline_data: Dict[str, Any],
        policy_support_data: Dict[str, Any],
        usage: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        review_body = review_text.strip()
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
                "max_results": self.config.retrieval.max_results,
            },
            "execution": {
                "run_until_stage": run_until_stage,
                "executed_stages": executed_stages,
                "resume_from_stage": resume_from_stage,
                "restored_stages": restored_stages,
            },
            "review": {
                "enabled": True,
                "has_content": bool(review_body),
                "char_count": len(review_body),
            },
            "usage": usage,
            "idea_title": self._get_title(idea_data),
            "idea_paper_type": self._get_paper_type(idea_data),
            "outline_has_chapters": "chapters" in outline_data,
            "policy_support_enabled": bool(policy_support_data),
        }

    def _get_title(self, idea_data: Dict[str, Any]) -> str:
        title = idea_data.get("thesis_title_cn")
        if isinstance(title, str) and title.strip():
            return title.strip()
        return ""

    def _get_paper_type(self, idea_data: Dict[str, Any]) -> str:
        paper_type_block = idea_data.get(IDEA_PAPER_TYPE_FIELD)
        if not isinstance(paper_type_block, dict):
            paper_type_block = idea_data.get("paper_type_judgement")
        if not isinstance(paper_type_block, dict):
            return ""
        normalized = self._normalize_paper_type_value(
            paper_type_block.get("recommended_type")
        )
        return normalized or ""

    def _is_policy_paper(self, idea_data: Dict[str, Any]) -> bool:
        return self._get_paper_type(idea_data) == DEFAULT_PAPER_TYPE

    def _normalize_idea_paper_type(self, idea_data: Dict[str, Any]) -> Dict[str, Any]:
        paper_type_block = idea_data.get(IDEA_PAPER_TYPE_FIELD)
        if not isinstance(paper_type_block, dict):
            paper_type_block = idea_data.get("paper_type_judgement")
        if not isinstance(paper_type_block, dict):
            paper_type_block = {}

        normalized_type = self._normalize_paper_type_value(
            paper_type_block.get("recommended_type")
        )
        fallback_used = False
        if not normalized_type:
            normalized_type = DEFAULT_PAPER_TYPE
            fallback_used = True

        if isinstance(paper_type_block.get("fallback_used"), bool):
            fallback_used = bool(paper_type_block.get("fallback_used")) or fallback_used

        decision_basis_raw = paper_type_block.get("decision_basis")
        decision_basis = (
            decision_basis_raw if isinstance(decision_basis_raw, dict) else {}
        )

        topic_features = str(decision_basis.get("topic_features") or "").strip() or "待补充"
        data_conditions = str(decision_basis.get("data_conditions") or "").strip() or "待补充"
        student_preference = (
            str(decision_basis.get("student_preference") or "").strip() or "未明确"
        )

        selection_reason = str(paper_type_block.get("selection_reason") or "").strip()
        if not selection_reason:
            if fallback_used:
                selection_reason = "缺少稳定判定信息，默认采用“对策建议型”以保证本科阶段可执行。"
            else:
                selection_reason = (
                    f"综合题目特征、数据条件与学生偏好，推荐“{normalized_type}”。"
                )

        branch_path_raw = paper_type_block.get("branch_path")
        branch_path: list[str] = []
        if isinstance(branch_path_raw, list):
            for step in branch_path_raw:
                text = str(step).strip()
                if text and text not in branch_path:
                    branch_path.append(text)
        if not branch_path:
            branch_path = list(
                PAPER_TYPE_BRANCH_PATHS.get(
                    normalized_type,
                    PAPER_TYPE_BRANCH_PATHS[DEFAULT_PAPER_TYPE],
                )
            )

        idea_data[IDEA_PAPER_TYPE_FIELD] = {
            "recommended_type": normalized_type,
            "decision_basis": {
                "topic_features": topic_features,
                "data_conditions": data_conditions,
                "student_preference": student_preference,
            },
            "selection_reason": selection_reason,
            "fallback_used": fallback_used,
            "branch_path": branch_path,
        }
        idea_data["paper_type_judgement"] = idea_data[IDEA_PAPER_TYPE_FIELD]
        return idea_data

    def _normalize_paper_type_value(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text in PAPER_TYPE_BRANCH_PATHS:
            return text
        lowered = text.lower()
        alias = PAPER_TYPE_ALIASES.get(lowered)
        if alias:
            return alias
        for keyword, paper_type in PAPER_TYPE_ALIASES.items():
            if keyword in lowered:
                return paper_type
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

    def _build_review_continuation_prompt(self, current_review: str) -> str:
        tail = self._tail_text(
            current_review, self.config.workflow.continuation_tail_chars
        )
        missing_sections: list[str] = []
        if "结构化证据表" not in current_review:
            missing_sections.append("结构化证据表")
        if "文献综述正文" not in current_review:
            missing_sections.append("文献综述正文")
        if "A) 引文清单" not in current_review and "A）引文清单" not in current_review:
            missing_sections.append("A) 引文清单")
        if "B) 研究缺口清单" not in current_review and "B）研究缺口清单" not in current_review:
            missing_sections.append("B) 研究缺口清单")
        missing_hint = "、".join(missing_sections) if missing_sections else "剩余未完成内容"
        return (
            "你正在续写同一份管理学文献综述 Markdown，上一轮输出可能因长度限制被截断。\n"
            "请紧接上文继续写，不要重复已输出内容，不要重写前文。\n\n"
            "已生成内容末尾（仅用于衔接）：\n"
            "```markdown\n"
            f"{tail}\n"
            "```\n\n"
            "要求：\n"
            "1. 只输出新增内容。\n"
            "2. 保持与上一轮一致的结构和语气。\n"
            f"3. 优先补齐：{missing_hint}。\n"
            "4. 直到整份文献综述完整结束。\n"
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

    def _emit_llm_request(
        self,
        *,
        stage: str,
        operation: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        round_idx: int | None = None,
    ) -> None:
        self._emit(
            "llm_request",
            stage=stage,
            operation=operation,
            round=round_idx,
            model=self.config.llm.model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_char_limit=LLM_LOG_PROMPT_CHAR_LIMIT,
            system_prompt_char_count=len(system_prompt),
            user_prompt_char_count=len(user_prompt),
            system_prompt=self._clip_for_log(system_prompt),
            user_prompt=self._clip_for_log(user_prompt),
            system_prompt_truncated=len(system_prompt) > LLM_LOG_PROMPT_CHAR_LIMIT,
            user_prompt_truncated=len(user_prompt) > LLM_LOG_PROMPT_CHAR_LIMIT,
        )

    def _clip_for_log(self, text: str, max_chars: int = LLM_LOG_PROMPT_CHAR_LIMIT) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

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

    def _review_has_required_sections(self, text: str) -> bool:
        if "结构化证据表" not in text:
            return False
        if "文献综述正文" not in text:
            return False
        has_a = "A) 引文清单" in text or "A）引文清单" in text
        has_b = "B) 研究缺口清单" in text or "B）研究缺口清单" in text
        return has_a and has_b

    def _review_needs_continuation(
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
        if self._looks_cutoff(full_text):
            return True
        if not self._review_has_required_sections(full_text):
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
