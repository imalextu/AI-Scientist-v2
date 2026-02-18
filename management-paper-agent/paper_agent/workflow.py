from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from .config import AppConfig
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

LITERATURE_REVIEW_SYSTEM_PROMPT = """
你是管理学文献综述写作导师。请基于给定主题、题目与文献证据，产出高质量中文文献综述。
要求：
1. 严格围绕输入文献，不得捏造不存在的来源。
2. 结构化呈现研究问题、机制、变量、方法与结论。
3. 正文强调比较、综合与批判，不是逐篇摘抄。
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
            "review": 2,
            "idea": 3,
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

        literature_items: list[dict[str, str]] = []
        literature_context = "暂无可用文献线索。"

        review_text = ""
        review_context = "文献综述阶段尚未执行。"
        review_usage: Dict[str, int] = {
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

        if stage_order[start_stage] > stage_order["idea"]:
            cached_idea = cache_payload.get("idea_data")
            if not isinstance(cached_idea, dict):
                raise ValueError("恢复运行缺少 01_idea.json 缓存内容。")
            idea_data = cached_idea
            write_json(run_dir / "01_idea.json", idea_data)
            self._emit(
                "stage_restored",
                stage="idea",
                path=str(run_dir / "01_idea.json"),
                content=json.dumps(idea_data, ensure_ascii=False, indent=2),
            )
            restored_stages.append("idea")

        if stage_order[start_stage] > stage_order["outline"]:
            cached_outline = cache_payload.get("outline_data")
            if not isinstance(cached_outline, dict):
                raise ValueError("恢复运行缺少 02_outline.json 缓存内容。")
            outline_data = cached_outline
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

        if _should_execute("review"):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="review")
            review_title = (forced_title or "").strip()
            if not review_title:
                topic_lines = [
                    re.sub(r"\s+", " ", line).strip()
                    for line in topic_text.splitlines()
                    if line.strip()
                ]
                review_title = topic_lines[0] if topic_lines else "未命名题目"
            review_text, review_usage = self._run_literature_review_stage(
                topic_text=topic_text,
                paper_title=review_title,
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

        if _should_execute("idea"):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="idea")
            idea_data, idea_raw, idea_usage = self._run_idea_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                review_context=review_context,
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

        if _should_execute("outline"):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="outline")
            outline_data, outline_raw, outline_usage = self._run_outline_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                review_context=review_context,
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

        if _should_execute("paper"):
            self._ensure_not_cancelled()
            self._emit("stage_started", stage="paper")
            thesis_md, thesis_raw, paper_usage = self._run_paper_stage(
                topic_text=topic_text,
                literature_context=literature_context,
                review_context=review_context,
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
                "paper": paper_usage,
            },
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

    def _expand_literature_queries_with_llm(self, topic_text: str) -> list[str]:
        self._ensure_not_cancelled()
        user_prompt = (
            "请将以下论文题目改写为中英文检索标题。\n\n"
            f"题目：{topic_text.strip()}\n\n"
            "请按要求返回 JSON。"
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
        review_context: str,
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
                "review_context": review_context,
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
        review_context: str,
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
                "review_context": review_context,
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
        review_context: str,
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
                "review_context": review_context,
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
