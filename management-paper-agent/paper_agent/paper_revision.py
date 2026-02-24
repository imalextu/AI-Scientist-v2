from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from .llm_client import LLMClient, RequestCancelledError

DeltaCallback = Callable[[str], None]
CancelChecker = Callable[[], bool]
RequestEventCallback = Callable[[Dict[str, Any]], None]

PAPER_AUDIT_SYSTEM_PROMPT = """
你是管理学本科论文正文审核专家。请严格审阅论文初稿，并给出可执行的修改建议。
要求：
1. 仅依据输入正文与上下文审核，不得编造不存在的数据或来源。
2. 优先检查：研究问题一致性、论证链条完整性、方法描述可执行性、术语准确性、结论边界、格式规范。
3. 输出 Markdown，结构必须包含：
   - 审核结论
   - 必改问题清单（表格，列：序号 | 问题位置 | 问题描述 | 修改建议 | 优先级）
   - 可优化建议
   - 修改执行要点（可直接用于改写）
4. 建议要具体、可落地，避免空泛表述。
""".strip()

PAPER_REVISION_SYSTEM_PROMPT = """
你是管理学本科论文改写助手。请根据审核意见对正文进行一次完整修订。
要求：
1. 保持原题目、章节结构和学术语气。
2. 优先落实“必改问题清单”和“修改执行要点”。
3. 不得凭空补造可验证事实或统计结果。
4. 输出“修订后的完整论文 Markdown 正文”，不要输出解释、对照表或额外说明。
""".strip()


@dataclass
class PaperAuditRevisionResult:
    audit_text: str
    revised_text: str
    audit_raw: str
    revised_raw: str
    usage: Dict[str, Dict[str, int]]


class PaperRevisionService:
    def __init__(
        self,
        *,
        reviewer_client: LLMClient,
        reviser_client: LLMClient,
        continuation_tail_chars: int = 12000,
    ):
        self.reviewer_client = reviewer_client
        self.reviser_client = reviser_client
        self.continuation_tail_chars = max(0, int(continuation_tail_chars))

    def audit_paper(
        self,
        *,
        topic_text: str,
        paper_draft: str,
        idea_data: Dict[str, Any],
        outline_data: Dict[str, Any],
        review_context: str,
        literature_context: str,
        temperature: float | None,
        max_tokens: int | None,
        cancel_checker: CancelChecker | None = None,
        on_delta: DeltaCallback | None = None,
        on_request: RequestEventCallback | None = None,
        on_round_completed: RequestEventCallback | None = None,
    ) -> Tuple[str, str, Dict[str, int], str]:
        user_prompt = self._build_audit_prompt(
            topic_text=topic_text,
            paper_draft=paper_draft,
            idea_data=idea_data,
            outline_data=outline_data,
            review_context=review_context,
            literature_context=literature_context,
        )
        self._emit_request(
            on_request,
            operation="paper_audit",
            system_prompt=PAPER_AUDIT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            round_idx=1,
            model=self.reviewer_client.config.model,
        )
        raw_text, usage, finish_reason = self._safe_complete_with_meta(
            self.reviewer_client,
            system_prompt=PAPER_AUDIT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            on_delta=on_delta,
            cancel_checker=cancel_checker,
        )
        self._emit_round_completed(
            on_round_completed,
            operation="paper_audit",
            round_idx=1,
            finish_reason=finish_reason,
        )
        clean = self._strip_markdown_fence(raw_text).strip()
        usage_with_round = dict(usage)
        usage_with_round["rounds"] = 1
        return clean, raw_text, usage_with_round, finish_reason

    def revise_with_feedback(
        self,
        *,
        paper_draft: str,
        feedback_text: str,
        temperature: float | None,
        max_tokens: int | None,
        max_rounds: int,
        cancel_checker: CancelChecker | None = None,
        on_delta: DeltaCallback | None = None,
        on_request: RequestEventCallback | None = None,
        on_round_completed: RequestEventCallback | None = None,
    ) -> Tuple[str, str, Dict[str, int]]:
        usage_total = self._usage_template()
        raw_chunks: list[str] = []
        merged_text = ""
        rounds = max(1, int(max_rounds))

        current_prompt = self._build_revision_prompt(
            paper_draft=paper_draft,
            feedback_text=feedback_text,
        )

        for round_idx in range(rounds):
            current_round = round_idx + 1
            self._emit_request(
                on_request,
                operation="paper_revision",
                system_prompt=PAPER_REVISION_SYSTEM_PROMPT,
                user_prompt=current_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                round_idx=current_round,
                model=self.reviser_client.config.model,
            )
            raw_text, usage, finish_reason = self._safe_complete_with_meta(
                self.reviser_client,
                system_prompt=PAPER_REVISION_SYSTEM_PROMPT,
                user_prompt=current_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_delta=on_delta,
                cancel_checker=cancel_checker,
            )
            self._emit_round_completed(
                on_round_completed,
                operation="paper_revision",
                round_idx=current_round,
                finish_reason=finish_reason,
            )
            raw_chunks.append(raw_text)
            self._accumulate_usage(usage_total, usage)

            clean_chunk = self._strip_markdown_fence(raw_text)
            merged_text = self._append_with_overlap(merged_text, clean_chunk)

            if not self._needs_continuation(
                full_text=merged_text,
                finish_reason=finish_reason,
                round_idx=round_idx,
                max_rounds=rounds,
            ):
                break

            current_prompt = self._build_revision_continuation_prompt(merged_text)

        usage_total["rounds"] = len(raw_chunks)
        raw_joined = "\n\n\n<!-- REVISION CONTINUATION ROUND -->\n\n".join(raw_chunks).strip()
        return merged_text.strip(), raw_joined, usage_total

    def audit_and_revise_once(
        self,
        *,
        topic_text: str,
        paper_draft: str,
        idea_data: Dict[str, Any],
        outline_data: Dict[str, Any],
        review_context: str,
        literature_context: str,
        audit_temperature: float | None,
        audit_max_tokens: int | None,
        revision_temperature: float | None,
        revision_max_tokens: int | None,
        revision_max_rounds: int,
        cancel_checker: CancelChecker | None = None,
        on_audit_delta: DeltaCallback | None = None,
        on_revision_delta: DeltaCallback | None = None,
        on_request: RequestEventCallback | None = None,
        on_round_completed: RequestEventCallback | None = None,
    ) -> PaperAuditRevisionResult:
        audit_text, audit_raw, audit_usage, _ = self.audit_paper(
            topic_text=topic_text,
            paper_draft=paper_draft,
            idea_data=idea_data,
            outline_data=outline_data,
            review_context=review_context,
            literature_context=literature_context,
            temperature=audit_temperature,
            max_tokens=audit_max_tokens,
            cancel_checker=cancel_checker,
            on_delta=on_audit_delta,
            on_request=on_request,
            on_round_completed=on_round_completed,
        )
        revised_text, revised_raw, revised_usage = self.revise_with_feedback(
            paper_draft=paper_draft,
            feedback_text=audit_text,
            temperature=revision_temperature,
            max_tokens=revision_max_tokens,
            max_rounds=revision_max_rounds,
            cancel_checker=cancel_checker,
            on_delta=on_revision_delta,
            on_request=on_request,
            on_round_completed=on_round_completed,
        )

        usage_total = self._usage_template()
        self._accumulate_usage(usage_total, audit_usage)
        self._accumulate_usage(usage_total, revised_usage)
        usage_total["rounds"] = int(audit_usage.get("rounds", 0)) + int(
            revised_usage.get("rounds", 0)
        )

        return PaperAuditRevisionResult(
            audit_text=audit_text,
            revised_text=revised_text,
            audit_raw=audit_raw,
            revised_raw=revised_raw,
            usage={
                "audit": audit_usage,
                "revision": revised_usage,
                "total": usage_total,
            },
        )

    def _build_audit_prompt(
        self,
        *,
        topic_text: str,
        paper_draft: str,
        idea_data: Dict[str, Any],
        outline_data: Dict[str, Any],
        review_context: str,
        literature_context: str,
    ) -> str:
        return (
            "请审阅以下管理学本科论文初稿，并给出结构化审核意见。\n\n"
            f"论文主题：{topic_text}\n\n"
            "选题设计 JSON：\n"
            f"{self._pretty_json(idea_data)}\n\n"
            "论文大纲 JSON：\n"
            f"{self._pretty_json(outline_data)}\n\n"
            "文献综述结论：\n"
            f"{review_context}\n\n"
            "文献线索：\n"
            f"{literature_context}\n\n"
            "论文初稿：\n"
            "```markdown\n"
            f"{paper_draft}\n"
            "```\n"
        )

    def _build_revision_prompt(self, *, paper_draft: str, feedback_text: str) -> str:
        return (
            "请依据审核意见，对论文初稿进行一次完整修订并输出完整正文。\n\n"
            "审核意见：\n"
            "```markdown\n"
            f"{feedback_text}\n"
            "```\n\n"
            "论文初稿：\n"
            "```markdown\n"
            f"{paper_draft}\n"
            "```\n"
        )

    def _build_revision_continuation_prompt(self, current_text: str) -> str:
        tail = self._tail_text(current_text, self.continuation_tail_chars)
        missing_sections = []
        if "参考文献" not in current_text:
            missing_sections.append("参考文献")
        if "致谢" not in current_text:
            missing_sections.append("致谢")
        missing_hint = "、".join(missing_sections) if missing_sections else "后续未完成章节"

        return (
            "你正在续写同一份“修订后论文 Markdown”，上一轮可能因为长度限制被截断。\n"
            "请仅输出后续新增内容，不要重复前文。\n\n"
            "已生成内容末尾：\n"
            "```markdown\n"
            f"{tail}\n"
            "```\n\n"
            "要求：\n"
            "1. 从中断处自然续写，不要重写前文。\n"
            f"2. 优先补齐：{missing_hint}。\n"
            "3. 输出到整篇论文完整结束。\n"
        )

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
        if not self._has_tail_sections(full_text):
            return True
        if self._looks_cutoff(full_text):
            return True
        return False

    def _tail_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return text
        return text[-max_chars:]

    def _strip_markdown_fence(self, text: str) -> str:
        stripped = (text or "").strip()
        if not stripped.startswith("```"):
            return stripped
        pattern = r"^```(?:markdown|md)?\s*(.*?)\s*```$"
        match = re.match(pattern, stripped, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return stripped

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
        if right in left:
            return left
        return f"{left}\n\n{right}"

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

    def _has_tail_sections(self, text: str) -> bool:
        return "参考文献" in text and "致谢" in text

    def _usage_template(self) -> Dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rounds": 0,
        }

    def _accumulate_usage(self, total: Dict[str, int], usage: Dict[str, int]) -> None:
        total["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        total["completion_tokens"] += int(usage.get("completion_tokens", 0))
        total["total_tokens"] += int(usage.get("total_tokens", 0))

    def _pretty_json(self, value: Any) -> str:
        try:
            import json

            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)

    def _emit_request(
        self,
        callback: RequestEventCallback | None,
        *,
        operation: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        round_idx: int | None,
        model: str,
    ) -> None:
        if not callback:
            return
        callback(
            {
                "operation": operation,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "round": round_idx,
                "model": model,
            }
        )

    def _emit_round_completed(
        self,
        callback: RequestEventCallback | None,
        *,
        operation: str,
        round_idx: int,
        finish_reason: str,
    ) -> None:
        if not callback:
            return
        callback(
            {
                "operation": operation,
                "round": round_idx,
                "finish_reason": finish_reason,
            }
        )

    def _safe_complete_with_meta(
        self,
        client: LLMClient,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        on_delta: DeltaCallback | None,
        cancel_checker: CancelChecker | None,
    ) -> Tuple[str, Dict[str, int], str]:
        try:
            return client.complete_with_meta(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_delta=on_delta,
                cancel_checker=cancel_checker,
            )
        except RequestCancelledError:
            raise
