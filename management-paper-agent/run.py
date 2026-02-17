#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paper_agent.config import AppConfig, load_config
from paper_agent.workflow import ThesisWorkflow


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_input_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    script_relative = SCRIPT_DIR / candidate
    return script_relative.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="中文管理学本科论文生成器")
    parser.add_argument(
        "--config",
        type=str,
        default="config.example.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="论文主题描述（与 --topic-file 二选一）",
    )
    parser.add_argument(
        "--topic-file",
        type=str,
        default="",
        help="论文主题描述文件路径（与 --topic 二选一）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="强制指定中文论文题目（可选）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="覆盖配置中的模型名（可选）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="覆盖配置中的输出目录（可选）",
    )
    parser.add_argument(
        "--disable-retrieval",
        action="store_true",
        help="关闭文献检索",
    )
    return parser.parse_args()


def load_topic(topic: str, topic_file: str) -> str:
    if topic.strip():
        return topic.strip()
    if topic_file.strip():
        topic_path = resolve_input_path(topic_file)
        return topic_path.read_text(encoding="utf-8").strip()
    raise ValueError("请提供 --topic 或 --topic-file")


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.model.strip():
        config.llm.model = args.model.strip()
    if args.output_dir.strip():
        config.paper.output_dir = args.output_dir.strip()
    if args.disable_retrieval:
        config.retrieval.enabled = False
    return config


def main() -> None:
    args = parse_args()
    config_path = resolve_input_path(args.config)
    config = load_config(config_path)
    config = apply_overrides(config, args)
    topic_text = load_topic(args.topic, args.topic_file)
    forced_title = args.title.strip() if args.title.strip() else None

    workflow = ThesisWorkflow(config)
    run_dir = workflow.run(topic_text=topic_text, forced_title=forced_title)
    print(f"论文草稿已生成：{run_dir}")


if __name__ == "__main__":
    main()
