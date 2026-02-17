# 管理学本科论文生成器（中文）

这是一个独立于主仓库的新项目，结构参考当前 `AI-Scientist-v2` 的实现思路：  
`配置驱动 + 分阶段工作流 + prompts 模板 + 产物落盘`。

项目目标是生成中文管理学本科论文初稿，流程分为三步：
1. 选题与研究设计生成（`01_idea.json`）
2. 论文大纲生成（`02_outline.json`）
3. 论文正文草稿生成（`03_thesis.md`）

## 目录结构

```text
management-paper-agent/
├── config.example.yaml
├── examples/
│   └── topic_example.md
├── paper_agent/
│   ├── config.py
│   ├── llm_client.py
│   ├── retrieval.py
│   ├── utils.py
│   └── workflow.py
├── prompts/
│   ├── idea_generation.md
│   ├── outline.md
│   └── paper_writer.md
├── requirements.txt
└── run.py
```

## 快速开始

1. 安装依赖

```bash
cd management-paper-agent
pip install -r requirements.txt
```

2. 配置模型

- 默认使用 OpenAI 兼容接口（如 OpenRouter）。
- 建议设置环境变量：

```bash
export OPENROUTER_API_KEY="YOUR_KEY"
```

3. 运行生成

```bash
python run.py --topic-file examples/topic_example.md
```

或直接传入题目描述：

```bash
python run.py --topic "数字化转型背景下中小企业组织韧性提升机制研究"
```

生成结果会写入 `outputs/<timestamp>_<title-slug>/`。

## 可选参数

```bash
python run.py \
  --config config.example.yaml \
  --topic-file examples/topic_example.md \
  --title "数字化转型背景下中小企业组织韧性提升机制研究" \
  --model "anthropic/claude-3.5-sonnet" \
  --output-dir outputs \
  --disable-retrieval
```

## 说明

- 文献检索支持 `crossref` / `semantic_scholar` / `arxiv`，可在配置里切换。
- 该项目默认只生成 Markdown 草稿，不自动导出 Word/PDF。
- 如需学院格式，可在 `prompts/paper_writer.md` 中继续补充版式要求。
