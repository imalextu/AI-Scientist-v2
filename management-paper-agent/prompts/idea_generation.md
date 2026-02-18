你将帮助管理学本科生完成毕业论文选题设计。

## 输入信息
- 主题描述：{topic}
- 学科：{domain}
- 受众：{audience}
- 写作语言：{language}
- 目标字数：{min_words} - {max_words}
- 文献综述结论：
{review_context}
- 参考文献线索：
{literature_context}

## 任务要求
1. 生成一个可落地的管理学本科论文选题方案。
2. 研究范围要适中，不要过大。
3. 方法要适配本科阶段可执行条件（问卷、访谈、案例、公开数据等）。
4. 明确潜在风险与应对方向。

## 输出格式
严格输出 JSON，不要输出任何额外解释。结构如下：

```json
{
  "thesis_title_cn": "中文题目",
  "thesis_title_en": "English Title",
  "problem_statement": "核心研究问题",
  "study_objectives": [
    "目标1",
    "目标2",
    "目标3"
  ],
  "theoretical_basis": [
    "理论基础1",
    "理论基础2"
  ],
  "methodology": {
    "approach": "总体方法（如定量/定性/混合）",
    "sample": "样本设计与规模建议",
    "data_sources": [
      "数据来源1",
      "数据来源2"
    ],
    "analysis_methods": [
      "分析方法1",
      "分析方法2"
    ]
  },
  "chapter_plan": [
    {
      "chapter": "第一章 绪论",
      "core_points": [
        "要点1",
        "要点2"
      ]
    }
  ],
  "innovation_points": [
    "创新点1",
    "创新点2"
  ],
  "feasibility_risks": [
    "风险1",
    "风险2"
  ]
}
```
