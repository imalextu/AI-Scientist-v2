请基于以下输入生成管理学本科毕业论文详细大纲。

## 主题
{topic}

## 选题设计 JSON
{idea_json}

## 文献综述结论
{review_context}

## 参考文献线索
{literature_context}

## 对策建议型增强输入（若非对策建议型可忽略）
{policy_support_context}

## 输出要求
1. 输出严格 JSON，不要附加解释。
2. 必须读取选题设计中的 `论文类型判定.recommended_type`，并按类型生成分支化章节结构。
3. 若 `论文类型判定` 缺失或无效，默认按 `对策建议型` 生成。
4. 按本科论文常见结构给出章节、节、核心论点、证据需求。
5. 给出每章建议字数和进度安排。
6. 三类分支至少包含以下核心章节（名称可微调，但语义必须保留）：
   - `定量实证型`：理论框架与研究假设、研究设计(问卷)、数据分析与假设检验
   - `案例研究型`：分析框架构建、案例数据收集、案例分析
   - `对策建议型`：轻量理论与现状调研、问题诊断、对策设计
7. 若为 `对策建议型`，必须在 JSON 增加 `policy_support` 字段，完整包含三部分：
   - `web_research`（企业/行业信息采集结果）
   - `three_layer_analysis`（现状-问题-原因三层分析）
   - `countermeasure_system`（对策体系设计）

```json
{
  "title_cn": "中文题目",
  "title_en": "English Title",
  "paper_type_plan": {
    "type": "定量实证型/案例研究型/对策建议型",
    "fallback_used": false,
    "branch_path": [
      "步骤1",
      "步骤2",
      "步骤3"
    ]
  },
  "policy_support": {
    "web_research": {},
    "three_layer_analysis": {},
    "countermeasure_system": {}
  },
  "chapters": [
    {
      "id": "第1章",
      "name": "绪论",
      "target_words": 1800,
      "sections": [
        {
          "heading": "1.1 研究背景",
          "key_points": [
            "关键论点1",
            "关键论点2"
          ],
          "evidence_needed": [
            "证据类型1",
            "证据类型2"
          ]
        }
      ]
    }
  ],
  "method_plan": {
    "data_collection_steps": [
      "步骤1",
      "步骤2"
    ],
    "analysis_steps": [
      "步骤1",
      "步骤2"
    ]
  },
  "schedule": [
    {
      "week": "第1周",
      "task": "任务",
      "deliverable": "产出"
    }
  ]
}
```
