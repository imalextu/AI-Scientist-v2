请基于以下输入生成管理学本科毕业论文详细大纲。

## 主题
{topic}

## 选题设计 JSON
{idea_json}

## 研究树搜索结论
{research_context}

## 参考文献线索
{literature_context}

## 输出要求
1. 输出严格 JSON，不要附加解释。
2. 按本科论文常见结构给出章节、节、核心论点、证据需求。
3. 给出每章建议字数和进度安排。

```json
{
  "title_cn": "中文题目",
  "title_en": "English Title",
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
