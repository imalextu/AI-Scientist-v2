你将帮助管理学本科生完成毕业论文“选题设计”，并在文献综述前确定研究问题、理论锚点与论文类型分支路径。

## 输入信息
- 预设题目（如有）：{paper_title}
- 主题描述：{topic}
- 学科：{domain}
- 受众：{audience}
- 写作语言：{language}
- 目标字数：{min_words} - {max_words}
- 参考文献线索：
{literature_context}

## 任务要求
1. 对题目做可研究性评估，至少覆盖：
   - 是否可证伪
   - 变量是否可操作化
   - 数据是否可得
2. 生成 3-5 个候选研究问题（RQ），每个附带：
   - 理论切入点
   - 预期贡献
   - 难度评级（低/中/高）
3. 推荐最优 RQ 并说明理由。
4. 输出确定的研究问题 + 初步理论锚点（如资源基础观、制度理论、AMO 框架等）。
5. 新增并强制输出“论文类型判定”字段，根据以下三类三选一自动推荐：
   - `定量实证型`
   - `案例研究型`
   - `对策建议型`
6. “论文类型判定”必须覆盖三类判断依据：
   - 题目特征（变量关系、机制检验需求、研究对象边界）
   - 数据条件（可获取的数据类型、样本规模、时间与资源约束）
   - 学生偏好（若输入中未明确，写“未明确”）
7. 若信息不足或冲突导致无法稳定判定，必须兜底推荐 `对策建议型`，并在 `fallback_used` 标记为 `true`。
8. 根据推荐类型给出分支路径：
   - `定量实证型`：理论框架+假设 -> 研究设计(问卷) -> 数据分析
   - `案例研究型`：分析框架构建 -> 案例数据收集 -> 案例分析
   - `对策建议型`：轻量理论+现状调研 -> 问题诊断 -> 对策设计
9. 方法建议必须本科可执行，优先问卷、访谈、案例、公开数据库等可得数据来源，并与推荐论文类型一致。

## 输出格式
严格输出 JSON，不要输出任何额外解释。结构如下：

```json
{
  "thesis_title_cn": "中文题目",
  "thesis_title_en": "English Title",
  "researchability_assessment": {
    "falsifiability": {
      "rating": "高/中/低",
      "reason": "是否可证伪及原因"
    },
    "operationalization": {
      "rating": "高/中/低",
      "key_variables": [
        "变量1",
        "变量2"
      ],
      "reason": "变量可操作化判断"
    },
    "data_availability": {
      "rating": "高/中/低",
      "possible_data_sources": [
        "数据来源1",
        "数据来源2"
      ],
      "reason": "数据可得性判断"
    },
    "overall_feasibility": "高/中/低",
    "major_risks": [
      "风险1",
      "风险2"
    ]
  },
  "candidate_rqs": [
    {
      "id": "RQ1",
      "question": "候选研究问题",
      "theoretical_entry_point": "理论切入点",
      "expected_contribution": "预期贡献",
      "difficulty_rating": "低/中/高"
    }
  ],
  "recommended_rq": {
    "id": "RQ2",
    "question": "推荐研究问题",
    "selection_reason": "推荐理由",
    "preliminary_theoretical_anchors": [
      "理论锚点1",
      "理论锚点2"
    ]
  },
  "final_research_question": "最终确定研究问题",
  "theoretical_basis": [
    "理论基础1",
    "理论基础2"
  ],
  "论文类型判定": {
    "recommended_type": "定量实证型/案例研究型/对策建议型",
    "decision_basis": {
      "topic_features": "题目特征判断",
      "data_conditions": "数据条件判断",
      "student_preference": "学生偏好（若无则写未明确）"
    },
    "selection_reason": "综合推荐理由",
    "fallback_used": false,
    "branch_path": [
      "步骤1",
      "步骤2",
      "步骤3"
    ]
  },
  "study_objectives": [
    "目标1",
    "目标2",
    "目标3"
  ],
  "methodology": {
    "approach": "总体方法（需与推荐论文类型一致）",
    "sample": "样本设计与规模建议",
    "data_sources": [
      "数据来源1",
      "数据来源2"
    ],
    "analysis_methods": [
      "分析方法1",
      "分析方法2"
    ]
  }
}
```
