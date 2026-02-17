你正在执行管理学论文的研究树搜索阶段，请生成一个候选研究节点。

## 研究主题
{topic}

## 当前阶段
- 阶段序号：{stage_index}/{stage_total}
- 阶段标识：{stage_key}
- 阶段名称：{stage_name}
- 阶段目标：{stage_goal}
- 当前迭代：{iteration}
- 分支编号：{branch_index}

## 父节点（用于继承）
{parent_node}

## 当前阶段已探索节点摘要（用于避免重复）
{current_tree_snapshot}

## 文献线索
{literature_context}

## 生成要求
1. 必须满足当前阶段目标。
2. 与父节点保持连续，但要有明确改进点。
3. 方案必须适配本科阶段可执行条件。
4. 不允许输出解释文字，只输出 JSON。

## 输出 JSON 结构
```json
{
  "candidate_title": "候选方案名称",
  "research_question": "核心研究问题",
  "focus_boundary": "研究对象与边界定义",
  "core_hypotheses": [
    "假设1",
    "假设2"
  ],
  "variable_design": {
    "independent_variables": [
      "自变量1"
    ],
    "dependent_variables": [
      "因变量1"
    ],
    "control_variables": [
      "控制变量1"
    ]
  },
  "method_design": {
    "approach": "定量/定性/混合",
    "sample_plan": "样本规模与来源",
    "data_sources": [
      "来源1"
    ],
    "analysis_steps": [
      "步骤1",
      "步骤2"
    ]
  },
  "expected_contributions": [
    "预期贡献1"
  ],
  "risk_controls": [
    "风险与应对1"
  ],
  "next_expansion_hint": "下一轮可扩展方向"
}
```
