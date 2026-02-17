请评估以下研究树搜索候选节点，判断其是否满足当前阶段目标。

## 研究主题
{topic}

## 当前阶段
- 阶段序号：{stage_index}
- 阶段标识：{stage_key}
- 阶段名称：{stage_name}
- 阶段目标：{stage_goal}
- 当前迭代：{iteration}

## 父节点（参考）
{parent_node}

## 待评估候选节点 JSON
{proposal_json}

## 文献线索
{literature_context}

## 评估要求
1. 综合考虑可执行性、逻辑完整性、创新性、与阶段目标匹配度。
2. `score` 必须是 0-100 的数值。
3. `decision` 只能是 `keep`、`revise` 或 `discard`。
4. 不允许输出解释文字，只输出 JSON。

## 输出 JSON 结构
```json
{
  "score": 78,
  "decision": "keep",
  "strengths": [
    "优势1"
  ],
  "weaknesses": [
    "不足1"
  ],
  "revision_actions": [
    "改进建议1"
  ],
  "reasoning": "简要评估理由"
}
```
