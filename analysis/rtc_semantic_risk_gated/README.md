# RTC-v3 B1R 风险门控筛选报告

更新时间：2026-09-02

## 结论

`semantic_intervention_risk_floor=0.10` 保住了 DBA 安全性，但没有充分回收
LIE 下的良性质量损失，因此不进入 B2。下一步仍属于 B1R，只把一个参数提高到
0.50，使语义机制仅对 restricted/quarantined 风险状态介入。

## 实验可信度

- 范围：seed42、恶意比例 0.3、IID、20 clients、participation 0.5、60 rounds、
  attack start round 11。
- LIE 固定 z=0.5；DBA 使用 strong contract。
- LIE 和 DBA 候选均完成 60 轮，exit code=0；11/11 质量门通过。
- LIE trial-plan hash：
  `de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`。
- DBA trial-plan hash：
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`。
- 这是单 seed 机制筛选，不是统计显著性结论。

## 结果

| 攻击 | 指标 | RTC full | floor=0.10 | 差值 |
|---|---:|---:|---:|---:|
| LIE | active ACC | 77.1174% | 77.1858% | +0.0684 pp |
| LIE | final ACC | 81.92% | 82.46% | +0.54 pp |
| LIE | 恶意权重 | 28.5556% | 28.5362% | -0.0194 pp |
| LIE | 恶意 impact | 31.4735% | 31.4187% | -0.0547 pp |
| LIE | 良性权重 | 66.9199% | 67.2665% | +0.3466 pp |
| LIE | zero-update mass | 4.5245% | 4.1973% | -0.3272 pp |
| DBA | active ACC | 83.9266% | 83.9052% | -0.0214 pp |
| DBA | active ASR | 2.2536% | 2.2302% | -0.0233 pp |
| DBA | 恶意权重 | 0.1083% | 0.1068% | -0.0014 pp |
| DBA | zero-update mass | 33.4469% | 33.5519% | +0.1049 pp |

LIE 的唯一失败门槛是 active ACC 增益不足 +0.3 pp。安全门全部通过。

## 机制诊断

LIE 首次分叉发生在 round26 的良性 cid18：risk=0.0590，full 权重为
0.09410，floor=0.10 候选恢复到 0.1。这证明 RTC 确实存在低风险良性质量损失，
但它不是主要剩余损失。

在 floor=0.10 候选中，risk≤0.1 的良性客户端轮次基本不再产生缺失质量；剩余
良性缺失质量主要来自 watch 状态：

| 风险桶 | 客户端轮次 | 每轮缺失质量 | 良性缺失质量占比 |
|---|---:|---:|---:|
| risk≤0.1 | 327 | 约 0 | 约 0% |
| 0.1<risk<0.5（watch） | 25 | 3.1983 pp | 77.4% |
| 0.5≤risk<0.8（restricted） | 5 | 0.9352 pp | 22.6% |
| risk≥0.8 | 0 | 0 | 0% |

因此，“RTC 错误损失良性质量”这个判断方向成立，但应精确为：当前 LIE 的主要
utility tax 来自少数良性客户端进入 watch 后仍受到 soft penalty、temporal q cap
和 hard exposure，而不是 risk≤0.1 的低风险客户端。

## 下一步实验边界

候选：`rtc_semantic_restricted_only`，唯一变化为
`semantic_intervention_risk_floor=0.50`。实验仍只有 LIE z=0.5 与 strong DBA
两个候选单元，并复用上述 RTC full 基线。

代码、405 项回归测试和 dry-run matrix 已通过。人工启动命令：

```powershell
& 'D:\workspace\FL2\experiments\run_rtc_semantic_restricted_only_ablation.ps1' -Execute
```

接受条件：LIE active ACC 至少提升 0.3 pp；DBA active ASR≤10%；DBA 恶意权重
≤5%；DBA active ACC 降幅≤0.2 pp；全部质量门通过。训练完成并人工恢复目标模式
前，不进入 B2。

恢复后使用以下命令分析。分析器会先强制检查两个候选均为 completed、exit code 0、
last round 60、轮次完整、质量门全过且 trial-plan hash 严格配对：

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' 'D:\workspace\FL2\analysis\rtc_semantic_risk_gated\analyze_results.py' --candidate-dir 'D:\workspace\FL2\logs\rtc_v3_semantic_restricted_only_seed42_mf03' --candidate-defense 'rtc_semantic_restricted_only' --output-dir 'D:\workspace\FL2\analysis\rtc_semantic_restricted_only'
```

## 可复现产物

- `comparison.csv`：full 与 floor=0.10 的核心指标。
- `client_group_summary.csv`：良性/恶意客户端分组权重、impact 和语义状态。
- `first_weight_divergence.csv`：首次权重分叉。
- `decision.json`：机器可读验收判定。
- `analyze_results.py`：支持指定候选目录/防御名/输出目录的配对分析器。
- `artifact.json` 与 `report_snapshot.sqlite`：报告快照。
