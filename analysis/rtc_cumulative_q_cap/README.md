# RTC-v3 B2 cumulative-q cap 验收报告

## 结论

`semantic_intervention_risk_floor=0.50` 上叠加线性
`cumulative_q_cap_power=1` 通过 B2 全部预设门槛，可以冻结并进入 B3。
它在不损失良性质量的情况下减少了 LIE 恶意质量，且没有改变 DBA 的安全轨迹。

## 完成性和配对

- seed42、恶意比例 0.3、IID、20 客户端、每轮采样 10 个、60 rounds。
- LIE 固定 z=0.5；DBA 使用冻结 strong contract；攻击期为 round 11–60。
- 两个候选均 `completed`、exit code 0、last round 60，各 61 行。
- 11/11 质量门通过。
- LIE trial-plan hash：
  `de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`。
- DBA trial-plan hash：
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`。

## 核心结果

| 攻击 | 指标 | B1R floor=0.5 | B2 linear q cap | 差值 |
|---|---:|---:|---:|---:|
| LIE | active ACC | 77.5966% | 78.3058% | +0.7092 pp |
| LIE | final ACC | 84.07% | 86.38% | +2.31 pp |
| LIE | 良性权重 | 71.4000% | 71.4000% | 0 pp |
| LIE | 恶意权重 | 28.5775% | 25.3088% | -3.2687 pp |
| LIE | 恶意 impact | 30.2719% | 27.7523% | -2.5195 pp |
| LIE | zero-update mass | 0.0225% | 3.2912% | +3.2687 pp |
| DBA | active ACC | 83.9258% | 83.9258% | 0 pp |
| DBA | final ACC | 87.50% | 87.50% | 0 pp |
| DBA | active ASR | 2.3753% | 2.3753% | 0 pp |
| DBA | 恶意权重 | 0.2540% | 0.2540% | 0 pp |
| DBA | 恶意 impact | 0.4310% | 0.4310% | 0 pp |

## 机制与误伤审计

两种攻击各 500 个活跃期客户端行均满足
`aggregation_weight <= nominal_mass * rtc_v3_client_q_cap`，容差 `1e-10`，无越界。
LIE 的 cap-active 客户端行共 61 个，全部为恶意客户端；良性命中为 0。
首次权重分叉发生在 round34 的恶意 cid7：`cumulative_q=0.991141`，权重从
10% 降至 9.9114%。DBA 恶意客户端虽被新 cap 命中，但其 quarantined 权重原本
已经低于 cap，所以候选与 B1R 轨迹完全一致。

这说明 B2 的提升不是继续“错误损失良性质量”，而是开始限制累计异常的恶意质量。
代价是 LIE zero-update mass 增至 3.2912%，这正是 B3 anchor recycle 的目标。

## 限制

当前只有 seed42，不能解释为多 seed 显著性结论。B2 尚未覆盖 scaling 或其他强攻击；
LIE 恶意权重仍为 25.31%，离 Multi-Krum 的历史参考约 12.3% 仍有差距。

## B3 下一步

冻结 floor=0.5 与 linear q cap，只增加 coordinate-median anchor recycle。
验收要求 zero-update mass 至少减半或不高于 2%，ACC 提升，且 ASR 和恶意 impact
不恶化。训练前需先完成代码/别名、测试、dry-run、分析器和唯一人工执行命令。
