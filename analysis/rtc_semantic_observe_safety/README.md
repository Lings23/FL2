# RTC semantic-observe 安全性复核与下一步微调

## 结论

`semantic_ablation=observe` 不能作为 RTC 的全局默认。它在 LIE z=0.5
下回收了一部分 ACC，但会移除 DBA 防御最关键的高风险语义约束，使 DBA
active ASR 从 2.25% 升至 97.08%。Gaussian noise 的语义风险接近零，开关
语义干预几乎没有影响。

更准确的原因是：RTC 当前把低于 watch 阈值的弱语义波动也送入 soft penalty
和 temporal q cap。LIE 的首个权重分叉发生在 round 26：良性 cid18 的
semantic risk 仅 0.059（watch 阈值为 0.1），但权重仍由 0.1 被 q cap 压到
0.0941；该轮恶意客户端的 semantic risk 都为 0。相反，DBA 首个攻击轮的
4 个恶意客户端 risk 为 0.917–0.969、q=0.3，完整 RTC 将其单客户端权重压至
0–0.0057，而 observe 恢复为 0.1。

因此，问题不是“RTC 一概错误损失良性质量”，而是“语义信号在 LIE 下方向
失配，而且低置信风险也触发了权重干预”。下一项微调应保留高风险语义防御，
仅让 risk≤0.1 的观测继续更新状态、但不进入 QP risk penalty 和 q cap。

## 配对结果（攻击 active rounds 11–60）

| 攻击 | 模式 | Active ACC | Final ACC | 恶意权重份额 | 恶意 impact 份额 | Active ASR |
|---|---:|---:|---:|---:|---:|---:|
| DBA | full | 83.9266% | 87.37% | 0.1083% | 0.1828% | 2.2536% |
| DBA | observe | 83.8384% | 88.53% | 28.5038% | 31.6511% | 97.0831% |
| Gaussian | full | 83.8064% | 87.07% | 25.2144% | 27.9850% | — |
| Gaussian | observe | 83.8240% | 87.25% | 25.4083% | 28.1399% | — |

DBA 的 final ACC 看似提升 1.16 pp，但这是以后门成功为代价，不能被当作有效
收益；安全排序必须先要求 ASR 达标，再比较 ACC。

## 数据质量说明

13 个质量门现已全部通过。原 DBA 门错误地要求同一轮每个活跃攻击者必须使用
不同 fragment；但 6 个固定恶意身份按 rank mod 4 映射到 4 个 DBA fragment，
同轮出现重复 fragment 是协议允许的。修正后的门验证：

1. 每个选中的恶意身份都有 fragment；
2. 观测 fragment 与 trial plan 的固定身份映射完全一致；
3. 每个活跃 DBA 客户端均执行 update scaling；
4. full/local ASR 指标齐全。

该修正只重算质量门，没有重跑训练。

## 下一项最小实验

候选 `rtc_semantic_risk_gated` 设置
`semantic_intervention_risk_floor=0.10`。只运行两个候选单元：LIE z=0.5 与
strong DBA；现有 full 基线被复用，trial-plan hash 已核对一致。

执行：

```powershell
& 'D:\workspace\FL2\experiments\run_rtc_semantic_risk_gated_ablation.ps1' -Execute
```

初筛通过条件：

- LIE active ACC 相对 full 至少提高 0.3 pp；
- DBA active ASR 不高于 10%，恶意权重份额不高于 5%；
- DBA active ACC 相对 full 不下降超过 0.2 pp；
- 所有质量门通过。

若通过，再进入多 seed 验证；若 DBA 安全性未保留，则下一步只调低 floor，不
同时改动其他 RTC 参数。
