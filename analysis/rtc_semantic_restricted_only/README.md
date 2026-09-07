# RTC-v3 B1R restricted-only 验收报告

## 结论

`semantic_intervention_risk_floor=0.50` 通过 B1R 全部预设门槛，可以冻结并进入
B2。它恢复了 LIE 下被错误压低的良性 watch 状态质量，同时没有重现全局 observe
在 DBA 上的安全失守。

## 完成性和配对

- seed42、恶意比例0.3、IID、20客户端、每轮采样10个、60 rounds。
- LIE 固定 z=0.5；DBA 使用冻结 strong contract；攻击期为round 11–60。
- 两个候选均 `completed`、exit code 0、last round 60，各61行。
- 11/11质量门通过。
- LIE trial-plan hash：
  `de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`。
- DBA trial-plan hash：
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`。

## 核心结果

| 攻击 | 指标 | RTC full | floor=0.50 | 差值 |
|---|---:|---:|---:|---:|
| LIE | active ACC | 77.1174% | 77.5966% | +0.4792 pp |
| LIE | final ACC | 81.92% | 84.07% | +2.15 pp |
| LIE | 良性权重 | 66.9200% | 71.4000% | +4.4801 pp |
| LIE | 恶意权重 | 28.5556% | 28.5775% | +0.0219 pp |
| LIE | 恶意 impact | 31.4735% | 30.2719% | -1.2016 pp |
| LIE | zero-update mass | 4.5245% | 0.0225% | -4.5020 pp |
| DBA | active ACC | 83.9266% | 83.9258% | -0.0008 pp |
| DBA | final ACC | 87.37% | 87.50% | +0.13 pp |
| DBA | active ASR | 2.2536% | 2.3753% | +0.1218 pp |
| DBA | 恶意权重 | 0.1083% | 0.2540% | +0.1457 pp |
| DBA | zero-update mass | 33.4469% | 31.5460% | -1.9009 pp |

## 机制解释

LIE 首次分叉是round26良性cid18：两组 semantic risk 均为5.9031%，候选将权重
从9.4097%恢复到10%。候选的良性权重达到每轮所选良性客户端的完整 nominal
质量，而恶意权重基本不变，说明 ACC 收益主要来自解除良性 watch 状态误伤。

DBA 首次分叉发生在round11两名恶意客户端；其 risk 为93.35%和91.71%，仍属于
quarantined。候选略微放宽其权重，但active ASR、恶意权重和ACC均通过预设安全门。

## 限制

当前只有seed42，不能解释为多seed显著性结论。B1R仍未降低LIE的28.58%恶意
聚合权重，因此只解决了utility tax，没有解决inlier恶意质量过高。

## B2 唯一下一步

冻结floor=0.5，只加入线性客户端cap：

```text
cap_i = nominal_i × min(semantic_q_i, cumulative_q_i, direction_q_i)
cumulative_q_cap_power = 1
```

离线重放预计LIE恶意权重约降至25.04%，良性权重保持71.4%；该估计没有模拟
QP重新分配和模型路径依赖，不能代替训练。

人工执行命令：

```powershell
& 'D:\workspace\FL2\experiments\run_rtc_cumulative_q_cap_ablation.ps1' -Execute
```

验收条件：LIE恶意权重至少下降3 pp且active ACC降幅≤0.2 pp；DBA active
ASR≤10%、恶意权重≤5%、active ACC降幅≤0.2 pp；全部质量门通过。训练完成前
不得进入B3。
