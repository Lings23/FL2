# RTC-v3 B4 residual rank-cap 分析说明

B4 以已冻结的 B3R-F0.51 为基线，只增加一个 residual rank-cap 机制：每轮按
full-resolution residual norm 排名前 2 的客户端，其上限变为
`nominal_mass * min(existing_q_cap, 0.5)`。原有缺失质量继续按 0.51 路由到
accepted coordinate-median anchor；由 rank-cap 相对同轮无 rank-cap QP 新造成的
实际质量损失按 1.0 路由到同一 accepted anchor。在无其他约束时，两名客户端各从
0.1 降到 0.05，因此约 10% 质量进入 anchor。

最小实验共 3 个 seed42 单元：

1. LIE z=0.5 的 B3R-F0.51 基线（此前缺失，必须补跑）；
2. LIE z=0.5 的 B4 候选；
3. strong DBA 的 B4 候选，基线复用已完成的 B3R-F0.51 DBA。

人工训练完成后的唯一分析命令：

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' `
  'D:\workspace\FL2\analysis\rtc_residual_rank_cap_b4\analyze_results.py' `
  --lie-baseline-dir 'D:\workspace\FL2\logs\rtc_v3_residual_rank_cap_b4_lie_baseline_seed42_mf03' `
  --dba-baseline-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3r_f051_seed42_mf03' `
  --candidate-dir 'D:\workspace\FL2\logs\rtc_v3_residual_rank_cap_b4_seed42_mf03' `
  --output-dir 'D:\workspace\FL2\analysis\rtc_residual_rank_cap_b4'
```

预注册验收门：

- LIE active 恶意客户端权重不高于 15.5%，且必须低于基线；
- LIE 与 DBA active ACC 均不低于各自 B3R-F0.51 基线；
- DBA active ASR、恶意权重和恶意 impact 相对基线增幅均不超过 0.5 pp；
- LIE 恶意 impact 增幅不超过 0.5 pp；
- 两种攻击 zero-update mass 增幅均不超过 0.5 pp；
- 两种攻击的良性 rank-cap 命中率均不超过 10%，良性 restricted/quarantine
  增幅均不超过 0.5 pp；
- 每个活跃轮严格选中 residual norm 前 2 名，客户端 cap 不超过 0.5，reference QP
  全部有效，rank-recycle target 等于 rank-cap 实际移除质量，且单轮移除质量不超过 0.1；
- 3 个新实验单元均完成 60 轮、exit code=0、全部质量门通过；trial-plan hash、
  attack contract、初始模型、采样序列和恶意身份严格配对。

若任一门失败，B4 拒绝并回退到 B3R-F0.51；不得把 ACC 收益用于覆盖安全退化。
