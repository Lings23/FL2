# RTC-v3 B3 anchor recycle 分析说明

该目录比较严格配对的 `rtc_cumulative_q_cap` 与
`rtc_cumulative_q_cap_anchor`，攻击为 strong DBA 和 strong scaling backdoor。
B3 已完成但被拒绝：完整 anchor recycle 将 DBA active ASR 增加 49.792 pp、
将 scaling active ASR 增加 25.524 pp，尽管 ACC 提升且 zero mass 清零。

复算命令：

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' `
  'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3\analyze_results.py' `
  --experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3_seed42_mf03' `
  --output-dir 'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3'
```

预设门槛对两个攻击分别适用：active zero-update mass 至少减半或不高于 2%；
active ACC 不低于 B2；active ASR、恶意 impact、恶意权重的增幅均不超过
0.5 pp；anchor recycle 必须实际生效。另需全部完成性、质量门和严格配对通过。

输出包含 `comparison.csv`、`deltas.csv`、`client_group_summary.csv`、
`first_anchor_activation.csv`、`first_weight_divergence.csv`、
`asr_trajectory_events.csv`、`mass_balance_audit.csv` 和 `decision.json`。

技术报告输入和可审计快照由下列命令重建：

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' `
  'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3\build_artifact.py'
```

生成 `artifact.json` 与 `report_snapshot.sqlite`；`artifact.json` 已通过 Data
Analytics artifact validator，并作为单一 MCP report surface 渲染。
