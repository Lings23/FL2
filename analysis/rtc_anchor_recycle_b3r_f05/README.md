# RTC-v3 B3R-F0.5 accepted-anchor 分析说明

该目录用于验收 `rtc_cumulative_q_cap_accepted_anchor` 的 fraction-only 重试。
相对已完成但 DBA ASR 门失败的 B3R，只把 `anchor_recycle_fraction` 从 `1.0`
改为 `0.5`；accepted weighting、B2 floor=0.5、linear cumulative-q cap、攻击合同
和 trial plan 均保持不变。

人工训练完成后的唯一分析命令：

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' `
  'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3r\analyze_results.py' `
  --baseline-experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3_seed42_mf03' `
  --reference-experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3r_seed42_mf03' `
  --candidate-experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3r_f05_seed42_mf03' `
  --expected-anchor-recycle-fraction 0.5 `
  --stage-name b3r_f05 `
  --output-dir 'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3r_f05'
```

门槛保持不变并分别适用于 DBA、scaling backdoor：active zero-update mass 至少
减半或不高于 2%；active ACC 不低于 B2；active ASR、恶意 impact、恶意权重
增幅均不超过 0.5 pp；anchor recycle 实际生效。另要求候选两单元完整、全部质量门
通过、与 B2/B3/B3R trial plan 严格一致，并确认运行时 weighting=`accepted`、
fraction=`0.5`。
