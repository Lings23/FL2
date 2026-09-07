# RTC-v3 B3R accepted-weight anchor 分析说明

该目录验收 `rtc_cumulative_q_cap_accepted_anchor`。B3R 相对失败的 B3 仅把
recycle anchor 的权重来源从 solver 前 nominal weights 改为 solver 后 accepted
weights；B2 floor=0.5、linear cumulative-q cap、full recycle fraction、攻击合同和
trial plan 均保持不变。

人工训练完成后的唯一分析命令：

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' `
  'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3r\analyze_results.py' `
  --baseline-experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3_seed42_mf03' `
  --candidate-experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3r_seed42_mf03' `
  --output-dir 'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3r'
```

预设门槛与 B3 相同并分别适用于 DBA、scaling backdoor：active zero-update mass
至少减半或不高于2%；active ACC不低于B2；active ASR、恶意impact、恶意权重
增幅均不超过0.5 pp；anchor recycle实际生效。另要求候选两单元完整、全部质量门
通过、与B2/B3 trial plan严格一致，并确认运行时weighting为`accepted`。
