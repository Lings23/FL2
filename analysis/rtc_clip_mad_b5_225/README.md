# RTC-v3 B5 clipping MAD=2.25 分析说明

B5 从已冻结的 B3R-F0.51 出发，只将 norm clipping 阈值中的 MAD 系数由默认
`2.5` 调为 `2.25`。不包含已拒绝的 B4 residual rank-cap，也不改 floor、线性
cumulative-q cap、accepted-anchor weighting 或 recycle fraction=0.51。

最小实验只有两个 seed42 candidate 单元：LIE z=0.5 与 strong DBA。两者基线均
严格复用已有 B3R-F0.51；每个运行的 round1–10 尚未启用攻击，作为同一 trial plan
下的 clean clipping 安全窗口，因此无需另跑不配对的 clean 单元。

人工训练完成后的唯一分析命令：

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' `
  'D:\workspace\FL2\analysis\rtc_clip_mad_b5_225\analyze_results.py' `
  --lie-baseline-dir 'D:\workspace\FL2\logs\rtc_v3_residual_rank_cap_b4_lie_baseline_seed42_mf03' `
  --dba-baseline-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3r_f051_seed42_mf03' `
  --candidate-dir 'D:\workspace\FL2\logs\rtc_v3_clip_mad_b5_225_seed42_mf03' `
  --output-dir 'D:\workspace\FL2\analysis\rtc_clip_mad_b5_225'
```

预注册门：两攻击 pre-attack clean clipping rate 与 active benign clipping rate 均
严格低于 8%；LIE active ACC 严格改善，LIE/DBA active ACC 均不低于基线；DBA
active ASR、两攻击恶意权重/impact、zero mass、良性 restricted/quarantine 相对
基线增幅均不超过 0.5 pp；运行时 MAD k 必须恒为2.25；两个单元完成60轮、全部
质量门、trial-plan/attack-contract和单参数差异检查通过。

任一门失败即拒绝2.25并回退B3R-F0.51。只有失败机制证据明确支持进一步收紧时，
才另行准备独立的2.0实验；本阶段不预先运行2.0。

## 已完成判定

两个候选均完成60轮并通过全部质量门，但`accepted_for_v1=false`。LIE active ACC
下降0.0390 pp，DBA下降0.2438 pp；DBA攻击前clean clipping为8%，不满足严格
`<8%`。LIE恶意权重/impact虽下降0.2979/0.3299 pp，但不足以覆盖效用门失败。
因此拒绝2.25，不运行更激进的2.0，最终候选仍为B3R-F0.51（默认MAD k=2.5）。

技术报告的可复算源为`artifact.json`、`report_snapshot.sqlite`和`build_artifact.py`；
artifact validator已通过并已在MCP report surface渲染。
