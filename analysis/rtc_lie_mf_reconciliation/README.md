# LIE mean ACC 配置口径核对

结论：当前冻结的B3R-F0.51在LIE z=0.5、mf=0.3、round11–60上的mean ACC为
78.0796%，相对严格同trial-plan的原始RTC full 77.1174%提高0.9622个百分点，
不是倒退。

历史目录`logs/rtc_v3_lie_z05_seed42_mf03`虽然名字包含`mf03`，但其manifest、
run_id、Krum f和trim fraction均为0.2。该目录summary.csv中的79.0324%本身计算
正确，却属于不同恶意比例和trial plan，不能作为mf=0.3微调基线。

指标定义：active mean ACC为`planned_attack_active=1`的round11–60共50行
`server_accuracy`算术平均。三条数据均完成60轮、exit code=0且各自质量门通过。

严格可比对照：

- 原始RTC full：mf=0.3，trial-plan hash `de965e...`，active mean ACC 77.1174%。
- 当前B3R-F0.51：mf=0.3，同trial-plan hash，active mean ACC 78.0796%。
- 差值：+0.009622，即+0.9622个百分点；final ACC 81.92%→86.03%。

数据质量修复建议：实验目录后缀不得作为配置证据；汇总和分析入口应增加
`目录mf标签 == manifest.malicious_fraction == run_id m字段`的一致性门，历史误命名
目录应在文档中显式标记为mf=0.2，避免再次混用。
