# LIE z=0.5 ACC 口径核对

本目录解释旧 `rtc_v3_lie_z05_seed42_mf03` 的 RTC active ACC 79.0324% 与当前冻结候选 78.0796% 为什么不能直接判为微调倒退。

计算口径：读取各 `rounds/*.csv`，筛选 round 11–60，共 50 个攻击活跃轮，对 `server_accuracy` 取算术平均；final ACC 取 round 60。所有四个运行均为 seed42、LIE z=0.5。

关键数据质量问题：旧目录名包含 `mf03`，但 `experiment_matrix.csv`、round CSV 文件名和逐轮字段均记录 `malicious_fraction=0.2`。当前正式微调口径为 `malicious_fraction=0.3`。旧结果还使用不同的 trial-plan 与 attack implementation hash。因此 79.0324% 和 78.0796% 不是配对实验。

当前 mf=0.3 路径内，原始 RTC active ACC 为 77.1174%，B2 线性 cumulative-q cap 为 78.3058%，冻结 B3R-F0.51 为 78.0796%。所以冻结方案相对当前原始 RTC 高 0.9622 pp，但相对 LIE 单项最优 B2 低 0.2262 pp。B3R-F0.51 的冻结依据是 DBA+scaling 的预注册跨攻击门，而不是 LIE 单项最高 ACC。

注意：当前原始 RTC 与 B2 的 attack contract、trial-plan、z 和恶意比例相同，但 attack implementation hash 不同；其 1.1884 pp 改善属于强描述性证据，不应当作唯一机制的严格因果估计。B2 与 B3R-F0.51 的 implementation hash 和 trial-plan 完全一致，因此二者 -0.2262 pp 的 LIE 差异是更严格的同口径对照。
