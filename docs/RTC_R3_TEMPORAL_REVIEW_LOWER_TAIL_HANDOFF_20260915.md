# R3跨轮观测验收与小残差持续性cap人工交接

日期2026-09-15。R3-temporal观测通过质量验收，未接受新的防御。M2=B0+R1c+R2保持已接受主线。
下一候选只新增持续低trainable残差cap，未训练、未验证收益。所有真实实验仍由人工执行。

## 已完成观测与否定的假设

服务器10项均completed/exit0/last_round60；每项61轮、600条客户端记录，42/42 runner质量门通过。
220份归档源码与25份计划产物全部匹配；协议及M2接受凭据未变。四组父版本/观测版严格配对通过，
逐轮ACC/loss/聚合sketch、逐客户端权重与原有cap完全一致，独立因果历史重算一致。
实际服务器仍为/home/jia_zhang/hqr/FL2，Python3.11.16/torch2.14.0+cu130/numpy2.4.6/双RTX5090。

冻结分析器有一个CSV类型表示问题：512/True在Flower日志成为512.0/1.0。离线导入审计用有限数值精确比较，
并新增测试拒绝511.9、NaN及布尔2.0；未放宽任何阈值，也未修改原始锁、日志或归档源码。
完整审计analysis/rtc_r3_temporal_review/verify.py、review.json；逐条因果分数causal_scores.csv。
原锁SHA256：f22fb117240047a63aac80edeae5968188f2fa6b0c54cacdc8807db278595dbd。
原r3t_sources.zip SHA256：4e8ce4c413662d446be7ef55ffb7259bcf855e84d0f190a3e6c4e0a702fa26cc。

高方向持续性假设未获支持：clean103/104有效分数均值约.5842/.5856；LIE44/45攻击有效分数约.4842/.5109。
LIE攻击方向有效历史仅24/179和97/150；51/179、16/150条当前残差为零，无定义方向不能填0当正常方向。
因此不对“高coherence”直接设置cap，也不将观察通过解释为原假设成立。

新的线索是trainable残差接近当前nominal coordinate-median anchor：LIE44/45恶意残差中位数约.000090/.185620，
良性约3.1396/3.2107。旧累计统计使用全浮点残差且只检验过大值，不能假定对很小的trainable残差敏感。
原始幅度、方向与跨轮机制均不修改；新候选单独评估低尾持续性。

## 只用clean确定的固定规则

在读取阈值与独立holdout结果前写入lower_tail_design.json：
1. 每principal聚合同轮客户端的trainable残差范数，按名义质量加权；参考排除整个自身principal。
2. 至少9个其他正名义质量principal，参考是其残差范数中位数，包含零值；中位数≤1e-12或参考不足时弃权并清空连续计数。
3. ratio=自身残差/参考中位数；阈值取clean103全部600条有效ratio的最小值，严格小于阈值才是低尾。
4. 连续两次参与均低尾才令q_lower=0；正常或无效观测重置，不参与时不增加计数，不按攻击标签重置历史。
5. 在现有cap之后取min(existing_q,q_lower)，保留全部预算及accepted-anchor .51回填逻辑。

冻结阈值0.8738694595117251、连续次数2、cap0；校准config/rtc_r3_lower_tail_calibration.json。
canonical SHA256：666ef4c6c31726fdf206deeed88947c802b433cbf997ba0493a164c82d64ba71。
clean103与独立clean104均0/600新增标记。旧LIE44/45离线标记163/179、133/150，良性0/321、0/350，首标记均round12。
生产模块与独立标量重放一致，记录在lower_tail_screen.json。以上不是闭环训练结果，也不证明ACC收益。

较小残差不在一般意义上等于恶意；诚实更新也可能接近中心，尤其非IID或高度相似数据。
全体残差为零时参考退化会弃权；其他合法小残差仍存在误伤风险，须由新闭环及后续非IID验证约束。
不能将该候选称为已防住所有慢攻击或已具备理论保证。

## 新批次24项

固定seeds201、202；本地未发现对应旧训练产物。原拟用的49/52已出现于旧校准/smoke，因此在新训练前改选，
没有根据新结果筛seed。两个seed均为本轮工程验证，最终泛化仍须另行设计。

| 每个seed | 方法 | 单元 |
|---|---|---:|
| clean | M2+低尾observe、M2+低尾cap | 2 |
| Sign-flip scale1 | 同上、MK | 3 |
| Gaussian mean0/std.1 | 同上、MK、RFA | 4 |
| LIE z.5/all_updates | 同上、MK | 3 |

共24个新单元，旧基线复用0。父版本别名rtc_i12_lower_observe，候选rtc_i12_lower_cap；
二者配置只差lower_tail_mode。父版本的新增observe已用多轮合成聚合验证模型、权重、累计状态均不变。
CIFAR10/ResNet18无预训练、IID20客户端每轮10人、60轮、本地5轮、batch48；
SGD lr.01/momentum.9/decay.0001/cosine；恶意比例.3，攻击round11--60；strict/principal_uniform/确定性训练。
M2两个原cap继续开启；floor.5、cumulative power1、MAD2.5、accepted回填.51。
MK f3选5；RFA迭代3/smoothing1e-6/按样本数加权。
每client CPU1/GPU.25、object store3072MiB、最低可用内存10240MiB、等待120秒、重试0；不得静默改资源或训练参数。

预注册门以config/rtc_r3_lower_tail_protocol.json为准，各seed分别全部满足：
- LIE active ACC相对M2至少+1个百分点，final不下降，恶意权重至多父版本80%。
- clean/Sign-flip/Gaussian的active及final ACC退化均不超过0.2个百分点。
- Sign-flip/Gaussian恶意权重增加≤0.001；父版本及候选实际联合新增cap良性误标各≤1%。
- 24项完整，全部适用runner门、冻结合同、严格配对、有限性及双cap/低尾状态独立重放。
- 预算≤1e-8，几何重构相对误差≤1e-4，旧累计重放≤1e-9；低尾streak/flag/q精确一致。
失败只拒绝新增低尾cap，保留M2；禁止事后放宽门、改seed、自动补跑。每个条件报告与MK/RFA差距，不声称通过即胜过对手。
LIE .25、Min-Max/Min-Sum、Random-v2、label reversal、DBA/scaling及最终新seed/非IID仍待完成。

## 验证与冻结

23项纯合成测试通过，含事务式状态、两次触发/恢复、零参考、principal别名去重、无效数值、
phase6预算与cap落实、observe惯性、旧M2/方向/幅度回归、损坏低尾证据拒收、新增及旧攻击退化拒收。
PowerShell语法和24项dry-run通过；Bash -n通过、LF换行。缺少真实结果时分析器拒收已核实。
Linux实机dry-run尚未执行；Windows验证不替代Linux验证。
Windows目录D:\workspace\FL2\logs\rtc_r3_lower_tail，245份源码、51份冻结产物、status/raw均0。
锁SHA256：3d70d7b038c755eaf749a3ea045de16264221f042e58e03a6dba6e6e2ebee807。
r3l_sources.zip SHA256：39519784664172a5601109cd542946b871e8704422cabd5625f14b0bb8c878e9。
Windows环境Python3.11.9/torch2.12.1+cu126/numpy2.4.4/RTX4060；Linux按实际服务器另行prepare。

## 手动Git同步

依用户最新偏好，本轮未自动提交/推送。包括此前尚未提交的R3观测依赖；只加入下列明确任务文件。
先确认暂存区没有其他任务内容，再执行。大日志/数据集不提交；新批次源码归档随日志传回。

```powershell
Set-Location 'D:\workspace\FL2'
git diff --cached --stat
git add -- defenses/rtc/v3.py defenses/rtc/temporal_observe.py defenses/rtc/lower_tail.py strategies/fed_strategy.py experiments/rtc_v3/byzantine.py experiments/rtc_r3_temporal_observation.py experiments/rtc_r3_lower_tail_stage.py experiments/run_rtc_r3_temporal_observation.ps1 experiments/run_rtc_r3_temporal_observation.sh experiments/run_rtc_r3_lower_tail.ps1 experiments/run_rtc_r3_lower_tail.sh tests/test_rtc_r3_temporal_observation.py tests/test_rtc_r3_lower_tail.py tests/test_rtc_r3_lower_tail_stage.py config/rtc_i12_accepted.json config/rtc_r3_temporal_observation.json config/rtc_r3_lower_tail_calibration.json config/rtc_r3_lower_tail_protocol.json analysis/rtc_i12_review/audit_import.py analysis/rtc_i12_review/review.json analysis/rtc_r3_temporal_review/verify.py analysis/rtc_r3_temporal_review/review.json analysis/rtc_r3_temporal_review/lower_tail_design.json analysis/rtc_r3_temporal_review/lower_tail_screen.json analysis/rtc_r3_temporal_review/causal_scores.csv
git add -f -- docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md docs/RTC_I12_REVIEW_R3_TEMPORAL_HANDOFF_20260915.md docs/RTC_R3_TEMPORAL_REVIEW_LOWER_TAIL_HANDOFF_20260915.md
git diff --cached --check
git commit -m "Validate RTC temporal observations and prepare persistent lower-tail cap"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin HEAD:refs/heads/codex/periodic-attack-defenses
```

## 人工运行

Windows已验证入口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_r3_lower_tail.ps1' -Execute
```

完成后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_r3_lower_tail.ps1' -Analyze
```

Linux更新后先只验证：

```bash
cd /home/jia_zhang/hqr/FL2
git pull --ff-only origin codex/periodic-attack-defenses
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_r3_lower_tail.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

返回Linux dry-run及锁/环境后核验，再交接真实训练命令。两平台是替代入口，勿重复两套24项。
Linux输出/home/jia_zhang/hqr/FL2/logs/rtc_r3_lower_tail，结果与r3l_lock.json/r3l_sources.zip一并保留。
实际训练完成后在原服务器分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_r3_lower_tail.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

已完成单元经runner验证后可跳过；失败/未完成/孤儿产物拒绝自动重跑，需人工复核。
不支持中间轮checkpoint续训。当前阶段WAITING_FOR_MANUAL_EXPERIMENT；新边界计数1，正式blocked需宿主连续三回合。
未启动训练，不后台等待，不推进依赖这批结果的后续机制。整体RTC改进目标未完成。
