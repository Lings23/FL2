# H2 batch96验收与H2b历史写入修复交接

日期2026-09-17。用户指定检查logs/rtc_reference_eligibility，该目录已是服务器完整结果。
H2质量通过、候选拒绝但保留；H2b未运行，当前WAITING_FOR_MANUAL_EXPERIMENT。

## 实际合同与验收

24/24计划run_id的status为completed、exit0、last_round60，每项61个唯一轮号、600条客户端记录，92项runner门通过。
250份归档源码、52份冻结产物哈希核验，原R1c/R2、资格状态、累计、预算、质量与严格配对重放通过。
独立重建14400条谱参考，最大cosine差8.8913e-12；本地与服务器31项候选门一致，29通过、2失败。
原始rounds CSV把2/3整数元数据写为2.0/3.0；离线导入仅规范化这两个整数字段的字符串表示，
共1920处，拒绝小数与非有限值，未改原日志、判断值或验收阈值。所有规范化条目记录于review。
审计analysis/rtc_reference_eligibility_review/verify.py、review.json；派生浮点比较1e-12，布尔/结构精确匹配。
锁65de3cb6680acff790ab3f4a97a6a0d34ac94cdcbd880c18d410c3a324fa436e；
源码zip 7638f68db62e8327bc54cae37e2691332b97d01fb458d966574202dd53402dca。

用户在训练前明确选择batch96；24项resolved与实际raw配置全部为96，父子唯一差异仍为资格mode。
服务器实际每客户端GPU配额.1，所有对照一致；Linux Python3.11.16/Torch2.14.0+cu130/numpy2.4.6，双RTX5090。
本地与服务器准备源码差异已定位为batch96、GPU.1和旧版command包装；审计执行代码经AST/源码核对一致。
以本批batch96严格配对为准，旧batch48证据不能混入本批比较或声称原48合同已执行。

| Sign-flip | M2平均/最终ACC | H2平均/最终ACC | MK平均/最终ACC | 良性误标M2→H2 |
|---|---:|---:|---:|---:|
| seed201 | 80.1984/84.20 | 79.9716/84.52 | 79.9004/85.16 | 4/351→0/351 |
| seed204 | 80.1820/85.34 | 80.1820/85.34 | 79.0634/84.83 | 0/351→0/351 |

ACC单位%；攻击期平均为round11--60。seed201平均差-.2268pp，超过-.2pp门.0268pp；最终+.32pp通过。
攻击期每轮平均恶意权重5.3510%→5.6019%，增加.2509pp，超过.1pp门。
其余七个条件/seed配对无聚合轨迹/权重分叉；clean/Gaussian/LIE平均及最终ACC保持。
因此不晋升H2，不放宽旧门；它消除了开发seed误伤，值得保留并修复具体缺陷。
完整24项指标含权重、zero/anchor质量、方向代理、首次分叉及耗时在review.summaries/divergences；
不同客户端LOO轴的负向贡献只是诊断代理，不能当成总有害方向，targeted ASR在本批为不适用。

## 根因及唯一修复

round52六名攻击者占据谱参考，H2正确撤销四条良性原标记（cids6/11/12/19），良性各恢复.1权重。
但其记忆仍采用原始标记，因此把这四名良性主体记为上次拒绝。
round53，参考中的11/19/6被判不合格，合格4/7低于所需5/7，H2撤销攻击客户端0/10/16的正确拒绝，
给出约.09687/.09679/.1权重。首次M2-H2权重和聚合分叉为round52。
该链条有逐客户端日志支持；不能把全部ACC差完全归因于单轮权重，闭环仍须验证。

H2b只改变历史写入：资格门撤销的R1c判断保持该主体旧状态，当前R2明确拒绝则记为拒绝。
未弃权的原判断照常更新；正常参与可清除，缺席保留，未知且弃权仍未知。
同主体多个客户端时，明确拒绝优先；否则任一弃权保留旧值，全部正常才清除。成功聚合后同时提交。
不把未获确认的判断写成新的拒绝，也不通过弃权清除已有拒绝历史；不调整三分之二门、R1c/R2、累计或回填。
H1关闭。生产逻辑不使用攻击身份或类型。
旧M2/H2的16条轨迹按固定规则离线重放：保留round52四条误伤修复，并恢复H2轨迹round53三条攻击拒绝，
其余条件/seed不新增差异；不预测ACC、不声称闭环通过。入口replay_memory.py和h2b_offline.json。

## 保留登记

M2-H2已登记analysis/rtc_retained_candidates/M2-H2/candidate.json，状态retained_for_memory_repair；
独立快照原锁、协议、父子配置、review及302份审计输入（包括服务器源码），逐条哈希核验。
audit_bundle.zip共97097261字节，SHA256 5422938b0157db17f1fe66718ae3066891b7c565b78757d7211bcb8f5cd1e91f。
该大包仅本地保存，Git忽略，须另行备份；不含checkpoint/stdout，恢复到独立目录。
M2-G1原11项快照哈希再次通过，仍保留待修正并重新集成。
H2b不启用G1；其结论明确后必须返回低尾参考污染修复与LIE收益/安全回归，或提交明确替代证据。
batch96不能直接继承旧batch48的G1校准泛化或+5.1524pp收益结论，须重新验证。

## 下一批32项及预登记门

三组RTC：M2=rtc_i12_eligibility_observe；原H2=rtc_i12_eligibility_cap（未接受研究对照）；
H2b=rtc_i12_eligibility_confirmed。H2b相对H2唯一参数变化为reference_eligibility_memory=confirmed，其他配置相同。
原H2与M2仅eligibility mode不同，分别核验配置一致性。
seeds201开发/205新工程筛选，205注册前没有本地运行产物；每seed clean3、Sign4、Gaussian5、LIE.5四项，共32新单元、复用0。
201的全部trial-plan与本次batch96原结果一致，保留多数攻击压力轮。新源码/新合同下全部重做配对，不混用旧缓存。

参数：CIFAR10/ResNet18无预训练，IID20客户端每轮10人；60轮、本地5轮、batch96；
SGD lr.01、momentum.9、weight_decay.0001、cosine。mf.3，攻击11--60。
Sign scale1；Gaussian mean0/std.1；LIE z.5/all_updates；MK f3选5；Gaussian加RFA3步/smoothing1e-6。
原R1c/R2校准、cumulative power1、floor.5、MAD2.5、accepted回填.51。
Ray每client CPU1/GPU.1、object store3072MiB、最低可用内存10240MiB、等待120秒，自动重试0。
协议config/rtc_reference_memory_protocol.json；H2b入口在运行配置与CLI两处固定batch96，GPU固定.1。

每seed每条件均要求：候选联合良性误伤≤1%，且误伤条数不高于原H2；
相对M2和H2两者的active/final ACC差均≥-.002；各攻击期每轮平均恶意权重增量均≤.001。
seed201 Sign还要求相对M2误伤严格减少、相对H2撤销恶意标记数严格减少。
32项完整、全部质量/严格配对、原机制及历史状态重放通过；budget≤1e-8、几何重构相对≤1e-4、累计重放≤1e-9。
不通过则拒绝H2b，保留H2与G1研究记录；不改阈值、加可选seed或自动补跑。
首次多数攻击、on-off、非IID及targeted仍可能失败，本批不替代Random-v2、LIE.25、MinMax/MinSum、label reversal、DBA/scaling和最终验证。

19项纯合成检查通过（内存状态/原H2回归/真实聚合路径/600条合成日志/手动执行保护/CLI资源）；无客户端训练。
PowerShell语法、Bash -n/LF、Windows默认32项dry-run通过。缺结果时分析器按预期拒绝。
Windows冻结274份源码/60份产物，status/raw为0；Linux实际dry-run待返回。
锁84f3ce261ec6a2f771cde455a5a5e51cd8f32ba31aeb6eb115df2ef7894a833f；
源码zip ff19717b7479ba38bc232caeb9b0ffb26ada711585436fefc71a7df5cfbf5f03。

## 手动Git同步

按用户既有要求未自动提交或推送，本地HEAD73a1802。仅加入下列本回合文件，保留其他改动：

```powershell
Set-Location 'D:\workspace\FL2'
git add -- defenses/rtc/reference_eligibility.py defenses/rtc/v3.py experiments/rtc_v3/byzantine.py experiments/rtc_reference_memory_stage.py experiments/rtc_reference_memory_validation.py experiments/run_rtc_reference_memory.ps1 experiments/run_rtc_reference_memory.sh config/rtc_reference_memory_protocol.json tests/test_rtc_reference_memory.py tests/test_rtc_reference_memory_stage.py analysis/rtc_reference_eligibility_review analysis/rtc_retained_candidates/README.md analysis/rtc_retained_candidates/M2-H2 analysis/rtc_retained_candidates/M2-G1/followup_20260916.md
git add -f -- docs/RTC_H2_REVIEW_H2B_HANDOFF_20260917.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md
git diff --cached --stat
git diff --cached --check
git commit -m "Audit H2 batch96 and prepare confirmed-memory repair trial"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

不包含两个被忽略的大型候选证据包，Git同步不等于已备份它们。服务器更新前保留当地batch/GPU改动，不强制覆盖历史。

## 人工运行

Windows唯一启动命令（同入口默认dry-run已通过）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_memory.ps1' -Execute
```

输出D:\workspace\FL2\logs\rtc_reference_memory。完成后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_memory.ps1' -Analyze
```

Linux手动更新代码后，先进行无训练实机dry-run：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_memory.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

返回r1m_lock.json和实机输出后，再核验并交接Linux真实执行命令；Windows/Linux二选一，不重复两套32项。
Linux输出/home/jia_zhang/hqr/FL2/logs/rtc_reference_memory；实际训练完成后在原环境分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_memory.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

已完成单元经验证后跳过；failed/incomplete/orphan拒绝自动重跑，不支持中间轮checkpoint续训。
当前WAITING_FOR_MANUAL_EXPERIMENT。本回合progress：H2结果解除旧等待，完成验收/保留/根因诊断与H2b准备。
新人工边界计数1，正式blocked须连续三回合；未启动训练、不后台等待、不推进依赖H2b的新机制，完整目标未完成。
