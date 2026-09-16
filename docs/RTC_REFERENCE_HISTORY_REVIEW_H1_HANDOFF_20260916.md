# 参考历史观测验收与M2-H1人工实验交接

日期：2026-09-16。五项观测质量接受；H1仅为新候选，尚未接受。M2-G1保留待修正并重新集成。
当前阶段WAITING_FOR_MANUAL_EXPERIMENT，所有真实训练仍由用户手动执行。

## 返回产物与验收

logs/rtc_reference_history_observation现为外部Linux服务器产物：5/5 completed、exit0、last_round60，
各61个唯一轮号、600条客户端记录，21/21 runner质量门通过。234份源码与14份计划产物完整匹配。
原锁SHA256：909fb289c50a5db76f642c432f12c070f0a97baa76ddb4040abd5ab8b58c456e。
原源码zip SHA256：1097accb5c93dd739efcf0d397a6165e85c99ac5ecd9c631d177714811edb241。
严格配对、全部原预算/累计/重构/历史点积与hash链检查通过；独立重建3000条谱参考，最大cosine差1.8389e-15。
父版与观测版的ACC/loss、聚合sketch、客户端权重、R1/R2 cap和累计q逐项相同，说明观测不改变行为。
服务器与本地判定一致；派生浮点重算采用1e-12容差，布尔/类别/结构保持精确比较，不放宽实验质量门。
审计analysis/rtc_reference_history_review/verify.py及review.json。原始日志、锁、服务器observation不修改。

| seed201 条件 | M2平均/最终ACC | 观测版平均/最终ACC | MK平均/最终ACC |
|---|---:|---:|---:|
| clean，round1--60 | 80.8757 / 89.04 | 完全相同 | 未运行 |
| Sign-flip，round11--60 | 83.7182 / 87.41 | 完全相同 | 83.3378 / 87.47 |

## 历史方向证据与限制

Sign-flip原误伤仍为4/351。round52的3个误伤客户端所用谱参考全由6名攻击者构成；
其参考与上一轮实际trainable更新的cosine分别为-.0277951、-.0277461、-.0277657。
round60另一误伤的参考为良性，历史cosine=.0368480；本次假设不能消除它。

如果只允许历史cosine>0时执行原方向cap，在已观测的旧轨迹上，良性标记会从4降至1，
但攻击标记也会从115降至79，撤去36条原攻击标记。因此不能单凭误伤下降宣称鲁棒性提高。
clean有589/600条历史参考可用，其中172条cosine非正；但原R1c标记为0。
历史参考并非总是正向，更不是可信oracle；自然优化振荡和攻击污染都可能使其失效。
全量分布、缺失和标记记录已在review中保留，没有只挑round52。

这支持检验一个有限问题：把历史方向作为原R1c拒绝判断的第二项确认，是否可降低参考翻转误伤，
同时保持原有攻击防御效用与权重抑制。不能据此改写旧批次失败或声称解决多数攻击。

## 唯一新机制H1

父版本rtc_i12_guard_observe=M2+只读历史确认日志；候选rtc_i12_guard_cap=M2-H1。
唯一配置差异reference_guard_mode: observe→cap。
原R1c已标记危险时，只有参考有效、上一轮实际trainable聚合位移非零、两者cosine严格>0才保留方向cap。
历史不存在、方向无定义、cosine≤0时仅对这项R1c判断弃权，不翻转/替换客户端更新，不撤销R2或累计预算。
阈值0表示固定同半空间确认，不从攻击数值选取最有利分割点，不做阈值搜索。
状态只保存上一轮实际聚合位移，包含accepted-anchor贡献；在本轮成功聚合后提交，当前轮不能进入自身历史。
生产代码不读取恶意标签或攻击类型；标签仅用于分析。

M2的R2、cumulative power1、floor .5、MAD k2.5、accepted回填.51不变。
M2-G1原24项失败和完整归档不变。本批修复父版本，不启用尚未接受的G1；后续必须返回G1参考污染修复和组合验证，
或者提供明确替代证据，不能因为H1阶段推进就丢弃G1的LIE平均+5.1524pp收益。

## 冻结实验与验收条件

24个新单元、缓存复用0：seeds201/203，每seed clean父/候选2项、Sign-flip父/候选/MK3项、
Gaussian父/候选/MK/RFA4项、LIE .5父/候选/MK3项。
201是已知失败的开发复核；203在注册前没有本地实验产物，用作独立工程筛选，不是统计显著性验证。
seed201的clean/Sign-flip trial-plan与已完成实验一致，保留6攻击者轮次，不修改随机压力协议。

训练：CIFAR10/ResNet18无预训练、IID20客户端每轮10人、60轮、本地5轮、batch48，
SGD lr .01、momentum .9、weight decay .0001、cosine；mf .3、攻击11--60。
Sign-flip scale1；Gaussian mean0/std.1；LIE z .5/all_updates；MK f3选5，RFA3步/smoothing1e-6。
Ray每client CPU1/GPU.25，object store3072MiB、最低可用内存10240MiB、内存等待120秒、自动重试0。

每个seed、每个条件均须通过，不使用均值掩盖单seed退化：

- 候选联合良性误标率≤1%。已知父版seed201失败仍报告，不要求一个待修复的父版本先通过才能检验修复。
- seed201 Sign-flip良性误标条数严格少于配对父版。
- clean/Sign-flip/Gaussian/LIE平均和最终ACC差均≥-.002，即最多下降0.2个百分点。
- 每个攻击的候选每轮恶意权重≤父版+.001，即最多增加0.1个百分点。
- 全部计划完成、全部质量门、严格配对、有限性、原机制与新guard独立重放、跨轮向量hash链通过。
- budget≤1e-8、几何重构相对误差≤1e-4、累计重放≤1e-9、历史夹角重算≤1e-10、扩展Gram半正定相对容差1e-8。

不能为了让H1通过而放宽恶意权重门。若历史确认撤去过多正确攻击标记，导致权重或效用回归失败，拒绝H1。
这份新协议用于修复已知缺陷，不改变原低尾24项对父/候选均要求≤1%的历史判定。
当前没有targeted安全或非IID证据；其验证仍是后续任务，不能以本批通过替代。

## 验证与冻结产物

19项不同的纯合成检查通过：完整RTC聚合路径、只读不变性、正/负/零/缺失历史、预算、
600条合成客户端的完整验证器、伪造guard/点积/hash拒收、回归门及失败运行禁止启动；未加载数据训练。
PowerShell语法与24项默认dry-run通过；Bash -n和LF检查通过，Linux实际dry-run尚待返回。
无结果时运行分析器按预期拒绝，未生成候选通过结论。
Windows输出logs/rtc_reference_guard，259份源码、51份冻结产物、24格、status/raw均0。
锁SHA256：45b9faca8f973702e522d49707aa53c9c49573a295053a59864b832d949f6af2。
源码zip SHA256：957119e049b47c1f7de338468936b07067ad1b9f193709f85ae36376b2a670c5。
协议config/rtc_reference_guard_protocol.json；设计analysis/rtc_reference_history_review/design.json。

## 人工Git同步

遵循用户“手动提交”指示，助手没有提交或推送；本地HEAD为7b27517。
以下包含新代码、分析证据、文档和M2-G1登记；其约117MB audit_bundle.zip已被独立.gitignore排除，须另行备份。
其他未提交改动、数据集、checkpoint和大体积日志不包含在本清单中。

```powershell
Set-Location 'D:\workspace\FL2'
git add -- defenses/rtc/v3.py defenses/rtc/reference_guard.py experiments/rtc_v3/byzantine.py experiments/rtc_reference_guard_stage.py experiments/rtc_reference_guard_validation.py experiments/run_rtc_reference_guard.ps1 experiments/run_rtc_reference_guard.sh config/rtc_reference_guard_protocol.json tests/test_rtc_reference_guard.py tests/test_rtc_reference_guard_stage.py analysis/rtc_reference_history_review analysis/rtc_retained_candidates
git add -f -- docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md docs/RTC_REFERENCE_HISTORY_REVIEW_H1_HANDOFF_20260916.md
git diff --cached --stat
git diff --cached --check
git commit -m "Audit reference history and prepare paired H1 corroboration trial"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

## 人工实验命令

Windows唯一启动命令，已完成同入口默认dry-run：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_guard.ps1' -Execute
```

完成后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_guard.ps1' -Analyze
```

Linux先手动拉取上述代码提交，随后仅执行dry-run：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_guard.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

返回r1g_lock.json和实机验证输出后再交接Linux真实执行命令。Windows/Linux是同批次替代入口，不重复两套24项。
实际训练后在原Linux环境分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_guard.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

Linux预期输出/home/jia_zhang/hqr/FL2/logs/rtc_reference_guard；保留原锁和r1g_sources.zip。
已完成单元经runner核验后跳过；failed/incomplete/orphan拒绝自动重跑，需要人工检查。
不支持中间轮checkpoint恢复，不擅自停止或重启用户实验。

本回合为progress：旧五项等待解除，完成观测验收与单机制候选交接。新人工边界计数1，
正式blocked须满足宿主连续三回合规则。未训练、不后台等待、不继续推进依赖H1结果的候选；总体目标未完成。
