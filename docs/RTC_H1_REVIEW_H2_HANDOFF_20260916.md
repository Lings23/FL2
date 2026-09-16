# H1闭环拒绝与H2参考主体资格人工交接

后续执行修正：Linux原准备在client参数断言处失败。H2已显式固定预登记训练参数并增加差异说明，
改用独立输出logs/rtc_reference_eligibility_configfix重新dry-run；最新命令见docs/RTC_H2_DRY_RUN_CONFIG_FIX.md。
下方原Windows锁及默认目录命令只保留为历史交接；不得用修改后源码直接执行旧冻结目录。

日期：2026-09-16。H1质量通过、候选拒绝；H2尚未运行。当前WAITING_FOR_MANUAL_EXPERIMENT。

## H1返回产物的独立核验

logs/rtc_reference_guard现为Linux服务器产物，不再是旧Windows准备目录。
逐manifest run_id读取status/<run_id>.json和rounds/<run_id>.csv：24/24 completed、exit0、last_round60，
每项61个唯一轮号、600条客户端记录，92/92 runner质量门通过。
此前以文件名包含status/rounds进行计数不适用于此结构，本次已按合同路径逐项核验。
242份归档源码、51份冻结产物核验；严格配对、原cap/累计/预算、历史点积/hash链全部通过。
独立重建14400条谱参考，最大cosine误差3.3611e-12；本地31项候选门与服务器一致，23通过、8失败。
派生浮点重算比较容差1e-12，布尔及结构精确比较；不放宽预注册质量门。
审计入口analysis/rtc_reference_guard_review/verify.py；review.json记录逐运行指标、误伤和来源哈希。
锁130303998e35f04d8f4eb41439e750dd14140b9f4b85c1936a900eb0b7cb7d51。
源码zip 9a8c45ca708af20bced412c595e34f84c41c35518808acc0ceef58356f8d811c。

| Sign-flip | M2平均/最终ACC | H1平均/最终ACC | MK平均/最终ACC | M2→H1恶意权重 |
|---|---:|---:|---:|---:|
| seed201 | 83.7182/87.41 | 82.4322/86.87 | 83.3378/87.47 | 4.6326%→9.2095% |
| seed203 | 84.0108/87.26 | 83.2910/86.30 | 82.5388/86.84 | 2.0881%→7.5629% |

ACC为百分数，攻击期round11--60；恶意权重为攻击期每轮平均聚合权重。
seed201误伤仍4/351，seed203仍0/363。两个seed的平均/最终ACC和恶意权重门失败，
seed201还违反误伤≤1%及严格减少门。clean/Gaussian/LIE的父子ACC和聚合轨迹保持一致，相关门通过。

H1从round31/32开始撤销正确方向拒绝。各自闭环内撤销39/43条恶意标记，
最终方向攻击标记从父版115/121降至86/85；这些计数来自不同轨迹，不能逐条等同相减。
seed201 round52的受污染参考在H1轨迹中历史cosine已变成约+.0098，因此继续误伤4名良性客户端。
旧轨迹上的“良性4→1”没有转化为闭环改善。故拒绝H1，不把历史聚合同向作为默认门，也不调阈值求通过。
历史I12接受范围保留，不能外推M2已通过seed201安全转移。

## H2固定机制与开发证据

父版M2+只读资格观测rtc_i12_eligibility_observe；候选rtc_i12_eligibility_cap。
唯一差异reference_eligibility_mode observe→cap；H1关闭，G1暂未启用。
原R1c要执行cap时，至少ceil(2*参考主体数/3)个参考主体须在其上一次参与时未被原R1c或R2标记。
这沿用既有三分之二确认比例，不搜索比例、次数、遗忘窗口或攻击专用阈值。
未知历史视为有资格，避免首次攻击时全面撤销原R1c。每主体保存最近一次参与的原R1c OR R2标记，
当前轮所有决定只读上一状态，成功聚合后统一更新；缺席保留，正常参与清除。
状态使用原判断，避免资格门撤销判断后自我清除证据。多客户端同主体只计一票，同主体任一原标记保留下一状态。
仅撤销不可靠参考导致的R1c cap；不重算谱参考、不翻转模型、不隔离主体、不撤销R2/累计/其他预算。
运行时不读取攻击类型或恶意身份。

动机：M2 seed201 round52六个参考中五个在上次参与已被原模块标记。
固定规则的旧轨迹离线重放将误伤4→1，两个seed的攻击标记115/121全部保留；其他三条件没有原方向标记可撤销。
规则与开发证据在analysis/rtc_reference_guard_review/h2_design.json、replay_h2.py和h2_offline.json。
这不是闭环ACC收益证明；原误标可污染历史资格，on-off攻击可清除状态，首次多数攻击与非IID仍未解决。
不能根据新结果再放宽门。若H2失败，保留M2及G1记录，重新依据证据诊断，不自动搜索。

## 冻结24项、参数与门

24新单元、复用0：seeds201/204，每seed clean父/候选2、Sign父/候选/MK3、Gaussian父/候选/MK/RFA4、LIE .5父/候选/MK3。
201是已知失败开发复核，四条件trial-plan与H1中seed201完全相同，不删多数攻击压力轮。
204注册前未发现既有本地运行产物，作为新的工程筛选，不宣称统计显著性或最终泛化。

CIFAR10/ResNet18无预训练，IID20客户端、每轮10人，60轮、本地5轮、batch48；
SGD lr .01、momentum .9、weight decay .0001、cosine。mf .3，攻击11--60。
Sign scale1；Gaussian mean0/std.1；LIE z.5/all_updates；MK f3选5，RFA3步/smoothing1e-6。
原R1c/R2校准保持；cumulative power1、floor.5、MAD k2.5、accepted回填.51。
Ray每client CPU1/GPU.25，object store3072MiB，最低可用内存10240MiB，等待120秒，自动重试0。

每seed、每条件独立验收：候选联合良性误标≤1%；seed201 Sign误伤条数严格少于父版；
所有条件active/final ACC候选减父版≥-.002；各攻击的攻击期每轮平均恶意权重增量≤.001。
不以跨seed均值掩盖失败。父版已知误伤失败照实报告，不改写旧批次。
24项全部完整、质量门、严格配对、有限性、R1c/R2与资格状态独立重放均通过；
budget≤1e-8、几何重构相对误差≤1e-4、累计重放≤1e-9，资格投票/状态/flag/q精确匹配。
协议config/rtc_reference_eligibility_protocol.json。DBA/scaling、Random-v2、LIE .25、MinMax/MinSum、label reversal、新seed/非IID最终验收仍待完成。

M2-G1继续retained_for_repair_and_reintegration；原配置及11项快照哈希再次核验一致。
H2只修复父版本，不代表已保留G1的LIE +5.1524pp组合收益。
H2结论明确后必须回到G1低尾参考污染修复与组合验证，或提供明确替代证据；不能直接跳过至R4/R5。

## 已完成准备

13项纯合成测试通过，覆盖H2历史状态/未知/恢复/重复主体、完整聚合只读不变性和预算、
600条合成日志全验证及伪造状态拒收、人工执行边界，并回归原H1测试。没有真实客户端训练。
Windows默认24项dry-run、PowerShell语法、Bash -n/LF和Python编译检查通过；Linux实际dry-run尚待返回。
无结果时分析器按预期因缺status拒绝，未生成接受结论。
Windows冻结267份源码/51份产物，status/rounds/raw均0。
锁4dd7b1ca353493feb633323e803dfe1dde19a26160060a98c3cb4c1914f22c1a；
源码zip f8a4accc5b154b7d662eda810b46c47d21e3f2f5408cffdaadc899d36e06648d。

## 手动Git同步

依用户指示留待手动提交/推送，本轮未自动同步。当前HEAD ba54ee2，分支codex/periodic-attack-defenses。
只暂存本次改动，提交前检查暂存区没有其他工作；旧冻结日志、checkpoint及G1约117MB证据包不随代码上传。

```powershell
Set-Location 'D:\workspace\FL2'
git add -- defenses/rtc/v3.py defenses/rtc/reference_eligibility.py experiments/rtc_v3/byzantine.py experiments/rtc_reference_eligibility_stage.py experiments/rtc_reference_eligibility_validation.py experiments/run_rtc_reference_eligibility.ps1 experiments/run_rtc_reference_eligibility.sh config/rtc_reference_eligibility_protocol.json tests/test_rtc_reference_eligibility.py tests/test_rtc_reference_eligibility_stage.py analysis/rtc_reference_guard_review analysis/rtc_retained_candidates/README.md analysis/rtc_retained_candidates/M2-G1/followup_20260916.md
git add -f -- docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md docs/RTC_H1_REVIEW_H2_HANDOFF_20260916.md
git diff --cached --stat
git diff --cached --check
git commit -m "Reject historical half-space guard and prepare H2 reference eligibility trial"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

## 人工实验入口

Windows唯一启动命令，已通过同入口默认dry-run：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_eligibility.ps1' -Execute
```

输出D:\workspace\FL2\logs\rtc_reference_eligibility；完成后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_eligibility.ps1' -Analyze
```

Linux先手动拉取本次提交，使用已由H1锁核实的服务器路径，仅执行无训练dry-run：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_eligibility.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

返回实机输出和r1e_lock.json后再交接Linux真实启动命令，不能把本地Windows准备说成Linux已验证。
Windows/Linux是替代入口，不需各跑24项。Linux预期输出/home/jia_zhang/hqr/FL2/logs/rtc_reference_eligibility。
实际训练完成后，在原Linux环境分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_eligibility.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

已完成单元经核验后跳过；失败、不完整或孤立日志拒绝自动重跑，不支持中间轮checkpoint续训。
本回合为progress：新H1终态解除旧阻塞，完成验收/诊断和H2准备。新人工边界计数1；正式blocked须连续三回合。
保持WAITING_FOR_MANUAL_EXPERIMENT；不训练、不后台等待、不推进依赖H2结果的候选。完整目标尚未完成。
