# M3 全攻击 Multi-Krum 补充对照

2026-09-23。用户最新要求优先补一组全量攻击 Multi-Krum，再评估 M3 的发布定位；覆盖此前“仅跑 RTC”的对照范围限制。
本批只补比较证据，不修改 M3，不启用 G2，不晋升 M4，不重新训练已有 M3 或其他防御。

## 能否推出 M3

可以冻结并展示 M3 研究版本，配合完整且诚实的比较表。M3 已是 M2+H2b 的限定范围主线，并非等本批通过才首次命名。
这批回答的是同实现、同输入下的经验比较。单 seed42 不能完成最终多 seed/非IID/消融验收。
已知 Min-Sum 良性 cap 4/344、全标签反转12/344超过1%，定向翻转峰值ASR83.8%；即使 Multi-Krum 更差，这些问题也没有被修复。
因此不预设全面胜出、所有攻击安全或生产部署接受，不因“跑完”自动扩大 M3 接受范围。

## 冻结内容

- 新训练 **12 单元**：clean、Gaussian、Random-v2、Sign、LIE .25/.5、Min-Max、Min-Sum、targeted label flip、all-reverse、Scaling、DBA。
- 唯一新防御 `rtc_i12_multikrum`，实际 Multi-Krum `f=3, m=5`；R1/R2 仅观测、不施加 cap。
- 复用 M3 的 **12 项完整结果**，原目录 `logs/rtc_m3_all_attacks_seed42` 不修改。
- seed42、CIFAR10/ResNet18、IID、20客户端每轮10、60轮、攻击11—60、恶意比例.3。
- batch96、本地5轮、SGD lr.01/momentum.9/decay.0001、cosine；每客户端CPU1/GPU.125、object store3072MiB、最低可用内存10240MiB、等待120秒、自动重试0。
- 所有攻击参数逐字继承 M3 冻结协议，Random-v2 Rademacher scale10 仍标探索性，未做 FedAvg 强度筛选。
- 父锁 SHA256 `350ed123c1b58df74efb2c8947bfce3beefd99cf98170971d57f5dfdd23af190`。
- 父源码包 SHA256 `f5f8ae948e79cf1cacea7c334a75c8068eb3dbfdf399fc7963880e79f4bd8435`。

入口在当前仓库，训练进程在新输出的 `frozen_source` 中执行；该目录的276份源码逐字来自已验收 M3 包，不能手动修改。
控制器与协议单独冻结。准备阶段对已有 M3 的12项完整日志重新核验，比较新解析配置，除了防御配置和路径外要求完全一致。
Linux的攻击implementation/contract/TrialPlan必须与父锁完全一致。结果阶段再核验数据、初始模型、恶意身份和每轮客户端/fit随机流；不在结果出现前宣称这些运行时配对已通过。

Windows旧哈希使用反斜杠、Linux使用正斜杠。Windows预检只生成显式派生的本机freeze，276份归档源码字节不变；父Linux哈希仍独立重建核验。
Windows锁标记 `training_host_compatible=false`，不允许将这些预检产物作为Linux训练合同或在Windows复用Linux M3训练结果。
训练必须在原双5090 Linux环境独立prepare：Python3.11.16、Torch2.14.0+cu130、NumPy2.4.6、CUDA13.0，以及父记录的Python路径和两块GPU。
环境变更直接拒绝，不能为了运行而绕过配对校验。

## 质量与比较判据

所有12项必须completed/exit0/last_round60、61唯一轮次、600唯一客户端记录、全部runner门及重算攻击执行门通过。
重新验证R1/R2观测、M3资格记忆/累计/预算与几何重构，MK每轮恰好5个.2权重；独立重算混淆矩阵ACC/标签ASR。
后门ASR没有逐图预测，只能核对9000分母、有效标志和整数成功数，不冒称完整独立重算。
整数CSV字段允许2.0/3.0，但拒绝小数与非有限值；不修改原CSV。

逐条件报告平均/最终/末10轮ACC，定向攻击报告平均/峰值/末10轮/最终ASR，另保留权重、投影、运行时间和超过f3的轮数。
预先标注检查：M3平均/最终ACC低于MK超过.2pp、任一ASR高于MK、M3良性新cap超过1%都单独列出；不以宏平均收益抵消。
这些是已见M3结果、未见新MK结果时登记的比较检查，不能改写成盲测预注册或追认旧M3全范围验收。
MK丢弃良性更新与RTC新cap误伤不是同一检测定义；MK只读观测标记不作为它的误报率。
`broad_release_accepted`、`final_goal_achieved`、统计显著优越性在本批不会自动置true。

## 本机验证与手动入口

19项合成/入口检查通过；PowerShell语法、Bash `-n`通过；Windows最终dry-run记录见状态文件。
Bash语法检查未执行脚本，更未训练。首次Windows预检暴露路径分隔符哈希问题，失败目录独立保留；后续修复不修改M3源锁或归档。
Linux实际dry-run尚未执行。按照目标提示词“尚未完成Linux dry-run时仅交接Linux验证命令”，本次先交接下面的不训练命令。

本机PowerShell只做预检：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_m3_multikrum.ps1'
```

服务器先手动同步本次代码，保留完整原M3日志及其中 `m3_sources.zip`，再执行唯一Linux验证命令：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_m3_multikrum.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data --output /home/jia_zhang/hqr/FL2/logs/rtc_m3_multikrum_seed42
```

这条命令不会训练。成功时应返回12新单元、12复用单元、`training_host_compatible=true`。
请返回该输出及新 `multikrum_lock.json`；核验后再交接真实训练命令。当前不提供未完成Linux实机验证的训练启动指令。
服务器输出必须是独立新目录，不能将本机Windows预检目录复制过去替代prepare。
若原服务器环境已变化，保留现状并报告差异；本批不会自动重跑M3来凑配对。

真正完成12项训练后，在原服务器分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_m3_multikrum.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --output /home/jia_zhang/hqr/FL2/logs/rtc_m3_multikrum_seed42 --analyze
```

分析生成 `analysis/decision.json`、`comparison.csv`、`comparison.md`。本批要求在原服务器分析；拷回本机后可读取派生表，跨平台导入审计另按原锁映射，不修改原始路径或哈希。
中断后，完整且核验通过的单元可跳过；失败/未完整/孤立产物会在任何剩余训练开始前拒绝。没有轮级checkpoint续训，不自动补跑。

## 手动Git

遵循用户后续“手动提交”偏好，未自动提交或推送。仅本次新增5项实现/配置/测试和相关文档：

```powershell
Set-Location 'D:\workspace\FL2'
git add -- experiments/rtc_m3_multikrum.py experiments/run_rtc_m3_multikrum.ps1 experiments/run_rtc_m3_multikrum.sh config/rtc_m3_multikrum_protocol.json tests/test_rtc_m3_multikrum.py
git add -f -- docs/RTC_M3_MULTIKRUM_HANDOFF_20260923.md docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare source-pinned Multi-Krum supplement for M3 all-attack comparison"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

这些命令不上传原始大日志/数据/源码包；服务器必须已有完整父批次，prepare会逐项校验。
其余未提交工作区改动不混入本批，旧冻结协议与失败结论不改写。

## 等待与后续范围

阶段 `WAITING_FOR_MANUAL_EXPERIMENT`，当前外部依赖是原Linux环境的手动dry-run回执，然后才是手动12项训练。
本回合是progress，新人工边界计数1，尚未满足宿主连续三个目标回合的正式blocked条件。
不启动任何训练，不后台等待，不推进依赖该比较的新候选。
M2-G1与M3-G2继续保留待修复，M2-H2原失败及M3后继关系保持；补MK不消除R1误伤、targeted峰值、低尾修复、独立seeds/非IID和消融义务。
