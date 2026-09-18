# G2 clean校准验收与G1/G2集成人工交接

日期2026-09-18。4项clean校准已完成且验收；G2算法尚未接受。下一批32项已准备，WAITING_FOR_MANUAL_EXPERIMENT。

## 本次真实验收

logs/rtc_g2_clean_calibration已被用户返回的Linux结果替换，旧Windows等待记录过期。
四项completed/exit0/last_round60，各61轮/600客户端；18项runner门全部通过。
264份源码、14份冻结产物哈希核验；客户端配对、原R1c/R2/confirmed历史、累计、预算、几何重构全部通过。
独立重建2400条谱参考最大cosine误差2.2325891e-15；原低尾观测中位数/比值/计数/q=1独立重放通过。
M3与只读观测逐轮ACC、loss、客户端权重、聚合sketch、拟合/防御随机流一致。
clean106平均/最终ACC为77.4788333%/87.22%，clean107为77.5926667%/86.78%，各自两组一致。

阈值只来自clean106：600条有效比值的最小值0.8996827459271943；严格ratio<threshold，连续两次参与。
独立直接范数计算的最小值为0.8996827459271942，与生产主体加权均值计算仅相差浮点舍入，复核容差1e-12。
clean107的600条均有效，最小比值0.9002906313990927；低尾与M3联合标记均0/600，三项留出门通过。
原服务器decision与独立重算一致；quality_accepted/calibration_accepted=true，candidate_accepted=false保持。
未用攻击结果选择阈值，没有修改107留出结果或旧G1阈值。

服务器锁9b6cff62135446f7d4a48c8876d6bfa71e83bd08a957fada3e63a85c452f6f41；
源码zip4c165b03de34eb29469e9704bc812610586a740fd029db316bcd0c84a6cf7037。
审计analysis/rtc_g2_clean_review/verify.py、review.json；新校准config/rtc_g2_lower_tail_calibration.json，保存来源哈希。
实际batch96/GPU.125符合本批协议；Linux Python3.11.16/Torch2.14.0+cu130/NumPy2.4.6/CUDA13/双RTX5090。
共享bridge与本地仅有已知准备默认batch/GPU差异，分析代码逐字核对相同；冻结源码和服务器原产物均保留不改。
实际仓库/home/jia_zhang/hqr/FL2及fl2环境优先于旧REMOTE_L20.md中的/root/FL2描述。

## 版本关系与唯一变化

M2=B0+R1c+R2；M3=M2+H2b资格门及confirmed历史写入。M3保持已接受范围，不因校准通过晋升M4。
本次实验批次名/输出目录为rtc_g2_integration，三个RTC运行组为：

| 研究版本 | 运行别名 | 低尾模式 | 参考策略 |
|---|---|---|---|
| M3＋只读G2观测（直接父对照） | rtc_i12_g2_observe | observe | raw_eligible |
| M3＋未修正G1，batch96重新校准控制组（G1-R） | rtc_i12_g1_recalibrated | cap | all |
| M3＋修正G2（唯一候选） | rtc_i12_g2_cap | cap | raw_eligible |

G2对直接父对照仅observe→cap；G2对G1-R仅参考策略all→raw_eligible。完整resolved配置逐项断言只允许上述差异。
三组均继承M3，不是从B0重启。G1-R和G2共用本次clean106/107阈值；两份clean日志没有R2拒绝，参考总体相同。
G1-R是原低尾机制在新父版本/batch96下的重新校准对照，不是旧batch48/G1原阈值的复现实验。
旧M2-G1的+5.1524pp及拒绝门、源码、配置、阈值原样保留，不能把本批控制组改称旧结果。

G2排除当前R2拒绝主体，保留至少9个其他正名义质量主体、两次参与、无效/正常清零、缺席保留、q0。
同主体任一客户端明确R2拒绝即排除整个主体；输入不读取攻击类型、身份标签或测试表现。
raw_eligible要求R2 cap开启，非法/无效配置拒绝。成功聚合后才提交低尾历史，原H2b历史规则不改。
10人协议任一R2拒绝会使参考不足，低尾因此弃权；原R2 cap仍生效，不撤销其他预算/惩罚。
该机制不修改坐标中位数anchor，R2漏检、恶意多数或非IID仍可能造成风险，不能预先宣称安全。
R1c/R2阈值、三分之二资格门、MAD2.5、floor.5、累计power1及accepted回填.51均不改。

## 固定32项矩阵与验收

seeds201（开发失败/收益复核）、206（注册前未发现本地运行文件的新工程seed）；复用0，新训练32项。
每seed clean三RTC组3项；Sign三RTC+MK4项；Gaussian三RTC+MK+RFA5项；LIE .5三RTC+MK4项。
所有父子/控制/挑战者在同一宿主及新合同下运行；不复用旧平台或旧batch基线。
CIFAR10/ResNet18无预训练、IID20客户端每轮10人、60轮、本地5epoch、batch96；
SGD lr.01/momentum.9/weight_decay.0001/cosine，strict/principal_uniform/确定性客户端训练。
mf.3、攻击11--60；Sign scale1；Gaussian mean0/std.1；LIE z.5/all_updates；MK f3选5；RFA3步/smoothing1e-6。
Ray CPU1/GPU.125每client、objectstore3072MiB、最低可用内存10240MiB、等待120秒、自动重试0。

所有seed分别验收，不能平均掩盖失败：
- 全32项完成及全部runner门，61轮/600客户端、来源/实际配置/数据初始化/抽样/随机流配对、有限性。
- 原M3机制重放；独立低尾参考、计数、资格、streak/q/applied重放；budget≤1e-8、几何重构相对≤1e-4、累计重放≤1e-9。
- G2联合良性误标≤1%，且条数不高于G1-R。
- LIE相对M3：攻击期平均ACC至少+1pp、最终ACC不下降、每轮平均恶意权重≤M3的80%。
- clean/Sign/Gaussian相对M3：平均/最终ACC差≥-.2pp；攻击期每轮平均恶意权重增加≤.1pp。
- 所有条件相对G1-R：平均/最终ACC差≥-.2pp；各攻击恶意权重增加≤.1pp，防止修复时抹掉G1效用。

协议config/rtc_g2_integration_protocol.json；clean窗口1--60，攻击11--60；全部条件/seed/gate须通过。
不追加可选seed、不事后调阈值、不自动补跑。失败保持M3并保留G1/G2证据，分析exit0不代表候选通过。
MK/RFA同时报告；本批是工程集成验收，不声称统计显著性/全面优于对手。
LIE .25、MinMax/MinSum、Random-v2、label reversal、DBA/scaling、最终新seeds/非IID及R4/R5保持待办。

## 验证与冻结

20项纯合成检查通过：参考排除/无效重置、实际聚合cap及预算、M3只读观测不变性、原G1/H2b回归、
600行原机制与新观察联合校验、伪造日志拒收、整数2.0兼容且拒绝小数/非有限值、每seed验收门和手动执行保护。
测试初次有一个合成输入未满足R2九个正范数peer条件，修正输入后全部通过；未改防御阈值来迁就测试。
PowerShell语法、Bash -n/LF、Windows默认32项dry-run通过，缺训练结果时分析器正确拒收。
本地冻结288份源码/60份产物，status/raw为0；未执行真实训练。
锁85fd8a5b25343d65b97740bed558a550f962dfc578072d106bd42d42d396e2e5；
源码zipbe85aa72a78cb502190dc5ea7fff916f8e8e7216e3ab94eaccbf5ac1c28b2ae6。
Linux脚本语法已验证，实机dry-run仍须用户返回；Windows验证不冒称Linux验证。
M2-G1及M2-H2共17项归档哈希再次一致；两个大型审计包仍仅本地，Git不包含这些包。

## 手动Git

用户既有手动提交要求继续有效，本回合未自动commit/push。当前HEAD11e5858。

```powershell
Set-Location 'D:\workspace\FL2'
git add -- defenses/rtc/v3.py experiments/rtc_v3/byzantine.py experiments/rtc_g2_integration.py experiments/rtc_g2_validation.py experiments/run_rtc_g2_integration.ps1 experiments/run_rtc_g2_integration.sh config/rtc_g2_lower_tail_calibration.json config/rtc_g2_integration_protocol.json tests/test_rtc_g2_integration.py analysis/rtc_g2_clean_review analysis/rtc_retained_candidates/README.md analysis/rtc_retained_candidates/M2-G1/candidate.json analysis/rtc_retained_candidates/M2-G1/followup_20260916.md
git add -f -- docs/RTC_G2_CLEAN_REVIEW_INTEGRATION_HANDOFF_20260918.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md
git diff --cached --stat
git diff --cached --check
git commit -m "Validate clean G2 calibration and prepare paired G1 G2 integration"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

检查暂存内容，保留其他改动；不强推或覆盖服务器环境改动。Git命令不授权自动训练。

## 人工执行

Windows唯一启动命令（已完成同入口dry-run）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_g2_integration.ps1' -Execute
```

完成后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_g2_integration.ps1' -Analyze
```

Linux更新代码后先执行无训练实机dry-run，返回锁与终端结果再交接训练命令：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_g2_integration.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

两平台二选一，不重复两套32项；输出logs/rtc_g2_integration。Linux训练结束后在原宿主分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_g2_integration.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

完整单元经验证可跳过；失败/不完整/孤立raw拒绝自动重跑，不支持中间轮checkpoint续训。
状态WAITING_FOR_MANUAL_EXPERIMENT；本回合progress，新阻塞计数1，宿主规定连续三回合后才正式blocked。
校准返回解除旧等待，当前仅交接新批次；不自动训练、不后台等待、不推进依赖G2验收的下一阶段。完整目标未完成。
