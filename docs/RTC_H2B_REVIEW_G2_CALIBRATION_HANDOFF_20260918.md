# H2b验收与G2 clean校准人工交接

日期：2026-09-18。H2b通过本批工程验收，登记M3；总体目标未完成。
当前阶段WAITING_FOR_MANUAL_EXPERIMENT：仅准备4项clean校准/观测验证，未运行训练。

## H2b真实验收

服务器结果目录logs/rtc_reference_memory；32/32 completed、exit0、last_round60，每项61轮、600条客户端记录。
98项runner门、62项候选门全部通过；257份服务器源码及60份冻结产物哈希核验。
独立重建19200条谱参考，最大cosine差1.7439114e-11；机制、confirmed历史、预算、累计及配对重放通过。
服务器decision与本地重算一致，原日志和判定未改；quorum的2.0/3.0直接由float等值校验通过，无需日志转换。
审计analysis/rtc_reference_memory_review/verify.py、review.json；接受记录config/rtc_reference_memory_accepted.json。

服务器锁4a6c14a582d21e0cd7c85adac065d32583925ff7724c971d3355cb68f9071a8f；
源码zip cf5846542529422bcb1ec0668c2cea49eac1c53ff34c890cddd61e30e42a3814。
实际Linux Python3.11.16、Torch2.14.0+cu130、NumPy2.4.6、CUDA13、双RTX5090。
所有32项resolved和raw配置均batch96、每客户端GPU配额.125。
资源偏差：原协议登记.1，服务器在准备命令中改为.125；其他训练配方和门槛不变。
本地/服务器阶段源码仅此一处变化；共享bridge另有batch48→96和GPU.25→.125的已核实准备差异。
资源偏差原样登记，不追改旧协议；结论限定实际.125同宿主配对，未混合旧.1或batch48结果。

| Sign seed201 | M2 | H2（原拒绝版） | H2b/M3 | Multi-Krum |
|---|---:|---:|---:|---:|
| 攻击期平均ACC % |80.1984|79.9716|80.3582|79.9004|
| 最终ACC % |84.20|84.52|85.06|85.16|
| 良性误标 /351 |4|0|0|不适用|
| 每轮恶意权重 % |5.3510|5.6019|5.2148|4.8000|

H2b对M2平均+.1598pp、最终+.86pp；对H2平均+.3866pp、最终+.54pp。
round52保留四条良性误标撤销；round53合格参考由4/7恢复7/7，正确拒绝攻击者0/10/16，权重全部0。
相对M2首次分叉52，相对H2首次分叉53。H2b撤销恶意标记3→0。
其他七个条件/seed配对无权重或聚合sketch分叉；clean/Gaussian/LIE及Sign205效用保持。
Sign205 M3平均79.7482、最终84.40，MK79.0900/84.31。
LIE .5仍未解决：M3 seeds201/205平均75.4416/72.5074，MK75.9836/74.7102。
M3只是IID、batch96、seeds201/205、四条件工程接受；不声称显著性、targeted或非IID泛化。
完整各条件ACC、权重、方向代理、回填质量与耗时见review.summaries；不同LOO参考轴的投影不解释成统一净攻击量。

版本继承关系是：M2=B0+R1c+R2；M3=M2+H2b机制（参考主体资格门及confirmed历史写入）。
H2b闭环实验从一开始就在M2上运行该组合，本次验收后将此组合晋升M3，保留M2的全部已接受机制。
`rtc_reference_memory`是H2b实验批次/输出目录名，包含M2、原H2、H2b组合及MK/RFA对照。
`rtc_i12_eligibility_confirmed`是该批次中“M2+H2b组合”的防御运行别名；接受后继续用于执行M3。
它不是仅运行H2b单一模块，也不是另一个研究版本；实验批次名、运行别名与主线版本名是三个不同层次。

## 保留候选及下一唯一研究问题

M2-H2的旧失败和快照不改，历史写入缺陷由H2b修复并在上述范围晋升M3。
M2-G1仍保留待修复/重新集成，原+5.1524pp LIE收益与安全失败都保留；本次没有默认启用G1。
两个登记共17项快照哈希再次核验一致；大型证据zip仍仅本地，Git同步不等于备份。

固定规则诊断：G2从G1参考中排除当轮R2明确拒绝主体，仍要求至少9个其他正质量主体。
同主体任一R2拒绝则该主体全部排除；参考不足或无效沿用重置连续次数，缺席不更新。
不改两次参与、q0和旧阈值来筛选攻击数据，不改M3的R1c/R2、资格历史、累计或回填。
10人协议中任一R2拒绝会导致低尾参考不足而弃权，这是明确取舍：R2继续约束已识别幅度攻击。
该过滤并不重算坐标中位数anchor，也不能保证在R2完全漏检或多数污染时安全。

对16条旧G1/M2观测轨迹按原阈值重放，Gaussian201候选8条低尾误标→0；Gaussian202的1条→0。
LIE候选201/202攻击标记125/141不变，良性1/0不变；clean、Sign低尾标记不变。
来源CSV逐条与保留审计哈希匹配；入口replay_g1_filter.py，结果g1_filter_replay.json。
这是旧轨迹证据，不证明修正后的闭环ACC，也不把旧batch48阈值默认推广到96。

## 本批4项：clean106校准、clean107留出

种子注册前扫描logs未发现对应训练文件；只属于本仓库记录未使用声明。
每seed两项：集成主线M3（M2+H2b机制，运行别名`rtc_i12_eligibility_confirmed`）；
M3+只读G1残差观测（运行别名`rtc_i12_confirmed_lower_observe`）。
唯一配置差异为lower_tail_mode=observe及其原观测校准路径，不施加低尾cap。
新G2打分器已实现并用合成输入测试，但本批仅离线回放，不接入训练决策。
新增4个训练单元、复用0；不包含任何攻击或后续自动启动。

CIFAR10、ResNet18无预训练、IID20人每轮10人，60轮、本地5epoch、batch96；
SGD lr.01/momentum.9/weight_decay.0001/cosine，严格配对、确定性客户端训练。
M3所有已接受机制保留，floor.5、cumulative power1、MAD2.5、accepted recycle.51。
Ray每client CPU1/GPU.125，object store3072MiB、最低可用内存10240MiB、等待120秒、自动重试0。
协议config/rtc_g2_clean_calibration_protocol.json，输出logs/rtc_g2_clean_calibration。

质量门：四项完整61轮/600客户端、runner所有门、配对及预算/重构/累计/资格状态验收；
两组逐轮ACC/loss、聚合sketch和客户端权重、原cap必须一致，计时无需相同。
观测日志独立重算原低尾中位数/比值/连续计数/q=1，确保观测不改变训练。
校准规则：只取clean106全部600条有效过滤后比值的最小值，严格ratio<threshold、连续两次；
必须0<threshold<1。clean107固定留出，600条有效、低尾标记率及与M3联合误标率均≤1%。
任何失败停止，不用107调整阈值；候选accepted固定false，即便校准合格也不是G2闭环接受。
之后再预登记M3/未修复G1/修正G2的LIE收益及clean/Sign/Gaussian安全回归，不静默丢弃G1。

21项纯合成检查通过，随后随机流一致性检查修改后重跑5项G2测试通过。
PowerShell语法/Windows四项默认dry-run、Bash -n/LF检查通过；Linux实际dry-run待用户返回。
缺少训练产物时分析器正确FileNotFoundError拒收，未开始训练。
本地冻结281份源码/14份产物，status/raw均0。
锁fc361b75bbc245680afa1be9ac1184a396582ef29c69930cc1a46122ecffbe36；
源码zip454907af4a910702c25df6b5ca42cb86eaeeee93ee86038df3c72de94822f8e4。

## 手动Git同步

遵循用户最新手动提交要求，本回合未自动commit/push。当前HEAD897a5e8。
仅提交本批文件；其他工作区改动保留，检查暂存区后由用户提交。

```powershell
Set-Location 'D:\workspace\FL2'
git add -- defenses/rtc/filtered_lower_tail.py experiments/rtc_v3/byzantine.py experiments/rtc_g2_clean_calibration.py experiments/run_rtc_g2_clean_calibration.ps1 experiments/run_rtc_g2_clean_calibration.sh config/rtc_reference_memory_accepted.json config/rtc_g2_clean_calibration_protocol.json tests/test_rtc_g2_clean_calibration.py analysis/rtc_reference_memory_review analysis/rtc_retained_candidates/README.md analysis/rtc_retained_candidates/M2-G1/candidate.json analysis/rtc_retained_candidates/M2-G1/followup_20260916.md analysis/rtc_retained_candidates/M2-H2/candidate.json
git add -f -- docs/RTC_H2B_REVIEW_G2_CALIBRATION_HANDOFF_20260918.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md
git diff --cached --stat
git diff --cached --check
git commit -m "Accept scoped H2b repair and prepare G2 clean calibration"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

服务器更新前保留本地环境改动，不强制覆盖旧日志。此批脚本已显式固定96/.125，不需要再改共享bridge。

## 人工执行边界

Windows唯一启动命令（已完成该入口dry-run）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_g2_clean_calibration.ps1' -Execute
```

完成后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_g2_clean_calibration.ps1' -Analyze
```

Linux更新代码后先只做无训练实机dry-run：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_g2_clean_calibration.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

Linux实机验证通过并返回后再交接其训练入口；两平台二选一，不重复两套4项。
Linux输出/home/jia_zhang/hqr/FL2/logs/rtc_g2_clean_calibration；训练完成后原宿主分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_g2_clean_calibration.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

完整单元经验证可跳过；failed/incomplete/orphan会拒绝自动重跑，不支持中间轮checkpoint续训。
阶段WAITING_FOR_MANUAL_EXPERIMENT；本回合progress，新人工边界计数1，正式blocked需同一阻塞连续三回合。
不自动训练、不后台等待、不提前启动攻击批次。R3其他攻击、R4、R5、targeted/非IID及最终报告未完成。
