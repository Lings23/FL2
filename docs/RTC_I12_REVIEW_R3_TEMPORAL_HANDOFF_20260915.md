# I12验收与R3跨轮残差观测交接

日期2026-09-15。I12通过本批工程验收；下一批R3只收集观测，不接受新防御，不启动训练。

## I12正式结论

用户复制的整批日志来自/home/jia_zhang/hqr/FL2的Linux服务器，并非原Windows准备批次。
40项completed/exit0/last_round60，逐项61轮、600客户端记录；104项runner质量门全部通过。
严格配对、实际模式、预算、几何重构、累计重放通过，独立重算24000条方向参考，最大cosine误差6.153e-12。
补回的四份服务器文件全部与原锁匹配；213份源码与66份计划产物完整核对。
72项组合门和26项独立幅度门通过。接受M2=B0+R1c+R2，限定IID/seeds44、45的clean、Sign-flip、Gaussian、LIE .5。
本次不是统计显著性或全攻击泛化证明。原R2单seed+0.2pp门失败历史保持不变。

| 两seed均值，active/final ACC % | B0 | M2 | Multi-Krum |
|---|---:|---:|---:|
| clean | 80.8309 / 89.310 | 80.8309 / 89.310 | 本批未安排 |
| Sign-flip | 77.2965 / 81.755 | 83.3571 / 87.460 | 82.7595 / 87.210 |
| Gaussian | 84.0389 / 87.820 | 84.2213 / 88.245 | 84.3480 / 87.860 |
| LIE .5 | 76.9455 / 86.630 | 76.9455 / 86.630 | 79.2681 / 87.310 |

Gaussian中RFA为84.8620/88.565。M2恶意权重由18.0648%降至0；clean、Sign-flip的原收益保留。
方向/幅度的联合新增良性cap误标均0；不表示既有累计机制从不限制良性权重。
LIE中四RTC组结果一致，新增两模块没有解决慢攻击；其active ACC仍比MK低2.3226pp。

权威证据：analysis/rtc_i12_review/review.json；独立导入审计audit_import.py。
服务器原锁SHA256：3011209b9cca01dba0a0789fe8c5606f9c23814a3084b60744a1c3d69faf3f6d。
完整源码归档analysis/rtc_i12_review/server_sources_complete.zip，SHA256：
d405e3a5efdcd0210a8209aa17f2934aab39064e64b2930a6ecf9137d51bb2f6。
便携接受凭据config/rtc_i12_accepted.json进入下一批冻结合同；原日志/锁不覆盖。

## R3下一步为何先观察方向持续性

M2的LIE seed44/45累计q首次下降为round33/30，攻击始于round11。
约62.61%/58.47%的恶意权重发生在q仍为1时；round11--20恶意平均z约1.0031/.9926，
良性约1.0066/1.0063，而累计kappa固定1.0547344481697019。
仅加重触发后的q惩罚不能覆盖主要延迟；简单降低门槛也无法凭这段幅度重合可靠识别攻击。
现有日志无客户端跨轮残差向量，无法离线重建跨轮方向一致性。因此先补观测，不改q指数/阈值/回填。

新增观测是trainable-only的“clipped delta减当前既有nominal coordinate-median anchor”，
固定splitmix64 seed20260915、512维归一化CountSketch；另保留真实残差范数与trainable维数。
不启用cone路由、不读取恶意身份作决策、不改变随机流、不输出cap，不更新防御状态。
离线统计每principal最近5次参与的单位残差sketch均值范数；零向量或不足5次时不可用，历史不按攻击时刻重置。
该值仅是压缩空间的持续性线索，不是全空间精确cosine，也不是净有害方向；非IID良性持续性尚未验证。

## 冻结矩阵与参数

| 条件 | seeds | 方法 | 新训练单元 |
|---|---|---|---:|
| clean | 103、104 | M2、M2+只读观测 | 4 |
| LIE z=.5 | 44、45 | M2、M2+只读观测、MK | 6 |

共10项，旧结果复用0。两个RTC组除temporal_residual_observe_only开关外配置严格相同。
clean103只作后续校准资料、104作独立clean验证；LIE44/45是开发诊断，不用于拟合阈值。
本批不设新的防御阈值；未来候选须另行预注册并完成包括既有攻击在内的回归验收。

CIFAR10/ResNet18无预训练/IID20客户端、每轮10人；60轮、本地5轮、batch48；
SGD lr.01/momentum.9/decay.0001/cosine；攻击round11--60、恶意比例.3、LIE all_updates；
strict/principal_uniform/确定性训练。M2双cap开启，floor.5、cumulative power1、accepted回填.51、MAD2.5。
MK f3选5。每client CPU1/GPU.25、object store3072MiB、最低可用内存10240MiB、等待120秒、重试0。
Linux有两张5090，不能静默更改batch或并发配置。执行Python显式指定实际conda环境。

## 质量验收与验证

15项合成测试通过，覆盖多轮phase6输出/权重/累计状态不变、trainable作用域、随机流不变、
持续/交替/零残差历史、人工执行边界、I12双cap回归。未运行任何含真实训练的测试。
PowerShell语法与10项最终dry-run通过；Bash -n通过，LF换行。Linux实际dry-run仍待人工执行。
缺少训练产物时分析器拒绝验收，已核实。真实观测验收必须满足：
- 10项完整、全部runner质量门、严格配对、有限值、原双cap与累计/预算门。
- 观测版与同批M2逐轮ACC/loss/聚合sketch、逐客户端权重及原cap完全相同。
- 观测版每条512维有限sketch、元数据/作用域一致；累计重放误差≤1e-9、预算≤1e-8、重构相对误差≤1e-4。
- 任一门失败则拒绝本批观测，不宣称防御提升；observation通过也保持candidate_accepted=false。

最终Windows目录D:\workspace\FL2\logs\rtc_r3_temporal_observation，237份源码、25份冻结产物，status/raw均0。
锁SHA256：07e18895b2ae781a476e8ac6b3701c3a8a6f92b4a09cc1ce00fea1e3f47ebce0。
随日志保存r3t_sources.zip，SHA256：2065f7b03b5e47b977c55999da93831258ddcdb044a083a4adb54f041f7cb5cb。
此前pre_archive_dryrun目录只是未训练草稿，不执行、不复用。

## 人工Git同步

按用户最新要求，以下修改由用户手动提交/推送；本次没有自动commit/push，也不声称GitHub已同步。
先查看暂存区，确认没有其他待提交内容。以下只加入本批源码/配置/审计与明确文档，不加入数据和训练日志。

```powershell
Set-Location 'D:\workspace\FL2'
git diff --cached --stat
git add -- defenses/rtc/v3.py defenses/rtc/temporal_observe.py strategies/fed_strategy.py experiments/rtc_v3/byzantine.py experiments/rtc_r3_temporal_observation.py experiments/run_rtc_r3_temporal_observation.ps1 experiments/run_rtc_r3_temporal_observation.sh tests/test_rtc_r3_temporal_observation.py config/rtc_i12_accepted.json config/rtc_r3_temporal_observation.json analysis/rtc_i12_review/audit_import.py analysis/rtc_i12_review/review.json
git add -f -- docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md docs/RTC_I12_REVIEW_R3_TEMPORAL_HANDOFF_20260915.md
git diff --cached --check
git commit -m "Accept I12 integration and prepare inert RTC temporal observation"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin HEAD:refs/heads/codex/periodic-attack-defenses
```

旧I12源码归档与训练日志不作为新批次运行时依赖；离线历史复核需要另行保留它们。

## Windows与Linux人工入口

Windows已完成dry-run，可以人工训练（与Linux二选一，勿重复两套）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_r3_temporal_observation.ps1' -Execute
```

Windows结果完成后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_r3_temporal_observation.ps1' -Analyze
```

Linux先更新工作分支，然后只验证，不训练：

```bash
cd /home/jia_zhang/hqr/FL2
git pull --ff-only origin codex/periodic-attack-defenses
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_r3_temporal_observation.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

返回Linux dry-run结果/环境/锁并核验后，才交接真实训练命令。Linux输出为
/home/jia_zhang/hqr/FL2/logs/rtc_r3_temporal_observation；不要复制Windows锁代替本机prepare。
之后在原服务器分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_r3_temporal_observation.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

脚本再次执行时仅在runner验证后跳过完整单元；失败/部分/孤儿产物先拒绝并要求人工复核。
不支持训练中间轮的checkpoint续训。拷贝整批结果时保留r3t_lock.json与r3t_sources.zip。

阶段WAITING_FOR_MANUAL_EXPERIMENT。新阻塞首次交接计数1，未满足连续三目标回合的正式blocked条件。
停止依赖新结果的开发，不后台等待，不推进下一算法候选。R4/R5、targeted及最终泛化目标仍未完成。
