# R3持续小残差验收与参考可靠性观测交接

日期：2026-09-15。原批次质量通过、候选未接受；保留LIE有效机制，先处理父版本参考可靠性缺陷。
总体目标未完成。下一边界：WAITING_FOR_MANUAL_EXPERIMENT，5项只读观测由用户手动执行。

## 24项服务器结果

来源logs/rtc_r3_lower_tail已被外部Linux结果替换，不是旧Windows准备目录。
24/24项completed、exit0、round60；各61个唯一轮号、600条客户端记录，92/92 runner质量门通过。
228份源码归档和51份计划产物均匹配原锁。严格配对、配置、攻击/数据/初始化/采样随机流、
预算/质量守恒/累计状态/低尾状态重放通过；独立重建14400条谱参考，最大cosine差3.8863e-12。
本地复算与服务器decision的全部38个验收门和指标一致，35门通过，3门失败。

审计：analysis/rtc_r3_lower_tail_review/verify.py；证据：同目录review.json。
原锁SHA256：a727c6d498f0e2f75c8ccbf022b0807623e51064df7cdc4a66d2f88865d3f966。
原源码zip SHA256：56cf30ef2e81a3a77708dffe425b7f0977be83c2960395803550ba7ad0e9a1e3。
原始源码、锁和服务器decision保持不变；新增审计仅做经过哈希核验的跨平台JSON路径映射。

下表为seeds201/202等权均值，百分数；clean窗口1--60，其余11--60，不能跨窗口当成攻击损失。

| 条件 | M2平均/最终ACC | M2+低尾cap平均/最终ACC | MK平均/最终ACC |
|---|---:|---:|---:|
| clean | 80.8079 / 89.140 | 80.8073 / 89.100 | 未运行 |
| Sign-flip | 83.7441 / 87.610 | 83.7468 / 87.580 | 83.3373 / 87.435 |
| Gaussian | 84.2281 / 88.005 | 84.1898 / 87.900 | 83.7200 / 87.360 |
| LIE .5 | 77.5934 / 87.505 | 82.7458 / 87.955 | 76.6116 / 87.225 |

LIE每seed平均增益分别+4.1152/+6.1896pp，最终+.23/+.67pp；恶意权重25.1162%→3.6%、
26.8754%→1.2%，两seed分别标记125/143、141/147攻击更新。LIE良性误标1/357、0/353。
这是保留机制的闭环证据，不能因回归失败就删除该方向，也不能据收益放宽原误伤门。

三项失败分别为Sign-flip seed201父版本和候选联合良性误标均4/351=1.1396%，
Gaussian seed201候选8/354=2.2599%。其他预注册效用、恶意权重和良性门均通过。
批次保持拒绝；低尾G是“有闭环效用证据、回归未通过、待修正”的研究模块，不能晋升M3。
M2的历史I12接受范围保持，不能声称其seed201安全转移通过。

## 两类缺陷与修复顺序

1. 父版本R1c：Sign-flip seed201 round52实际6/10参与者为攻击者，3条良性误标的谱参考
   全由这6人组成。多数确认只保证同轮一致性，无法证明多数方向可信。另一条良性误标在round60，
   参考为良性但本地更新弱反向。这两类不能混为“阈值稍微太宽”。候选round52同样误伤3条，
   另1条来自round47低尾cap。不得删掉6攻击者轮次、限制抽样后冒称修复原压力协议。
2. 新低尾G：Gaussian已由raw cap拒绝的超大残差仍进入低尾参考中位数；攻击者占5人时，
   良性客户端留一参考中的攻击者成为多数，使良性ratio被压低至约.20--.23。
   其他晚期误伤也有参考分布抬高与轨迹效应，不能全部归因于同一个极端轮。
   后续独立研究“参考排除已确认异常并重新校准/不足时弃权”；不随意降低min_peers=9或事后改阈值。
3. Gaussian补充：新seed201 round35有6个攻击者，R2留一中位数也被攻击者占据，ratio约1，
   该轮恶意权重31.3449%，活跃期平均.6269%；seed202为0。
   之前I12 seeds44/45的恶意权重为0是其已验证范围，不能外推为所有seed都会归零。
   当前缺陷在参考污染和强扰动识别的条件限制，继续增大std不会解决它。

先补M2参考可信度的跨轮证据，再单机制修复父版本；之后回到已保留G的参考修复和组合验收。
父版本修复不能永久丢弃G的LIE收益；最终范围仍包括全部攻击、DBA/scaling安全、非IID和新seed。

## 下一批为何需要观测

已有Gram只记录同轮向量关系，不能恢复跨轮旋转关系；没有本轮客户端向量与上一轮实际聚合位移的点积。
仅凭旧标量日志不能验证“上一轮实际更新能否识别本轮错误参考朝向”，故先采集精确全trainable点积。
上一轮实际更新也可能被污染，并非可信oracle；本批不创建阈值、cap或防御接受结论。

父版本rtc_i12_combined=M2，观测版rtc_i12_reference_observe唯一增加reference_history_observe_only=True。
R1c/R2实际cap和原累计/裁剪/accepted回填保留。G尚未接受，本批不启用它；这是明确的父版本缺陷诊断。
日志包含每客户端与上一轮实际聚合位移点积、谱参考夹角、可用性、前一位移范数和向量hash链。
训练和聚合随机流不变。只存上一轮trainable位移，约89MB float64持久状态，不保存全模型历史。

固定seed201：clean父版/观测2项，Sign-flip父版/观测/MK3项，共5个新训练单元，缓存复用0。
两份trial-plan与原seed201相同，保留round52的抽样压力；当前代码合同不同，旧结果不作同合同缓存。
这是已知失败seed的开发诊断，不冒称新seed或泛化验证。后续修正必须另行预注册新seed验证。

训练：CIFAR10、ResNet18无预训练、IID20客户端/每轮10人、60轮、本地5轮、batch48，
SGD lr=.01、momentum=.9、weight_decay=.0001、cosine；mf=.3、攻击round11--60、Sign-flip scale1。
M2 floor=.5、cumulative power1、accepted recycle=.51、MADk2.5；MK f3选择5。
Ray每client CPU1/GPU.25、object store3072MiB、最低可用内存10240MiB、内存等待120秒、自动重试0。

预登记质量门：全部5项完整、所有runner质量门及严格配对；父版/观测ACC、loss、模型sketch、
客户端权重、原cap和累计q逐项一致；budget≤1e-8、重构相对误差≤1e-4、累计重放≤1e-9，
跨轮向量hash链一致、前一范数匹配上轮实际位移范数、点积扩展Gram半正定相对容差1e-8、
独立重算夹角误差≤1e-10。初始轮或零向量必须记缺失，不填0伪造方向。
分析全部轮次、参考可用率、良性/攻击标记分布和参考污染；不得只挑round52报告。
candidate_accepted始终false，观测质量通过不等于M2安全门已修复。

12项纯合成测试通过，包含真实RTC聚合路径的观测不变性、history事务/零向量、错误点积与hash拒收、
失败运行禁止启动及原低尾回归；没有数据集训练。PowerShell语法与默认5项dry-run通过，Bash -n通过。
Linux实际dry-run尚未返回，故Linux只交接验证命令。
Windows冻结目录logs/rtc_reference_history_observation：251份源码、14份计划产物、5个单元、status/raw均0。
锁SHA256：c04a971db980be0d6e92a1b4bb94e950d18527de08745b3a63fdf51089211903。
源码zip SHA256：8f7958244fdb525e4749d90ea9aa0cc43ec87bc42a0161e697b5a396c21c2606。

## 人工Git同步

遵循用户最新“手动提交”指示，助手未提交或推送。本地基于16a1df2，远程分支是否包含新改动尚未核验。
只加入本次文件，保留其他未提交工作；数据、checkpoint、完整日志和源码zip不提交。

```powershell
Set-Location 'D:\workspace\FL2'
git add -- defenses/rtc/v3.py defenses/rtc/reference_history.py experiments/rtc_v3/byzantine.py config/rtc_reference_history_observation.json experiments/rtc_reference_history_observation.py experiments/run_rtc_reference_history_observation.ps1 experiments/run_rtc_reference_history_observation.sh tests/test_rtc_reference_history.py analysis/rtc_r3_lower_tail_review/verify.py analysis/rtc_r3_lower_tail_review/review.json
git add -f -- docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_LOWER_TAIL_REVIEW_REFERENCE_HISTORY_HANDOFF_20260915.md
git diff --cached --stat
git diff --cached --check
git commit -m "Audit lower-tail regression and prepare reference history observation"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

Linux代码需先包含上述提交。保留原服务器日志目录与源码zip；不要用本地Windows新锁覆盖Linux输出。

## 人工启动与分析

Windows已完成dry-run，唯一真实启动命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_history_observation.ps1' -Execute
```

完成后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_history_observation.ps1' -Analyze
```

Linux实际环境以最近服务器锁为准，REMOTE_L20.md中的旧/root/FL2/L20路径不适用于此服务器。
先人工git pull --ff-only origin codex/periodic-attack-defenses，然后仅执行验证：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_history_observation.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

返回Linux dry-run、r1h_lock.json及环境后核验，再交接真实训练命令。两平台二选一，不重复两套训练。
Linux预期输出/home/jia_zhang/hqr/FL2/logs/rtc_reference_history_observation，保留r1h_sources.zip。
未来实际训练完成后，在原服务器分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_history_observation.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

已完成单元由runner核验后跳过；failed/incomplete/orphan拒绝自动重跑，需要人工复核。
不支持中间轮checkpoint续训。不启动训练、不后台等待、不推进依赖新观测的候选。
本回合是progress；旧实验阻塞解除，新人工边界计数1，尚不符合宿主连续三回合正式blocked规则。
