# RTC 拜占庭鲁棒性目标状态

## 2026-09-15 / 最新：I12正式接受，R3跨轮残差观测等待人工执行

本节优先于下方历史状态。服务器源码包已补齐并与原锁逐文件一致，213份来源证据完整。
I12的40项、104项质量门、72项组合门和26项幅度门全部通过，接受M2=B0+R1c+R2，
范围限于IID/seeds44、45的clean/Sign-flip/Gaussian/LIE .5。原R2历史失败不改写，总目标未完成。
接受凭据config/rtc_i12_accepted.json，完整审计analysis/rtc_i12_review/review.json。
源码归档server_sources_complete.zip的SHA256为d405e3a5efdcd0210a8209aa17f2934aab39064e64b2930a6ecf9137d51bb2f6。

主线LIE累计首次惩罚round33/30，62.61%/58.47%恶意权重发生于q=1；早期z与良性重合。
因此下一批只观察trainable残差跨轮方向持续性，不通过简单加重后期惩罚冒称解决检测延迟。
M2不变，唯一新增为只读512维固定CountSketch导出；不加入新cap、不改变累计/回填/裁剪。
clean103/104各M2/观测两项，LIE .5 seeds44/45各M2/观测/MK三项，共10个新单元、旧基线复用0。

15项合成测试、PowerShell语法及最终10项dry-run、Bash语法通过；Linux实际dry-run待人工验证。
Windows输出logs/rtc_r3_temporal_observation，237份源码、25份冻结产物、status/raw均0；锁SHA256：
07e18895b2ae781a476e8ac6b3701c3a8a6f92b4a09cc1ce00fea1e3f47ebce0。
新批次随日志自动保存r3t_sources.zip，避免复制结果后缺少服务器源码。
协议config/rtc_r3_temporal_observation.json；完整参数/质量门/Git/双平台命令/恢复语义见
docs/RTC_I12_REVIEW_R3_TEMPORAL_HANDOFF_20260915.md。

阶段WAITING_FOR_MANUAL_EXPERIMENT；本回合为progress，旧源码阻塞解除，新人工边界计数1。
未启动任何训练、不后台等待、不晋升未经观测的新R3机制；正式blocked尚不满足宿主三回合规则。
按用户最新指示，本批文件由用户手动提交与推送，命令已写入交接文档；未宣称GitHub同步。

首次自动续行复核：R3-temporal的锁仍为07e18895…47ebce0、Windows准备环境，status/rounds/raw各0；
未发现本机python/pythonw/raylet进程，也未收到Linux实机dry-run证据。
上一回合为progress，本回合为no progress；无可轮询活动句柄，不记为verified wait。
同一人工实验阻塞连续计数2，尚未满足正式blocked条件。未重跑dry-run/测试、未训练或推进下一阶段。

第二次自动续行复核：锁及Windows环境不变，status/rounds/raw仍各0，本机无python/pythonw/raylet进程，
也未返回Linux dry-run证据。上一回合与本回合均为no progress，无活动句柄，不属于verified wait。
同一人工执行阻塞已连续3个目标回合成立；准备工作完成，无可独立推进的必要工作，按宿主规则正式blocked。
阶段保持WAITING_FOR_MANUAL_EXPERIMENT，等待用户的Linux验证或实验产物；总体目标未完成，禁止自动训练或推进下一阶段。

---

## 2026-09-15 / I12服务器结果已返回，数值门通过，待补源码证据

本节优先于下方历史交接。用户确认整个logs目录来自外部服务器复制；不能再把本地目录视为未执行的Windows准备批次。
阶段WAITING_FOR_SOURCE_EVIDENCE，总目标未完成。未启动训练、未修改服务器原始日志/锁、未晋升组合或推进新R3。
服务器锁SHA256：3011209b9cca01dba0a0789fe8c5606f9c23814a3084b60744a1c3d69faf3f6d。
环境Linux x86_64/Python3.11.16/torch2.14.0+cu130/numpy2.4.6/CUDA13.0，两张RTX5090；
仓库/home/jia_zhang/hqr/FL2，Python/home/jia_zhang/miniconda3/envs/fl2/bin/python3.11。
本批内部配对，不与旧Windows结果混算；协议内容与预注册config/rtc_i12_protocol.json完全一致。

40/40项completed、exit0、last_round60，逐项61个唯一轮号、600条客户端记录；104/104 runner门通过。
66份冻结计划产物哈希一致。离线导入审计仅对JSON读取做哈希校验后的绝对路径映射，保留原分析器的同宿主执行锁约束。
严格配对、原验证器机制/预算/重构/累计重放全部通过；独立向量参考复核24000条，最大cosine差6.153e-12。
预注册组合72/72门与独立幅度26/26门通过；candidate_accepted暂为null，源码完整性未通过前不宣布正式验收。

两seed等权均值（active/final ACC）：
- Sign-flip：B0 77.2965%/81.755%，组合83.3571%/87.460%，MK82.7595%/87.210%；方向/组合结果一致。
- Gaussian：B0/方向84.0389%/87.820%，幅度/组合84.2213%/88.245%，MK84.3480%/87.860%，RFA84.8620%/88.565%。
  组合将每轮恶意权重由18.0648%降至0；平均ACC增加0.1824pp，原R2历史+0.2pp门失败不改写。
- clean：四RTC组均80.8309%/89.310%；LIE .5四组均76.9455%/86.630%，MK79.2681%/87.310%。
  LIE尚无收益，相对MK平均低2.3226pp；不能把组合门通过称为已解决慢攻击。
- 四RTC组的实际联合良性cap误标率均0；该结论仅覆盖这批IID/seeds44、45，未作显著性/非IID/targeted推广。

服务器213份源码中209份恢复并匹配原始哈希（160份本地字节一致，46份仅换行差异，3份匹配git a0a90b0）。
尚缺4份原始服务器文件：
config/rtc_v3_formal_all_attacks_two_seed.linux.freeze.json；config/rtc_v3_seed42_signflip_v2_linux.freeze.json；
experiments/run_rtc_i12_bridge.sh；experiments/run_rtc_formal_attacks_seed42_linux.sh。
下一动作只补这四份来源证据，核验与锁中的哈希一致；无需重跑40项实验。完整锁与缺失哈希见review.json。
审计脚本analysis/rtc_i12_review/audit_import.py；结果analysis/rtc_i12_review/review.json。
补回文件保持原相对路径置于analysis/rtc_i12_review/server_sources后，运行该脚本重新核验。
若原文件已改变，须找回匹配的归档；不重写锁以迁就当前源码。

本回合属于progress：真实训练产物解除人工实验阻塞，完成跨平台产物及机制复核。
新的源码证据阻塞首次出现，尚不满足宿主连续三回合blocked条件；不后台等待。
遵循用户最新“手动提交”指示，本回合新增审计与状态修改留待人工git提交/推送，未声称GitHub已同步。

2026-09-15首次自动续行复核：服务器源码包及server_sources目录均未到达，服务器锁哈希不变。
上一回合为progress，本回合为no progress；没有已确认的活动任务句柄，不记为verified wait。
同一源码证据阻塞连续计数=2（含首次交接），尚未达到正式blocked阈值。
保持WAITING_FOR_SOURCE_EVIDENCE；未重复分析/测试、未启动训练、未推进依赖完整验收的新候选。

---

## 最新交接：R3观测已验收，I12等待人工实验

日期：2026-09-14。阶段WAITING_FOR_MANUAL_EXPERIMENT，目标未完成；本节优先于下方历史状态。
R3三项completed/exit0/round60，20/20质量门通过，原分析器与独立verify.py一致。
LIE .5：RTC/MK平均ACC78.0796%/76.9732%，final86.03%/87.09%；方向与raw标记均0/143攻击、0/357良性。
累计首次下降round34，63.7270%恶意权重发生于q=1；两机制不能据此宣称已覆盖慢攻击。
R3源码归档SHA256：53125c3d3a15c3c621a24572c77f94a4f51a85235e67f20fdf108a44482acee3。

I12父版本rtc_i12_direction（B0+R1c），候选rtc_i12_combined（仅再启用固定R2 raw cap），
辅助B0/N，seeds44/45，clean/Sign-flip/Gaussian/LIE.5共40个新训练单元，旧基线复用0。
Windows输出logs/rtc_i12_bridge_v2，锁SHA256：1669371a3c4ea30d735a1eb03d91ae03ecd170553ec91fa9b7c103262ec0073a。
11项I12测试及11项raw模块合成测试通过；Windows语法/dry-run和Bash语法通过，Linux实际dry-run待用户执行。
协议config/rtc_i12_protocol.json；完整参数、数值验收门、版本范围和恢复说明见
docs/RTC_R3_REVIEW_I12_HANDOFF_20260914.md。旧R2失败不改写，N去留与组合判定分开。

Windows人工命令：powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_i12_bridge.ps1' -Execute
分析：同一命令将-Execute换为-Analyze。
Linux先更新GitHub工作分支，再只执行bash /root/FL2/experiments/run_rtc_i12_bridge.sh，返回dry-run与环境供核验。
Linux真实训练命令待其dry-run通过后交接；两平台二选一，不重复运行两套40项。
GitHub本轮同步结果以实际commit/push核验及最终回复为准，尚未推送时不得宣称已同步。

本回合为progress；旧R3等待已经解除。新I12人工实验等待连续计数=1，正式blocked条件尚未满足。
不启动训练、不后台等待、不推进依赖I12结果的新R3算法；R4/R5和最终报告仍待完成。

---

## 当前状态：阶段顺序评估与逐步集成规则修订

更新日期：2026-09-14。本次仅完成流程评估和文档修订，总体算法改进目标尚未完成。
当前下一动作以本节为准；下方保留历史交接记录，其等待状态与研究去留不代表最新状态。

- 固定对照B0=B3R-F0.51；主线起点M1=B0+R1c，当前接受范围为已验收的seed43 Sign-flip/clean。
- R2原始批次未通过+0.2pp平均ACC增益门，历史判定保持；幅度模块N保留为有前景、待验证的研究分支。
- 下一步先完整验收R3-observe产物。三个计划run的status均completed、exit0、last_round60；
  本次仅核对状态文件，尚未完成全部质量门、严格配对与观测分析，不能称为实验验收通过。
- 该批R3是B0方向/幅度observe诊断，未启用R1c或R2的实际cap；原脚本、run_id、manifest和日志不改。
- 随后准备新预登记的I12/主线转移桥接：B0、B0+D、B0+N、B0+D+N；组合直接与M1比较。
  Gaussian/clean/Sign-flip及扩展条件的转移/回归门明确后，才可能晋升组合为新主线。
- 后续R3、R4继承当前已接受主线，只增加一个机制，同时验收新增终点与旧收益回归；失败保留父版本。
  R5负责已集成版本最终泛化及消融，不能首次组合各模块。
- I12确切seed、训练单元数、数值验收门、脚本与冻结合同尚待准备，本次没有交接新训练命令或启动训练。

评估：docs/RTC_STAGE_INTEGRATION_REVIEW_20260914.md。
执行规则：docs/RTC_BYZANTINE_GOAL_PROMPT.md。
技术方案：docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md。
原提示词与技术方案备份：docs/history/*before_integration_20260914.md。
所有真实实验仍由用户手动执行；以后每批脚本准备、测试和dry-run完成后，交接命令并等待结果。

---

# 历史记录（被上方当前状态更新的安排不再执行）

# R2验收与R3方向观测人工交接

日期：2026-09-14。R2原始幅度cap候选拒绝；R3-observe状态WAITING_FOR_MANUAL_EXPERIMENT。总体目标未完成。

## R2真实结果与拒绝依据

6项人工训练均completed、exit0、last_round60，每项61个唯一轮号、600条客户端记录。
22/22 runner质量门通过；原分析器验证冻结源码/参数/攻击/数据/初始化/采样/随机流以及预算与几何重构。
原8项验收门中7项通过，只有Gaussian平均ACC增益失败。分析器exit0代表分析完成，不代表候选通过。
独立verify.py直接重算所有关键指标、LOO范数比、cap落实、质量守恒、参考方向和配对数据，结论一致。

| Gaussian seed42 round11--60 | RTC基线 | R2幅度cap | Multi-Krum | RFA |
|---|---:|---:|---:|---:|
| 平均ACC | 83.9470% | 84.1034% | 84.0576% | 84.6922% |
| 最终ACC | 87.19% | 87.60% | 87.21% | 88.30% |
| 每轮恶意权重 | 17.1728% | 0% | 0% | 1.4804% |
| 良性raw标记 | 0/365 | 0/365 | 0/365 | 0/365 |
| 攻击raw标记 | 135/135 | 135/135 | 135/135 | 135/135 |

基线/MK/RFA标记仅观测，只有R2候选实际限制权重。
R2相对RTC平均+0.1564pp，小于预登记+0.2pp，差0.0436pp；最终+0.41pp。
相对MK平均+0.0458pp/最终+0.39pp；相对RFA平均-.5888pp/最终-.70pp。
不能把正向改善或接近门槛当作通过，也不能把单seed的小差值当统计显著结论。
clean两组平均81.0985%、最终89.47%，标记0/600，客户端CSV、ACC和聚合sketch轨迹一致。
首次Gaussian cap、客户端权重和sketch分叉均round11；clean无分叉。
配对可用攻击投影109/135，负向贡献代理1.23953e-5→0，各自LOO轴不同，不代表所有攻击影响为零。
Gaussian是非定向攻击，ASR不适用。随机压力11/50活跃轮超过f3，不能冒称理论保证。

## 机制诊断与下一动作

候选攻击范数比范围97.21--107.41，良性最大1.111；固定阈值3已经完整分开两组。
因此当前证据不支持再改范数阈值来增加检出率；更严阈值也无法把已经为零的恶意权重继续降低。
将q平方或替换阈值不能据现有证据保证获得缺少的0.0436pp效用，不做事后扫参。

| 权重分配与向量范数 | RTC基线 | R2幅度cap |
|---|---:|---:|
| 良性客户端权重和 | .729994 | .730000 |
| anchor mass | .050122 | .137700 |
| zero mass | .048156 | .132300 |
| effective mass | .951844 | .867700 |
| 实际trainable聚合范数 | 1.759702 | 1.079918 |

独立逐轮验证sum(client weights)+anchor+zero=1，anchor=.51×(1-sum weights)。
移除的攻击质量没有无条件转为良性权重；约49%的缺失质量保留为zero mass。
这可能影响收敛，但范数差还混合了被移除噪声及后续轨迹变化，不能解释成准确率损失的因果比例。
候选round11--20 ACC比基线低.652pp，之后四个十轮窗口分别高.445/.511/.298/.180pp。
这描述早期回退与后期收益；没有检验延迟cap、软cap或质量回收的反事实效果，不据此擅自推广这些改法。
RFA恶意权重非零且ACC更高，也再次说明“恶意权重更低”不是ACC更高的充分条件。

R2本次阶段以拒绝结束，保留原锁/阈值/判定，不接受进入组合；已通过R1c的结论保持。
其“通过后再做Random-v2转移”条件未满足，因此不对被拒绝版本安排转移训练。
继续独立R3诊断不是宣称R2通过；最终Random-v2、Gaussian及全部目标攻击的验证范围不减少。
已向用户提供先复核R2或继续R3的可选路线，未收到不同偏好时按推荐的R3路线准备；不以未回复授权训练。

## R3先诊断累计延迟

独立replay.py从12项V1存档（clean、LIE .25/.5、DBA各seeds42/46/47）的残差与固定校准重建累计状态。
7200条q完全一致，最大误差0。它复核现有检测器，不是新的训练或候选调参。
LIE .5首个攻击q下降round34/31/31，攻击始于round11；143/149/141条攻击记录中仅63/73/74条q<1。
q尚未下降期间的恶意权重占总恶意权重约63.73%/58.64%/55.60%。
直接q²在旧轨迹上最多再限制每轮约.02197/.02481/.02526的恶意权重，不改变q=1时期的延迟。
该数值只是旧轨迹上的上限比较，不是重新求解或新闭环效果。
clean q<1已有11/600、9/600、1/600，虽大都轻微；不能忽略q²对这些良性记录的额外限制。

因此下一步只补LIE的trainable方向证据：现有旧日志不能恢复这些向量方向，不能凭残差范数编造有符号累计量。
采用已实现且测试过的R1c谱/计票与R2 raw观测；两种模式都observe，q指数仍1，不启用任何新cap。
观测将检查参考可用性、方向标记是否比累计惩罚早、各principal参与次数、投影与anchor质量。
若现有参考对LIE失效，如实记录；不会为了得到分离调整已冻结阈值。观察完成不等于候选接受。

## 冻结的3项人工观测

新增训练3项、缓存复用0项：seed42 LIE z=.5 RTC B3R-F0.51/Multi-Krum两项，clean RTC一项。
独立输出logs/rtc_r3_lie_observation；沿用已有rtc_r2_raw_baseline/multikrum别名以复用只读观测实现，名字不表示启用R2 cap。
两个模式均observe；无R2 cap别名；原防御算法不修改。clean与前批同配置但在本批独立重跑，旧结果不覆盖。
LIE medium z=.5，coordinated_attack_knowledge=all_updates；mf=.3，round11--60；MK f3选5。
CIFAR10 ResNet18无预训练、IID20客户端每轮10人、60轮；本地5轮、batch48、SGD lr.01/momentum.9/decay.0001、cosine。
RTC floor=.5，cumulative power1，accepted回填.51，MADk2.5；strict/principal_uniform/确定性训练。
Ray每client CPU1/GPU.25，object store3072MiB，最低可用内存10240MiB、等待120秒，max-spec-retries0。

6项纯合成测试通过：三项观测完整分析、禁止接受候选、损坏轮次/采样/累计证据拒收、人工入口和缺失产物拒收。
PowerShell默认dry-run exit0，只生成manifest/matrix/trial plan/resolved；未调用训练入口。git diff --check通过。
R3锁SHA256：0687c4c25aeb33b3f44a290a731d9154137a23221f2144693cdb5594b2f6d87f；源码170份、产物10份。
LIE trial-plan：de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c。
clean trial-plan：1744c5203cce62dd2d9f8dcf6bed1ba285be3ef20c36b8643cd223135ea6d7ae。
观测只验收质量：完整61轮/600客户端、全部适用质量门和严格配对、来源与实际配置一致、有限性、
预算≤1e-8、几何重构相对≤1e-4、累计重放误差≤1e-9。candidate_accepted固定false，后续算法必须另行预登记与人工验证。

## 唯一人工启动与分析

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_r3_lie_observation.ps1' -Execute
```

完成后只分析：

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' 'D:\workspace\FL2\experiments\rtc_r3_observation.py' --analyze
```

新脚本遇到failed/incomplete/orphan产物会拒绝启动；不自动恢复中断轮次或补跑。

## 归档与等待状态

R2原锁20055715487715e0c3ada113f62f2ad78652b3b9451e416f1bd6862f94a59b88不变。
源码归档analysis/rtc_r2_review/r2_sources.zip，SHA256 f815c9050cbabf6f1f645cb6a71ef20d9af201ca9f9ea630471964bc80dc80a8。
原始decision和独立review保留；源码增加R3脚本后，原R2主分析器会因整库来源变化拒绝，历史可用归档与独立verify.py复核。
旧R2交接/阻塞记录保存在analysis/rtc_r2_review/r2_handoff_before_acceptance.md。

本回合为progress：R2完成产物解除旧阻塞，完成独立拒绝验收/归档、12项累计重放、R3观测准备。
R3新人工实验阻塞连续回合计数=1；阶段WAITING_FOR_MANUAL_EXPERIMENT。宿主需同一阻塞连续三个回合后才正式blocked。
交接后立即停止，不后台等待、不重复测试/dry-run、不训练、不预设R3候选有效。
恢复后先验收观测，再决定一个累计机制；R3转移、R4a/R4b、R5组合/泛化与最终报告仍待完成。

## R3交接后第一次自动续行复核

上一目标回合分类为progress：完成R2拒绝验收与归档、累计状态重放和R3观测人工交接。
本回合分类为no progress：仅复核外部状态，R3 status文件0、raw文件0，Python/pythonw/raylet进程0。
未确认活动训练句柄，不记为verified wait；同一人工实验阻塞连续回合计数=2（含首次交接）。
尚未满足连续三个回合的正式blocked条件。阶段保持WAITING_FOR_MANUAL_EXPERIMENT。
未启动训练、重跑测试/dry-run、修改冻结实验或推进后续候选；等待用户手动执行已交接三项实验。

## R3交接后第二次自动续行复核

上一回合与本回合均分类为no progress：必要状态复核未改变下一动作，没有活动训练句柄。
当前R3 status文件0、raw文件0，Python/pythonw/raylet进程0。
同一人工实验阻塞连续3个目标回合成立（首次交接与两次自动续行），无可独立推进的必要工作。
阶段保持WAITING_FOR_MANUAL_EXPERIMENT；按宿主规则正式标记目标blocked，等待人工执行三项观测。
未启动训练、修改冻结实验、重复测试/dry-run或推进后续候选。总目标未完成。

## 2026-09-14 / 用户反馈后的R2保留结论修正（以本节为最新研究去留判断）

R2改记为“未通过本次预登记效用验收，但保留为有前景的安全机制候选，待进一步验证”。
此前从单seed效用门失败直接推导结束R2后续验证，过于绝对；不再将其永久排除。
原协议、+0.2pp门、decision/review中的该批次拒绝结论及源码归档不变。
恢复固定多seed复核为计划待办；在新结果之前预登记完整seed集合、终点和停止规则，不试到通过为止。
若后续采用安全收益+效用非劣的组件定位，须明确登记为新研究问题，不追改旧结果。
详情见docs/RTC_R2_RETENTION_REVIEW_20260914.md。R3观测合同保持可执行；本次没有启动训练或新增训练批次。
