# RTC 拜占庭鲁棒性改进方案：已完成实验的补充诊断

日期：2026-09-11。状态：研究规划，未修改防御算法、未启动训练、未冻结新候选。

2026-09-14阶段流程修订：采用单项诊断与逐步集成并行的研究流程。以下原始证据和历史追加记录
保留各自时间语境；当前版本继承以RTC_BYZANTINE_GOAL_PROMPT.md及
RTC_STAGE_INTEGRATION_REVIEW_20260914.md为准。R1c是已验证Sign-flip/clean范围的主线起点，
R2保留待验证；先做I12/主线转移桥接，再在已接受主线上开展R3、R4，R5不再承担首次组合。

结论：新增证据将首要问题从 DBA 专项隔离转向一般拜占庭攻击的即时方向识别。
保留 B3R-F0.51 为研究对照；建议依次验证即时方向 cap、裁剪前异常幅度 cap、
慢速累积证据、DBA 隔离与 anchor 安全。每个候选相对上一已接受主线只改一个机制，
失败只回退本次新增部分；B3R-F0.51长期保留为固定归因对照。

## 1. 本次证据范围和可复现入口

源目录：`logs/rtc_v3_formal_all_attacks_two_seed_mf03`。
审计脚本：`analysis/rtc_formal_partial_20260911/audit_partial.py`。
输出：`audit.json`、`inventory.csv`、`runs.csv`、`windows.csv`、`rtc_groups.csv`、
`paired_comparisons.csv`、`excluded_old_statuses.csv`。

```powershell
& 'D:\workspace\FL2\.venv\Scripts\python.exe' 'D:\workspace\FL2\analysis\rtc_formal_partial_20260911\audit_partial.py'
```

以上命令只读取训练产物并重算本目录分析；后续重跑可能反映新的完成状态，快照时间见 audit.json。

- 182 个 manifest 计划格中，47 格已完成，134 格无对应 status，1 格失败。
- clean 为 FedAvg seeds42/46，共2格；Gaussian与Random-noise分别20格，三批共42格，
  已有质量门全部通过。clean并非所有防御的无攻击基线。
- Sign-flip当前manifest中seed42的FedAvg、RTC、Krum、Multi-Krum、Trimmed Mean共5格
  完成；Median因attack implementation hash不一致失败；整批quality_gates不存在。
  五格仅作为经过本次逐运行核验的初步证据，不能宣布该攻击正式验收。
- 排除4份不属于当前manifest的旧status，其中包含旧版Sign-flip已完成结果；不按目录文件数混算。
- 每个纳入运行核验completed/completed_cached、exit=0、round60、恰有0--60共61行、
  活跃期每轮10条客户端记录及唯一键、有限ACC/loss、可用的更新有限性字段。
- manifest与lock匹配；原始轮次中的trial-plan和attack implementation hash与spec匹配。
  RTC运行时floor=.5、q power=1、accepted recycle=.51、MAD=2.5全部核验。
- RTC与对手逐seed核验trial plan、攻击实现、数据manifest、初始模型、恶意身份以及
  实际选中客户端序列一致。可表示为标量客户端权重的算法，恶意权重另从客户端CSV重算。
- 当前源码/分析器与旧实验合同不一致，完整正式分析器preflight失败（本次先观察到attack freeze
  漂移，后续快照首先报`locked Random noise scope drifted`）。Random-noise已完成结果
  为`random_noise.rademacher_norm_matched.v1`、scale=1、weak；当前代码/分析器要求
  trainable-only v2。历史v1可作内部配对诊断，不能充当v2或更强强度的证据。
- 本次仅检查存档证据的内部一致性；不证明旧训练二进制与今天工作区完全相同，也不复用旧结果
  到新合同。新实验前先冻结当前源码/参数/数据合同，再判断基线是否可复用。

## 2. 已完成结果

下表是攻击活跃期round11--60平均ACC，百分数；先按run求平均，再对同攻击两个seed等权平均。
Sign-flip只有seed42，不与两个seed的项目混成总排名。

| 方法 | Gaussian，n=2 | Random-noise v1，n=2 | Sign-flip v2，n=1，初步 |
|---|---:|---:|---:|
| RTC B3R-F0.51 | 84.0479 | 83.5316 | 75.9582 |
| Multi-Krum | 84.1858 | 84.3691 | 82.6430 |
| RFA | 84.7939 | 83.6036 | 未完成 |
| Median | 84.5196 | 84.0885 | 无有效完成结果 |
| Trimmed Mean | 84.3273 | 84.1395 | 74.2834 |
| Krum | 79.2593 | 78.9710 | 71.5802 |
| FedAvg | 68.4742 | 83.4172 | 73.2868 |

精确数值以runs.csv为准。RTC-Multi-Krum平均ACC差为-0.1379/-0.8375/-6.6848 pp，
最终ACC差为-0.1000/-0.4650/-5.9700 pp。
RTC显著的问题集中于Sign-flip初步结果，并非全面落后所有常规方案；Gaussian差距很小。
上述均为描述性差值，不作显著性或等价性结论。

| 指标 | Gaussian RTC/MK | Random v1 RTC/MK | Sign-flip RTC/MK |
|---|---:|---:|---:|
| 恶意权重，% | 17.8423 / 0 | 26.4426 / 0.4000 | 27.8406 / 4.4000 |
| 恶意impact，% | 21.6973 / 0 | 29.5424 / 0.4021 | 29.9800 / 4.5591 |
| RTC zero mass，% | 4.7324 | 2.2821 | 2.2341 |
| RTC anchor mass，% | 4.9256 | 2.3753 | 2.3253 |

这些指标的最后几位按输出CSV核对；不把权重/impact当成方向安全的充分统计量。
Median/Trimmed Mean的标量客户端权重和impact缺失，保留NA，不能填0。
clean保留“恶意身份”只是实验身份分配，不代表该条件实际发动了攻击。

## 3. 根因：已证实的机制与待检验解释

### 3.1 对方向翻转缺少有效即时约束（最高优先级）

Sign-flip的cumulative q首次降低在round23，攻击自round11开始；162条恶意客户端记录
只有86条q<1，平均cumulative q=.9136。方向q全程为1，quarantine命中为0。
当前calibration中未启用独立direction配置，不能把“存在direction.py”解释为已实施方向防御。

round21--30，RTC平均ACC=70.477%，MK=81.508%，差11.031 pp；同期RTC恶意权重
34.5808%，MK为6%。两者攻击前round1--10平均ACC差仅-0.721 pp。
这支持持续接纳反向更新是重要解释；由于训练轨迹自round1分叉，仍需同模型反事实聚合实验
确认各机制的因果贡献，不能将全部6.68 pp差距直接分摊给一个指标。

### 3.2 裁剪保护数值，却使后续幅度检测丢失部分信息

Gaussian恶意raw norm约334.7，经裁剪后约13.95/14.31；良性约12.27/12.54。
后续anchor、residual、cumulative从clipped delta计算。明显的原始范数异常没有独立进入
即时client cap。恶意客户端100%被裁剪，但仍保留约17.84%聚合权重。
建议把“用于限制上传影响的裁剪”和“用于判断可信度的观测”分离，保留裁剪前异常证据。

Random v1恶意raw norm约14.0/14.3，与良性约12.4/12.7接近；它说明只补原始范数检测
仍不足以防御范数匹配攻击，需要方向和跨轮证据。

### 3.3 q惩罚偏软，QP倾向保留名义权重

当前cumulative q_min=.5；只靠该cap，最强累计惩罚也只是将允许上限降至nominal的一半。
这是cap的下限，不是最终权重的下限；其他预算仍可把最终权重压得更低。
QP目标包含接近nominal的二次项和风险代价；未被识别且约束不活跃时会保留较大权重。
日志中良性权重基本保持每条.1，Gaussian恶意平均cap约.636/.661；Random约.844/.857。

q²可以将cap下限从.5变为.25，但不会改变q=1期间的检测延迟。
不应同时修改q_min、指数、累计阈值；先定位延迟和残留分别贡献多少。

### 3.4 全局范数误伤与质量缺失是次要但真实的问题

Sign-flip良性clipping为9.17%，恶意为70.99%，良性cumulative cap命中为0。
因此不能用“降低semantic floor”修复当前主要问题；它可能重新引入已证实的语义误伤。
RTC在三种攻击下zero mass约2.2%--4.7%，但这不是更新范数损失或ACC损失的百分比。
无条件归一化剩余权重会再次放大被约束的恶意更新，也可能破坏hard budget。

### 3.5 现有impact只描述幅度，缺少符号与anchor来源

当前impact_norm = aggregation_weight * clipped_delta_norm，汇总时取该非负量的份额。
它不表示净攻击向量，也不区分抵消与同向累积；回填anchor的贡献亦未纳入客户端归因。
Gaussian中RFA的恶意impact约27.90%，高于RTC约21.70%，但RFA ACC更高，直接表明
“恶意impact越低则ACC越高”不是可靠推论。

新增观测应包括：trainable-only分层角度、对冻结参考的有符号投影、负向能量、
客户端项与anchor项各自的范数/方向、总聚合向量的方向。真实恶意身份只用于离线评估。
原日志未保存充分向量信息；上述方向量不能从范数CSV凭空恢复。

## 4. 顺序研究方案

### R0：固定实验合同并补最小方向诊断（首先完成）

1. 旧manifest与结果保持不可变；新源码使用新freeze和独立输出目录。区分随机噪声v1/v2，
   Sign-flip旧/新run_id，记录trainable参数与BN等非训练buffer的处理合同。
2. 给正式分析流程增加明确的partial审计入口；run_id必须在附加`_batch_root`等分析元数据前
   计算。当前正式分析器load_protocol附加该字段、_load_run再计算ID，存在ID污染风险；
   本次独立审计避免此路径，未修改正在使用的正式分析器。
3. 增加即时信号日志，先观察再设阈值：raw/clip ratio、对可靠参考的逐层cosine与投影、
   cap触发来源、首次恶意抑制、良性误伤、anchor方向及budget暴露。
4. 数据划分：现有seed42/46/47视为开发证据；校准使用独立clean数据/轮次，最终检验使用
   未参加阈值选择的新seeds。保留一个IID基准，再加入非IID压力检验。

### R1：单独增加即时方向client cap（第一个算法候选）

目标：Sign-flip在前几次参与时就降低危险方向的贡献，无需等累计范数证据积累。

- 首个可实施参考候选为同轮Multi-Krum筛选集合的平均更新，仅用于生成方向参考；
  其余RTC优化器、历史账本和accepted anchor不变。它是工程候选，并非证明安全的oracle。
- 对trainable-only更新计算带符号的方向一致性；参考范数过小、群体支持不足或方向分歧过大
  时弃权，不把低置信度参考当作硬拒绝证据。归一化仅用于打分，不无条件放大聚合向量。
- 用独立clean校准阈值，在足够置信度的负向/异常方向上加入q_direction_fast。
  `cap_i = nominal_i * min(existing_q_i, q_direction_fast_i)`；本候选不改回填比例和累计参数。
- 每个客户端能影响参考，存在参考污染/共谋/非IID误伤风险；记录参考集合、可信支持数，
  后续检查leave-one-out敏感性。不要把“离开主簇”直接等同于恶意。
- 若此参考不能可靠分离恶意与良性，停止该候选；再独立评估几何中位数参考，不能同时叠加。

最小执行设计：冻结新合同后seed42的Sign-flip运行RTC基线/候选/MK三格，clean运行RTC
基线/候选两格，共5格（在没有可严格复用基线时）。若观测已证明clean cap完全不触发，
仍保留首次clean闭环验证，避免忽略数值或参考路径改变。

建议预注册的工程筛选门：Sign-flip active ACC提高至少1 pp、final不下降；良性新cap
触发率不超过1%；恶意权重下降且负向投影下降；clean平均和final ACC退化均不超过0.2 pp；
所有预算、有限性、完成性和配对门通过。这些是待冻结的筛选建议，不是既有实验验收结论。

### R2/I12：幅度机制归因与方向主线集成桥接

保持用于数值稳定的MAD k=2.5。使用raw_norm/robust_benign_scale或raw/clip ratio的
高置信度异常作为单独q_raw，不让超大噪声被裁剪后恢复到接近普通可信度。
阈值来自独立clean分布；不能把所有clipped客户端拒绝。先用Gaussian+clean最小配对，
再用Random v2检验是否错误地把“范数匹配”当作安全。

单项B0+N结果只用于归因，不能代替组合验收。当前D=R1c已通过Sign-flip/clean阶段门，
N=R2未通过原单seed效用门但保留研究价值；预登记B0、B0+D、B0+N、B0+D+N四组，
直接比较组合相对B0+D的增量，并与B0比较总效果。至少覆盖Gaussian/clean/Sign-flip，
补LIE .5主线转移，进入targeted研究前补DBA/scaling回归。允许拆批人工执行。
原R2失败判定不改变；未通过新桥接门的组合只能是研究分支，不能称为已接受主线。
若N未被接受，后续仍继承B0+D，并在扩展条件前补其转移证据。

### R3：改进跨轮慢攻击证据

直接父版本为已接受且具备对应条件转移证据的M_k；新候选=M_k+一个累计改动。
旧B0上的诊断用于提出假设，不代表M_k闭环中的检测延迟；必要时补主线观测。
重放当前累计状态，区分参与次数不足、裁剪后z分离不足、threshold触发迟和q_min残留。
只有需要增强已经触发的惩罚时，单独试q power=2；需要更早发现时，独立研究持续有符号
偏移的累计量，并加入正常行为下的恢复/衰减机制。不得同时调多个参数。
开发筛选使用LIE z=.5；转移必须覆盖LIE .25、Min-Max、Min-Sum和clean/非IID。
同时验收clean、Sign-flip、Gaussian和已验证targeted条件的回归；不能预先宣称候选能防住。

### R4：DBA隔离与anchor安全，分两个候选

两个候选依次继承当时已接受的集成主线。既有cap改变accepted weights和anchor来源，
因此必须在实际组合上验证；失败只撤回本次改动，不重置此前有效机制。
- 第一个候选保持现有quarantine阈值，直接在solver设置高置信隔离client cap=0。
  上轮V1中458条DBA恶意记录全部quarantined但107条仍有正权重，是明确诊断入口。
- 仅在前项结果明确后，另行给实际回填anchor添加方向/semantic exposure及跨轮ledger，
  使用独立可靠参考，不能用anchor自身作为零residual的证明。保留空来源不回退的规则。
- DBA/scaling主终点为ASR及峰值/尾部表现；与RTC基线和MK同时比较，不用ACC收益抵消ASR退化。
- 不恢复已失败的nominal full recycle、accepted full recycle，不围绕.51做未注册细密搜索。

### R5：安全质量回收与泛化验收

只有即时筛选可信后，才研究缺失质量如何分配。若调整目标权重或回填，必须重新求解/验证所有
client、principal、server和跨轮预算；不能在solver后直接将client weights归一化到1。
使用相同攻击合同比较RTC基线、最终候选、MK；Gaussian可加入表现更好的RFA作为挑战者。
两条改动分别通过不意味着组合一定通过；应在每次接入主线时完成闭环增量与回归验收，
不能等到R5首次组合。R5主要验证已经集成的固定版本及关键消融；若还调整质量回收，
它仍是相对当前主线的新候选，必须先通过完整适用回归门。

最终条件覆盖clean、Gaussian、Random v2、Sign-flip、LIE .25/.5、Min-Max、Min-Sum、
label reversal、DBA、scaling。按攻击分别报告ACC、targeted ASR、恶意/良性权重、
有符号影响、zero/anchor mass、误伤、检测延迟和聚合耗时；不得仅给宏平均。
新seed数量按主要终点及所需置信区间设计；n=2或n=1只作工程筛选。

## 5. 比较解释的两条重要限制

每轮参与10人，配置Krum f=3，但随机抽样并不保证每轮恶意人数<=3。Sign-flip seed42中
20/50个活跃轮超过3，最多5人；Gaussian为11/50与12/50，Random为17/50与18/50。
这不破坏同计划的经验比较，但不能直接援引Krum理论保证，也不能将当前MK参考视为可靠oracle。
建议另设每轮满足攻击者上界的受控协议与原随机协议分开报告，不能事后改变原采样计划。

当前RTC每轮聚合约6--7秒，MK约1.1--1.2秒；这些是存档运行时间，不是严格硬件性能基准。
新增参考计算应记录耗时，先避免多次全维QP与重复距离矩阵。性能优化放在鲁棒性机制通过后。

## 6. 参考与本地代码定位

- `defenses/rtc/v3.py`：clipping→nominal anchor→clipped residual；cumulative client cap；accepted recycle预算。
- `defenses/rtc/cumulative.py`：单边累计统计与q映射。
- `defenses/rtc/solver.py`：保留nominal质量的二次目标。
- `config/rtc_v3_manifest_formal_iid_semantic.json`：当前full-resolution累计配置，q_min=.5。
- `strategies/fed_strategy.py`：客户端impact为weight×clipped norm。
- Krum原论文：https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
- RFA原论文：https://arxiv.org/abs/1912.13445
- FLTrust原论文：https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/
  该方法使用服务器根数据引导方向信任；本方案没有默认假设存在额外可信根数据。

2026-09-14下一动作：完整验收已完成的R3-observe三项B0诊断产物，再准备I12/主线转移桥接。
CPU/GPU真实训练继续全部由人工执行。本文件不替代历史实验结论，也不自动开始新的目标模式。

以下为历史阶段记录，涉及“下一步”“不进入组合”等表述只代表当时决定；
当前研究顺序及保留策略以上述修订和文末集成记录为准。

## 2026-09-12 / 按失败证据扩展R1参考设计

R0已拒绝Multi-Krum参考，R0b也因Sign-flip良性误标153/338违反≤1%门槛而拒绝几何LOO参考。
两者的拒绝保持有效。随后独立登记一次留一法谱多数参考，只用clean seed101校准阈值。
离线筛选标记127/162攻击记录、0/338良性记录，clean为5/600；这些只支持继续设计R1，尚未接受实际cap。
R1首次闭环固定q_fast=0的单一方向cap，不调整原累计/裁剪/回填机制；仍执行本计划的5格配对设计及验收门。
细节、弃权条件、已知限制和证据见`docs/RTC_R0B_REVIEW_R1_DESIGN_20260912.md`。
R2--R5依赖与最终目标不变；R1未通过前不推进后续阶段。

## 2026-09-12 / R1闭环验收后的修正

R1的5项seed42运行完成，21/21质量门和严格配对通过。Sign-flip平均ACC75.9582%→82.2524%，
但良性新cap10/338超过1%门，因此拒绝该版本，保留方向cap的研究方向。用户明确支持沿此方向修正。
独立复核定位误伤于后期弱负向良性更新，参考未受攻击者污染。登记的R1b分散度门离线筛选失败，未训练。
另行登记R1c逐参考客户端确认门，只用clean seed101确定负向噪声阈值，要求至少三分之二参考支持；
离线R1候选轨迹误标降至1/338、攻击标记122/162，其他已完成轨迹筛选亦通过。
下一批冻结后以新seed43做5格严格配对，不降低原R1验收标准，仍不推进R2。
完整证据、参数及人工交接见docs/RTC_R1_REVIEW_R1C_HANDOFF_20260912.md。


## 2026-09-13 / R1c通过与R2独立幅度候选

R1c seed43的5项人工实验完成，21项质量门、原10项验收门及独立复核均通过。
Sign-flip平均ACC78.5206%→83.4812%，同seed MK81.0842%；候选良性标记0/356，clean0/600且轨迹不变。
接受R1c为单seed工程候选，原R1/R1b拒绝有效，未宣称最终泛化。
按顺序准备R2：B3R-F0.51上单独raw trainable norm/LOO median>3置零；R1c方向仅观测。
6项seed42 Gaussian+clean配对，新增RFA挑战者。代码/合成测试/dry-run/合同已冻结，等待人工执行。
完整验收与命令见docs/RTC_R1C_REVIEW_R2_HANDOFF_20260913.md；R2通过后才另行准备Random-v2转移。


## 2026-09-14 / R2拒绝，进入独立R3观测

R2六项完成、22项质量门与独立核验通过；Gaussian平均ACC83.9470%→84.1034%，增益.1564pp未达.2pp门，故拒绝。
135/135攻击更新置零、良性0/365且clean0/600，不能以机制成功替代效用验收；不放宽旧门、不携带R2失败候选进入组合。
该候选的Random-v2转移通过条件未满足；最终Random-v2验证仍保留。继续独立R3，未将R2记为通过。
12项旧日志7200条累计q完整重放；LIE .5首触发round31--34，先补方向观测而非直接改q指数。
冻结seed42 LIE RTC/MK+clean RTC三项只读观测，等待人工执行。详见docs/RTC_R2_REVIEW_R3_OBSERVATION_HANDOFF_20260914.md。

## 2026-09-14 / 用户反馈后的R2保留结论修正

R2改记为“未通过本次预登记效用验收，但保留为有前景的安全机制候选，待进一步验证”。
此前从单seed效用门失败直接推导结束R2后续验证，过于绝对；不再将其永久排除。
原协议、+0.2pp门、decision/review中的该批次拒绝结论及源码归档不变。
恢复固定多seed复核为计划待办；在新结果之前预登记完整seed集合、终点和停止规则，不试到通过为止。
若后续采用安全收益+效用非劣的组件定位，须明确登记为新研究问题，不追改旧结果。
详情见docs/RTC_R2_RETENTION_REVIEW_20260914.md。R3观测合同保持可执行；本次没有启动训练或新增训练批次。

## 2026-09-14 / 阶段继承与集成时机修正（当前规则）

主线从M1=B0+R1c开始，接受范围目前限于已核验的Sign-flip/clean。R2保留为未接受研究模块，
可在新预登记I12分支验证与R1c的互补，不必先把旧单项失败改写为通过；进入研究组合不等于晋升主线。
桥接完成后，R3及R4均以当时已接受版本为直接父版本；每阶段兼顾新增问题与既有攻击回归。
R5交付单一实际集成版本的泛化、总效果及消融证据，不以多个孤立模块通过代替端到端验收。
R3-observe三项status已completed/exit0/round60，完整质量与配对验收待做；现有脚本与冻结产物不改。
具体版本表、桥接矩阵和执行边界见docs/RTC_STAGE_INTEGRATION_REVIEW_20260914.md。

## 2026-09-14 / R3独立验收与I12人工交接

R3观测3项与20项质量门通过，独立向量参考/原始日志/累计状态复核一致；LIE方向与raw攻击标记均0/143，
累计首次下降round34，约63.7270%恶意权重发生于q仍为1。观测有效不等于候选通过。
已准备I12四组桥接，固定seeds44/45，clean/Sign-flip/Gaussian/LIE .5及适用MK/RFA共40项，
相对R1c主线只再开启固定raw cap。Windows最终目录logs/rtc_i12_bridge_v2，默认dry-run；Linux脚本同步提供。
进入WAITING_FOR_MANUAL_EXPERIMENT，未启动训练；Linux实际dry-run尚待核验。
最新交接、版本关系、原失败保留及预登记门见docs/RTC_R3_REVIEW_I12_HANDOFF_20260914.md。

## 2026-09-15 / I12接受与M2跨轮残差观测

Linux服务器40项完整，104项质量门、72项组合门、26项幅度门与213份源码来源全部验收。
接受M2=B0+R1c+R2在IID/seeds44、45的四条件范围；clean/Sign-flip保留，Gaussian恶意权重归零。
LIE .5无新增收益、平均ACC仍比MK低2.3226pp；累计首次惩罚round33/30，约62.61%/58.47%恶意权重在q=1期间。
早期幅度z与良性重合，现有标量/同轮Gram不能还原跨轮残差方向。因此新批次只导出trainable残差固定512维sketch，
配对验证观测不改变M2行为；clean103/104两组及LIE44/45三组共10项，新cap/阈值均未启用。
源码随日志归档，双平台入口已准备；真实实验继续由用户手动执行，Linux先完成实机dry-run。
完整验收与下一边界见docs/RTC_I12_REVIEW_R3_TEMPORAL_HANDOFF_20260915.md。R4/R5与最终覆盖范围保持不变。

## 2026-09-15 / R3观测通过，转向持续小残差证据

服务器10项观测及42项质量门通过，观测版与M2训练/权重轨迹完全一致，220份源码归档核验完整。
高方向持续性未区分LIE；部分恶意trainable残差为零，无可定义方向。其相对peer中位数的小残差是新的开发线索。
先固定clean-only规则，再以clean103最小ratio确定阈值0.8738694595117251，连续两次低尾后q=0；
clean104独立验证0/600，旧LIE44/45离线标记163/179、133/150且良性均0，仅作诊断，不声称闭环收益。
已准备M2+该单一cap与M2配对，seeds201/202、clean/Sign/Gaussian/LIE .5及MK/RFA共24项人工实验。
要求LIE平均至少+1pp、最终不下降、恶意权重≤父版本80%；原条件效用非劣≤.2pp、联合良性误标≤1%及全部质量门。
小残差不是一般恶意判据，合法中心更新/非IID仍是风险；通过此批也不替代剩余攻击和最终泛化。
详细交接docs/RTC_R3_TEMPORAL_REVIEW_LOWER_TAIL_HANDOFF_20260915.md，当前WAITING_FOR_MANUAL_EXPERIMENT。


## 2026-09-15 / 低尾闭环收益与参考污染修正（最新）

24项Linux实验、92项runner门与完整228份来源通过验收；原38个候选门3项失败，保留拒绝。
LIE平均ACC两seed+5.1524pp、最终+.45pp，支持保留低尾机制；Gaussian seed201新增良性误伤8/354超1%。
父版本M2在Sign-flip seed201也有4/351误伤，主要在6/10攻击者控制谱参考时；历史I12接受不代表该seed安全转移。
Gaussian极端轮同样暴露raw留一中位数及低尾参考污染；不能靠加大std或删掉压力轮解决。
先收集M2的本轮谱参考与前轮实际trainable聚合方向的精确关系，现有同轮Gram无法恢复这一信息。
准备5项seed201 clean/Sign-flip严格配对，只读观测、未启用新cap，原抽样计划保持。
观测是必要诊断，不作为算法成功或新seed泛化；之后单机制修复父版本，再修复保留G并验证组合。
详细报告及人工边界docs/RTC_LOWER_TAIL_REVIEW_REFERENCE_HISTORY_HANDOFF_20260915.md。

2026-09-15用户确认保留：该组合以M2-G1独立登记在analysis/rtc_retained_candidates/M2-G1。
其全部ACC门通过，LIE收益和误伤失败分别记录；完整配置、源码与审计证据已归档。
后续不能因原批次rejected直接弃用：须在修复父版本后回到G的修正/集成，或用明确比较证据说明替代。
每个阶段交接和最终报告都需交代其收益去向，防止成功经验因阶段切换而遗失。


## 2026-09-16 / 参考历史观测验收与H1确认候选

5项Linux观测及21项质量门通过，234份源码完整核验，模型/权重轨迹与父版一致。
Sign-flip seed201的3条多数攻击参考误伤历史cosine约-.0278，另一条弱反向误伤为+.03685。
固定同半空间确认在旧轨迹可使良性标记4→1，但攻击标记115→79，说明必须检验防御损失，不能只看误伤改善。
新登记M2-H1，只为原R1c拒绝增加有效历史cosine>0确认；不翻转模型、不撤销其他cap、不搜索阈值。
24项seeds201/203 clean/Sign/Gaussian/LIE严格配对，候选误伤≤1%、效用退化≤.2pp、恶意权重增加≤.1pp；
201 Sign误伤还须严格减少。父版已知失败保留，原24项结论不改写；这是单独的修复协议。
M2-G1保持归档待修正并重新集成，H1不启用G1，后续必须交代其收益去向。
详细人工交接docs/RTC_REFERENCE_HISTORY_REVIEW_H1_HANDOFF_20260916.md；当前WAITING_FOR_MANUAL_EXPERIMENT。

## 2026-09-16 / H1闭环失败与H2主体资格修复

H1的24项服务器实验完整，92项质量门通过；31项候选门8失败。Sign-flip两个seed平均ACC下降1.2860/.7198pp，
恶意权重上升，201误伤仍4/351；故拒绝H1，不继承历史同向确认。其他三条件父子轨迹不变。
旧轨迹的历史cosine判别没有保持闭环收益：从31/32轮起撤销正确攻击拒绝，污染参考到52轮变为历史正向。
改为检验参考主体本身的上次原R1c/R2标记：至少三分之二参考主体未标记才允许当前R1c cap。
固定H2规则在旧M2轨迹误伤4→1且保留115/121全部攻击标记，不能据此声称闭环成功。
已准备201/204、clean/Sign/Gaussian/LIE24项严格配对人工实验，保留原误伤/效用/恶意权重门。
M2-G1归档哈希完整，仍待低尾参考修复并在修复后的父版上重新集成；未默认开启或静默丢弃。
最新交接docs/RTC_H1_REVIEW_H2_HANDOFF_20260916.md，当前WAITING_FOR_MANUAL_EXPERIMENT；真实训练全部手动。

## 2026-09-17 / H2保留与确认历史写入修复

指定logs/rtc_reference_eligibility返回24项batch96/GPU.1的Linux结果；质量、严格配对及250份源码通过。
31门29通过2失败：Sign201误伤4→0，但平均ACC-.2268pp及恶意权重+.2509pp违反原门，H2不晋升，独立保留待修复。
round52已被资格门撤销的四条误标仍写入原判断历史，round53使三个良性参考失去资格并放行三条攻击更新。
H2b仅在资格弃权时保留原历史，R2明确拒绝优先；旧轨迹可保留误伤修复并恢复三条攻击拒绝，但不证明闭环收益。
已准备M2/H2/H2b及挑战者32项、seeds201/205、batch96/GPU.1，要求同时对M2与H2效用/恶意权重非劣及误伤收益保留。
M2-G1持续保留，后续必须返回其参考修复与组合验收；batch96需重新检验旧48下校准与收益，不混算配对。
最新交接docs/RTC_H2_REVIEW_H2B_HANDOFF_20260917.md，当前WAITING_FOR_MANUAL_EXPERIMENT；未启动训练。


## 2026-09-18 最新衔接（优先于旧H2b等待记录）

H2b服务器32项完整，98质量门/62候选门及独立重放通过，接受为限定范围M3；
Sign201误伤4→0、对M2平均ACC+.1598pp/最终+.86pp，原H2历史误写入导致的三条恶意放行修复。
实际batch96/GPU.125，资源偏差已登记；其余七个条件/seed配对轨迹不变，不宣称非IID/targeted泛化。
接受记录config/rtc_reference_memory_accepted.json；旧M2和拒绝H2记录保持。
已回到M2-G1修复：G2只排除R2明确拒绝参考，保留9个其他主体/两次参与/无效重置，不改anchor。
旧轨迹Gaussian201低尾误标8→0且LIE攻击标记保持，只用于诊断，不预测闭环收益。
下一批4项clean106校准/107留出，各M3与只读观测两组，batch96/GPU.125；阈值仅来自106，107失败不调参。
准备、合成测试、Windows dry-run通过，Linux实机dry-run待返回，状态WAITING_FOR_MANUAL_EXPERIMENT。
详细交接docs/RTC_H2B_REVIEW_G2_CALIBRATION_HANDOFF_20260918.md。
校准后必须做G1/G2重新集成和LIE/既有攻击回归；不跳过R3其他攻击、R4/R5及最终targeted/非IID范围。
Git按用户后续“手动提交”指令，由用户执行，优先于本文早期自动同步条款。
