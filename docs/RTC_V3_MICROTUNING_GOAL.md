# RTC-v3 依序微调目标与状态记录

更新时间：2026-09-04

本文件是 RTC-v3 微调目标模式的持续记录。每次人工恢复目标模式后，必须先读取
本文件和当前实验日志，再更新“当前状态”；不得仅依赖对话摘要或记忆。

## 可直接用于目标模式的提示词

> 目标：基于 `D:\workspace\FL2` 中已经完成的强攻击、LIE z=0.5 和 RTC
> 机制日志，依序微调 RTC-v3，在不降低安全性的前提下提高 ACC、缩小相对
> Multi-Krum 的差距。权威计划与进度记录为
> `D:\workspace\FL2\docs\RTC_V3_MICROTUNING_GOAL.md`；每次恢复目标模式时先
> 读取该文件、实验 manifest/status/rounds/quality_gates 和当前代码，再继续，
> 并在每次阶段结论后更新该文件。
>
> 工作原则：
>
> 1. 严格按 B0→B1→B1R→B2→B3→B4→B5→V1 顺序推进；B1 是已完成的机制
>    诊断，不得把 `semantic_ablation=observe` 当成正式默认方案。当前阶段以
>    文件“当前状态”为准。
> 2. 每次只改变一个机制或一个参数，先做 seed42 最小筛选；不得同时叠加多个
>    未验证改动。只有当前阶段满足验收门槛后，才冻结该改动并进入下一阶段。
> 3. 实验必须严格配对。优先复用已有基线；只有 seed、恶意比例、采样计划、
>    attack contract 和 trial-plan hash 完全相同时才能复用。目录名不作为配置
>    证据，以 manifest/matrix 为准。
> 4. LIE 主实验固定 `z=0.5`；`z=1.0` 强度过大，不作为调参或主结论依据。
>    恶意比例固定 0.3。单 seed 结果只用于机制筛选，不作为显著性结论。
> 5. 每个阶段先完成代码、单元/回归测试、dry-run matrix、质量门和结果分析脚本，
>    然后给出唯一的 PowerShell 执行命令、准确参数、预计训练单元数、可复用基线
>    和验收条件。实验规模应尽可能小。
> 6. 到达需要 GPU 训练、消融或其他人工实验的边界后，不得自行启动训练；正式
>    将目标模式设为 blocked，等待人工执行并恢复。恢复后先检查真实进程/status、
>    完整轮数、exit code 和 quality gates；实验未完成则继续阻塞，不得提前进入
>    下一阶段。
> 7. 每次分析同时报告 ACC、安全指标（targeted attack 优先看 ASR）、恶意/良性
>    聚合权重、impact、zero-update mass、误伤率和首次机制分叉。不得用更高 ACC
>    掩盖 ASR 恶化。
> 8. 若阶段失败，只做与失败原因对应的最小回退或单参数调整；记录拒绝原因，不得
>    跳过阶段或把失败候选带入后续组合。
> 9. 完成 V1 多 seed 验证、全套质量门和最终 RTC/Multi-Krum 对照报告之前，不得
>    宣布整个目标完成。

## 阶段计划

| 阶段 | 单一问题/改动 | 最小 seed42 实验 | 进入下一阶段的条件 |
|---|---|---|---|
| B0 | 固定 `rtc_full`、Multi-Krum 与攻击基线 | 复用现有强攻击日志 | 配置、trial plan、质量门可信 |
| B1 | `semantic_ablation=observe`，定位语义误伤 | LIE z=0.5；DBA/Gaussian 安全转移 | 仅用于诊断；已证明不能全局启用 |
| B1R | 逐级提高语义介入 risk floor；先试 0.1，失败后仅单独试 0.5 | 每个候选 LIE z=0.5 + DBA，共 2 单元 | LIE active ACC ≥B0+0.3 pp；DBA ASR≤10%、恶意权重≤5%、active ACC 降幅≤0.2 pp |
| B2 | cumulative q 直接接入客户端 cap；先试 `q`，失败后才单独试 `q²` | LIE + DBA 安全护栏，优先 2 单元并复用基线 | LIE 恶意权重较 B1R 明显下降，ACC 不回退；DBA 安全门保持 |
| B3 | 将被拒绝质量回填到 coordinate-median anchor | 优先 DBA + scaling，共 2 单元 | zero-update mass 至少减半或≤2%；ACC 提升；ASR/恶意 impact 不恶化 |
| B4 | residual rank-cap：每轮 residual norm 前 2 名 cap×0.5，并将约 10% 质量交给 anchor | LIE + DBA，共 2 单元 | LIE 恶意权重趋近≤15.5%，良性误伤和 DBA ASR 均过门 |
| B5 | clipping MAD 系数最后调参：2.5→2.25；只有必要时再单独试 2.0 | 每个系数独立，LIE + clean/安全护栏 | 良性 clipping rate<8%，ACC 改善且安全指标不退化 |
| V1 | 冻结最终候选后做多 seed 正式验证 | 用户指定缩减为 seeds 42/46/47；clean、LIE z=0.25/0.5、DBA；需要时补其他强攻击 | 完整质量门、置信区间/配对统计和 RTC vs Multi-Krum 报告均完成；明确3-seed统计局限 |

## 已验证机制结论

- 当前正式微调链（B1R及以后）恶意比例固定为0.3；但历史目录
  `logs/rtc_v3_lie_z05_seed42_mf03`存在命名错误，其manifest、run_id和Krum参数均
  明确为恶意比例0.2。该目录只能作为mf=0.2历史参考，不能与mf=0.3结果直接比较。
- LIE 主实验强度为 z=0.5。
- B1/LIE：observe 相对 full 的 final ACC +2.15 pp、active ACC +0.4792 pp；
  首次分叉是 round26 的良性 cid18，risk=0.059<watch 0.1，但 full 仍用 q cap
  将权重 0.1→0.0941；当轮恶意客户端 semantic risk 为 0。
- B1/DBA：full 的 active ASR=2.2536%、恶意权重=0.1083%；observe 的
  active ASR=97.0831%、恶意权重=28.5038%。因此 observe 不能作为默认方案。
- B1/Gaussian：恶意 semantic risk≈0.012，observe 对 active ACC 只改变
  +0.0176 pp，说明该攻击的主要问题不是良性语义误伤。
- DBA 质量门曾错误要求同轮 fragment 唯一；现已按固定恶意身份的 rank mod 4
  映射验证，13/13 质量门通过且无需重跑训练。

## 当前状态

- 当前阶段：B5 `norm_clip_mad_k=2.25` 已完成并因效用/clean clipping门失败而拒绝；
  不再尝试更激进的2.0。最终冻结候选仍为B1R floor=0.50 + B2线性
  cumulative-q cap + B3R accepted-anchor recycle fraction=0.51，默认MAD k=2.5。
  当前已进入V1多seed正式验证的人工训练边界。
- B1R 完成性：LIE 与 DBA 均 completed、exit code=0、last round=60、各 61 行；
  11/11 质量门通过，trial-plan hash 与 RTC full 基线严格一致。
- B1R 判定：LIE active ACC +0.4792 pp、final ACC +2.15 pp、zero mass
  4.5245%→0.0225%；DBA active ACC -0.0008 pp、active ASR=2.3753%、
  恶意权重=0.2540%。全部预设效用与安全门槛通过。
- B2 唯一变化是在冻结的 `semantic_intervention_risk_floor=0.50` 上增加
  `cumulative_q_cap_power=1`；两个候选均 completed、exit code=0、last round=60、
  各61行，11/11质量门通过。两个 trial-plan hash 与 B1R 完全一致：LIE
  `de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`；DBA
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`。
- B2 判定：LIE active ACC 77.5966%→78.3058%（+0.7092 pp），final ACC
  84.07%→86.38%；恶意权重28.5775%→25.3088%（-3.2687 pp），恶意 impact
  30.2719%→27.7523%，良性权重保持71.4%，zero mass 0.0225%→3.2912%。
  DBA active ACC/ASR、恶意权重与impact均与B1R完全一致；全部预设门槛通过。
- B2 cap 审计：两种攻击各500个活跃期客户端行均无权重上限越界。LIE共有61个
  cap-active客户端行，全部为恶意，良性命中0；首次分叉round34恶意cid7，
  cumulative q=0.991141，权重0.1→0.099114。
- B3唯一变化：`rtc_cumulative_q_cap_anchor`在B2的floor=0.5与linear q cap上
  只增加`anchor_recycle_fraction=1.0`；底层defense type仍为`rtc_full`，不加入
  q²、residual rank-cap或clipping。
- B3全量测试413项通过，`git diff --check`无错误；dry-run matrix恰好4单元：
  strong DBA与strong scaling backdoor分别运行B2 baseline和B3 candidate。
  之所以不是2个候选单元，是因为scaling没有现成B2严格配对基线；统一runner也
  不能按攻击选择不同defense。4单元是单目录、单命令且无需新增runner机制的最简
  严格配对设计。
- B3攻击参数：seed42、恶意比例0.3、IID、60 rounds、attack start=11、20 clients、
  participation=0.5；DBA与scaling均`replacement_gain=1.0`、`poison_fraction=0.3`、
  aggregation-aware scaling；DBA另启用冻结的paper trigger与update scaling。
- B3配对hash：DBA trial-plan
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`；scaling
  trial-plan `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`。
- B3预设验收门：对DBA与scaling分别要求active zero mass至少减半或≤2%；
  active ACC不低于B2；active ASR、恶意impact和恶意权重增幅均≤0.5 pp；
  anchor recycle实际生效；全部完成性、质量门和严格配对通过。
- B3完成性：4个单元均`state=completed`、`exit_code=0`、`last_round=60`，各61行；
  13/13 quality gates通过。DBA与scaling配对hash分别保持
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`与
  `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`。
- B3判定：**拒绝，不冻结，不进入B4**。DBA active ACC +0.5144 pp、final ACC
  +1.25 pp、zero mass 31.5460%→0，但active ASR 2.3753%→52.1673%
  （+49.7920 pp）；scaling active ACC +0.1730 pp、final ACC +0.98 pp、zero mass
  29.2135%→0，但active ASR 1.7264%→27.2500%（+25.5236 pp）。两项ASR增幅均
  远超预设+0.5 pp上限。
- B3归因：候选平均回填约32%的server-owned coordinate-median anchor；该anchor在
  solver前按nominal权重由全部正质量客户端构造，包含随后被quarantine的攻击更新，
  但回填暴露不归因到客户端或principal。因而DBA恶意客户端权重0.2540%→0.2477%、
  scaling恶意权重0.1958%→0.2237%看似安全，实际后门方向经anchor重新进入聚合。
  DBA/scaling的anchor均在round11首次生效，客户端权重在round12首次分叉；完整
  recycle从此持续改变模型轨迹。
- B3唯一人工执行命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_anchor_recycle_b3_ablation.ps1' -Execute`
- 恢复后的唯一分析命令：
  `& 'D:\workspace\FL2\.venv\Scripts\python.exe' 'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3\analyze_results.py' --experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3_seed42_mf03' --output-dir 'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3'`
- B3拒绝报告已固化为`analysis/rtc_anchor_recycle_b3/artifact.json`与
  `report_snapshot.sqlite`；artifact validator通过，并已在MCP report surface渲染。
- B3R唯一机制变化：保留原nominal anchor用于residual与客户端约束；缺失质量回填
  改为solver后按最终accepted weights重算coordinate median，并用该新向量重算server
  anchor/total hard-budget exposure。若accepted总质量为0，不回退到nominal anchor。
- B3R代码验证：新增accepted/nominal分离、accepted为空不回退、新anchor hard-budget
  重算及非法参数测试；全量`418 passed`，`git diff --check`无错误。
- B3R实验目录：`logs/rtc_v3_anchor_recycle_b3r_seed42_mf03`；恰好2个candidate-only
  单元（strong DBA、strong scaling backdoor）。DBA为有效cache复用，scaling在首次
  外部中断后用相同run id和trial-plan hash完整重跑；两者最终均60轮、exit code=0、
  各61行，11/11质量门通过。manifest SHA256为
  `f7de91c35d85f3a05143639983e3e91de4e95746230b8c7d642af572c755abe8`。
- B3R严格配对：DBA与scaling trial-plan hash分别为
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`与
  `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`，与现有
  B2/B3相同；attack contract也相同。候选参数保持floor=0.5、q power=1、recycle
  fraction=1.0，仅相对失败B3增加`anchor_recycle_weighting=accepted`。
- B3R唯一人工执行命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_anchor_recycle_b3r_ablation.ps1' -Execute`
- B3R恢复后的唯一分析命令：
  `& 'D:\workspace\FL2\.venv\Scripts\python.exe' 'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3r\analyze_results.py' --baseline-experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3_seed42_mf03' --candidate-experiment-dir 'D:\workspace\FL2\logs\rtc_v3_anchor_recycle_b3r_seed42_mf03' --output-dir 'D:\workspace\FL2\analysis\rtc_anchor_recycle_b3r'`
- B3R预设验收门与B3一致：每种攻击分别要求active zero mass至少减半或≤2%；
  active ACC不低于B2；active ASR、恶意impact和恶意权重增幅均≤0.5 pp；anchor
  recycle实际生效；运行时weighting必须为accepted；完成性、全部质量门和严格配对通过。
- B3R判定：**拒绝，不冻结，不进入B4**。DBA active ACC较B2提高1.1536 pp、
  final ACC提高1.19 pp、zero mass 31.5460%→0，但active ASR增加0.6893 pp，超过
  预注册0.5 pp上限；scaling active ACC提高1.0902 pp、final ACC提高0.67 pp、
  active ASR下降0.0384 pp、zero mass 29.2135%→0。两种攻击的恶意权重增幅分别
  0.0152/0.0276 pp、恶意impact增幅0.0492/0.0646 pp，均过门；良性watch增幅
  0.0635/0.9167 pp，restricted/quarantine均不变。anchor首次在round11激活，
  客户端权重首次在round12于恶意客户端分叉；DBA ASR在round19首次超过+0.5 pp门。
- B3R-F0.5唯一变化：在B3R accepted weighting上只把
  `anchor_recycle_fraction=1.0→0.5`，目标是回填恰好一半缺失质量，以保留ACC收益并
  把DBA ASR增幅压回0.5 pp以内；不改floor、q power、rank-cap或clipping。
- B3R-F0.5 dry-run目录：`logs/rtc_v3_anchor_recycle_b3r_f05_seed42_mf03`，恰好2个
  candidate-only单元；manifest SHA256为
  `eaadb61ce7b56b74487e9b76a7940d071e580c55a64ad8017d60d9b16701e4c7`，内嵌spec
  hash为`55c1f58dd5759813d970ba574606f7772906a484174c79110924e76621a6a18b`。
  DBA/scaling trial-plan hash仍分别为
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`与
  `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`；与B3R
  的attack contract、weighting和其余参数完全一致。
- B3R-F0.5唯一人工执行命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_anchor_recycle_b3r_f05_ablation.ps1' -Execute`
- B3R-F0.5恢复后的唯一分析命令见
  `analysis/rtc_anchor_recycle_b3r_f05/README.md`；分析器会同时核验fraction=0.5、
  accepted weighting、完成性、质量门及B2/B3/B3R严格配对。
- B3R-F0.5验收门保持不变：每种攻击active zero mass至少减半或≤2%；active ACC
  不低于B2；active ASR、恶意impact和恶意权重增幅均≤0.5 pp；anchor实际生效；
  两个单元完成且全部质量门通过。
- B3R-F0.5完成性：DBA与scaling均`state=completed`、`exit_code=0`、
  `last_round=60`，各61行且攻击活跃期各50行；11/11 quality gates通过。运行时
  fraction仅为0.5、weighting仅为accepted，两组trial-plan与attack-contract hash
  均与B2/B3/B3R严格一致。
- B3R-F0.5判定：**拒绝，不冻结，不进入B4**。DBA active ACC/final ACC较B2分别
  +0.6848/+0.67 pp，active ASR +0.2584 pp，恶意权重/impact分别+0.0056/
  +0.0211 pp，zero mass 31.5460%→15.7702%（ratio=0.499911），全部过门。
  Scaling active ACC/final ACC +0.6184/+0.34 pp，active ASR -0.0453 pp，恶意
  权重/impact +0.0168/+0.0358 pp，均过门；但zero mass 29.2135%→14.6232%，
  ratio=0.500562，较预注册≤0.5门超出约0.0164 pp绝对质量，故整体拒绝。
- B3R-F0.5机制证据：两种攻击anchor均在round11首次激活，客户端权重在round12
  首次分叉且只涉及恶意客户端；DBA在round56首次出现单轮ASR增幅>0.5 pp。良性
  watch增幅为DBA +0.0635 pp、scaling +0.5833 pp，restricted/quarantine均不变；
  source availability均为100%，anchor未被hard budget限缩。
- B3R-F0.51唯一变化：只把B3R-F0.5的`anchor_recycle_fraction=0.50→0.51`；
  accepted weighting、floor=0.5、linear q cap、攻击合同和trial plan均不变。该
  1个百分点增量用于给scaling减半门留下稳健余量，不同时修改rank-cap或clipping。
- B3R-F0.51 dry-run目录：`logs/rtc_v3_anchor_recycle_b3r_f051_seed42_mf03`，恰好
  2个candidate-only单元（strong DBA、strong scaling backdoor）；manifest文件
  SHA256为`4ca5a8e4415de4dbfb8389d53748e53911b98685d9b7b0ed8dd0154518f42af2`，
  内嵌spec hash为`cb2e359a678e2f4cb5855ff68a589835776ffea321464975416aa18beb50fbf6`。
  两组trial-plan hash仍分别为`3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`
  与`d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`。
- B3R-F0.51验证：Windows PowerShell 5.1完整dry-run exit code=0；全量测试
  `424 passed`；`git diff --check`无空白错误。执行脚本SHA256为
  `cb9d4f5e3459d2ca9483fa0d915148969c57674280ac3971126d4ed6c77e3754`；分析器
  SHA256仍为`ae1dd8b63de425de9aa46ae78984aed826adea821590060a4295a8e2aa7eae7f`。
  当前status/rounds均为0，quality gates与raw训练日志不存在，确认未启动训练。
- B3R-F0.51唯一人工执行命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_anchor_recycle_b3r_f051_ablation.ps1' -Execute`
- B3R-F0.51恢复后的唯一分析命令见
  `analysis/rtc_anchor_recycle_b3r_f051/README.md`；验收门与B3R-F0.5完全相同。
- B3R-F0.51完成性：DBA与scaling均`state=completed`、`exit_code=0`、
  `last_round=60`、各61行且攻击活跃期各50行；11/11 quality gates通过。运行时
  fraction仅为0.51、weighting仅为accepted，DBA/scaling trial-plan hash仍分别为
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`与
  `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`，与B2/B3/B3R严格一致。
- B3R-F0.51判定：**接受并冻结，进入B4准备**。DBA active/final ACC较B2
  +0.6756/+0.53 pp，active ASR +0.1958 pp，恶意权重/impact +0.0032/+0.0222 pp，
  zero mass 31.5460%→15.4560%（ratio=0.489950）；scaling active/final ACC
  +0.6282/+0.49 pp，active ASR -0.0393 pp，恶意权重/impact +0.0155/+0.0322 pp，
  zero mass 29.2135%→14.3447%（ratio=0.491028）。全部预设门通过。
- B3R-F0.51机制证据：两种攻击anchor均在round11首次激活，round12首次客户端
  权重分叉且均只涉及恶意客户端。DBA在round58首次出现单轮ASR增幅>0.5 pp，
  但预注册的活跃期平均ASR增幅为0.1958 pp并通过。良性watch率分别下降
  0.5079/0.2500 pp，restricted/quarantine均不变，source availability=100%，
  anchor未被hard budget限缩。
- B3R-F0.51两攻击宏平均：active mean ACC=84.4081%，final mean ACC=87.9150%；
  严格配对的B2基线分别为83.7562%/87.4050%，即+0.6519/+0.5100 pp。该口径
  仅覆盖DBA与scaling的seed42，不得解读为全攻击、多seed最终mean ACC。
- B4唯一机制：每轮按full-resolution residual norm降序取前2名（并列按
  server-side principal/client identity确定性打破），将其客户端上限设为
  `nominal * min(existing_q_cap, 0.5)`。原有缺失质量继续按0.51回填；
  rank-cap相对同轮无rank-cap reference QP实际新移除的质量按1.0回填到
  solver后accepted-weight coordinate-median anchor。无其他约束时正好移除并回塦10%质量，
  且不会把已因DBA ASR失败的存量full-recycle重新引入。
- B4代码与测试：新增`rtc_b4_residual_rank_cap`别名，固定top_k=2、factor=0.5、
  rank-recycle fraction=1.0；已覆盖参数非法值、确定性并列、client cap、reference QP、
  新移除质量回填、anchor hard budget与实验matrix单机制契约；全量`440 passed`，
  `git diff --check`无空白错误，分析器`py_compile`通过。
- B4最小实验恰好3个seed42单元：LIE z=0.5的B3R-F0.51基线1个（此前缺失）；
  LIE z=0.5与strong DBA的B4 candidate各1个。DBA B3R-F0.51基线从已完成目录
  严格复用，不重跑；因此比笛卡4格尔积更小，同时保持每种攻击与冻结基线严格配对。
- B4攻击和配对：均为CIFAR-10 IID、seed42、恶意比0.3、60轮、round11起持续攻击、
  20客户端且participation=0.5。LIE固定z=0.5、knowledge=all_updates；DBA使用
  冻结strong paper trigger、replacement gain=1、poison fraction=0.3与update scaling。LIE/DBA
  trial-plan hash分别为`de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`与
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`。
- B4 dry-run产物：LIE基线目录`logs/rtc_v3_residual_rank_cap_b4_lie_baseline_seed42_mf03`，
  manifest SHA256=`2a7897330049d5eb7af162a2138872ca85a11599aede8f37bd79dc52aeb53dff`；
  候选目录`logs/rtc_v3_residual_rank_cap_b4_seed42_mf03`，manifest SHA256=
  `fd0ea03b17171b5448fe2d3edb707822f9aad11094f0b9b49812d26498af85af`。两目录当前
  status/rounds/quality gates/raw训练产物均为0，确认dry-run未启动训练。
- B4预注册门：LIE恶意权重≤15.5%且低于基线；LIE/DBA active ACC不低于各自基线；
  DBA active ASR、恶意权重、恶意impact增幅均≤0.5 pp，LIE恶意impact增幅≤0.5 pp；
  两攻击zero mass增幅均≤0.5 pp，良性rank-cap命中率均≤10%，良性restricted/
  quarantine增幅均≤0.5 pp；每个活跃轮必须精确选中前2名且cap≤0.5，reference QP
  100%有效，rank recycle target=实际新移除质量，单轮新移除质量≤0.1；3个新单元
  均完成、全部质量门及严格配对通过。
- B4唯一人工执行命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_residual_rank_cap_b4_ablation.ps1' -Execute`
- B4恢复后唯一分析命令见`analysis/rtc_residual_rank_cap_b4/README.md`。执行脚本、分析器与
  RTC实现SHA256分别为`00bd1acff1b678ce30c96a73d8523d0e89aeaeadaaa5ba59bab431c0c677201a`、
  `32874feed6ad811c563590c3746be65a4346e7a3c58b972bc555387ca9fb8cb8`、
  `b4c236a79caf6695f87c98ec2bc619831dd2ffcc56d9e79e280886c53b170084`。
- B4完成性：LIE基线、LIE候选与DBA候选3个单元均`state=completed`、
  `exit_code=0`、`last_round=60`且各61行；LIE基线10/10、候选11/11质量门通过，
  两组trial-plan/attack-contract与冻结基线严格配对，运行时rank契约全部通过。
- B4判定：**拒绝并回退B3R-F0.51，进入B5准备**。LIE active ACC
  78.0796%→76.8336%（-1.2460 pp），恶意权重25.1071%→20.8158%，仍高于15.5%门；
  DBA active ACC 84.6014%→84.5716%（-0.0298 pp）。其余安全、质量和配对门通过。
- B4机制归因：LIE前2 residual命中100次中68次恶意、32次良性，良性命中率
  8.9636%；DBA为94次恶意、6次良性。rank-cap确实降低LIE恶意质量，但同时移除
  良性权重并转给anchor，不能保持优化效用，因此属于“降低恶意质量但仍错误损失
  良性质量”的粗粒度方案，不能冻结。
- 当前可接受最优mean ACC：冻结的B3R-F0.51在预注册DBA+scaling口径下active/final
  宏平均为84.4081%/87.9150%；补入同配置LIE z=0.5后，LIE+DBA+scaling三攻击
  seed42宏平均为82.2986%/87.2867%。两者均非多seed最终结论。
- B5唯一变化：在冻结B3R-F0.51上将RTC-v3 norm clipping的MAD系数从默认2.5
  改为2.25；不带入已拒绝的B4 residual rank-cap，不改floor、q power、accepted
  anchor weighting或recycle fraction=0.51。新增运行时指标`rtc_v3_norm_clip_mad_k`。
- B5最小实验恰好2个candidate-only单元：LIE z=0.5与strong DBA；LIE基线复用
  B4补齐的B3R-F0.51目录，DBA基线复用既有B3R-F0.51目录。每个严格配对运行的
  round1–10均未启用攻击，作为plan-matched clean clipping安全窗口，故不另跑clean。
- B5 dry-run目录：`logs/rtc_v3_clip_mad_b5_225_seed42_mf03`；manifest文件SHA256为
  `79e88dda7cb4f00a5ae4089def6dac84b227469cf671ac046deadc05ffaeeb0a`。
  LIE/DBA trial-plan hash仍分别为`de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`
  与`3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`；
  attack-contract hash也与基线一致。
- B5验证：参数默认值/非法值/阈值变化与matrix单参数契约均有测试；最终全量`452 passed`，
  `git diff --check`无空白错误。Windows PowerShell 5.1 dry-run重复两次exit code=0、
  manifest hash稳定；status/rounds/raw/quality gates均为0，确认未启动训练。执行脚本、
  分析器、RTC实现与freeze SHA256依次为
  `b391d3e7b4958acdd8a51fd4c28db072636eb389ad6dfbc5e7791eaee4896ebd`、
  `36b4e7dcf05c07032fee71d62403883c409c0b579101be168a097d1a444261f5`、
  `a88c3e4fcd5e1e1321580049af6c14dca8c9c6a7c10d4265798b049eb068b2f9`、
  `38e691511ddee8b927bec637576e22557bc3594032cd4f36fd4932567ac6df98`。
- B5预注册门：两攻击的preattack clean与active benign clipping rate均严格<8%；
  LIE active ACC严格提高且LIE/DBA均不低于基线；DBA active ASR、两攻击恶意权重/
  impact、zero mass与良性restricted/quarantine增幅均≤0.5 pp；运行时k恒为2.25，
  完成性、全部质量门和严格配对/单参数检查通过。任一失败即拒绝2.25并回退；2.0不预跑。
- B5完成性：LIE与DBA候选均`completed`、`exit_code=0`、`last_round=60`且各61行；
  两目录全部quality gates通过，trial-plan、attack implementation、单参数差异和运行时
  k=2.25检查通过。分析器已修复为只在fit轮1–60检查运行时参数，排除评估行round0。
- B5判定：**拒绝并回退B3R-F0.51，不运行2.0。** LIE active/final ACC为
  78.0796%/86.03%→78.0406%/86.03%（active -0.0390 pp）；恶意权重/impact
  下降0.2979/0.3299 pp。DBA active/final ACC为84.6014%/88.03%→
  84.3576%/88.02%（active -0.2438 pp），active ASR下降0.1533 pp，但恶意权重/
  impact小幅增加0.0122/0.0161 pp。DBA攻击前clean clipping仍为8%，不满足严格
  `<8%`。失败检查为两攻击active ACC不低于基线、LIE严格改善和DBA clean clipping。
- B5机制：LIE active malicious clipping提高4.1958 pp但未转化为ACC收益；DBA恶意
  客户端原本已100%裁剪，候选只增加active benign clipping 1.1730 pp。round1已经
  出现裁剪集合/范数分叉，故后续LIE平均良性裁剪率降低是轨迹变化的内生结果，不能
  解释为k=2.25更宽松。技术报告`analysis/rtc_clip_mad_b5_225/artifact.json`已通过
  validator并在MCP report surface渲染。
- V1最小验证矩阵：按用户最新决定缩减为seeds 42/46/47；显式clean、LIE z=0.25、
  LIE z=0.5、strong DBA；每个条件严格配对冻结B3R-F0.51与Multi-Krum。共24格，
  其中当前实现hash一致的seed42 RTC LIE z=0.5与DBA两格复用，人工只需训练22个
  新单元。旧
  Multi-Krum seed42攻击实现hash已过期，不能复用。clean固定`attack=none`但保留
  部署假设恶意比例0.3/`f=3`，Multi-Krum每轮10个参与者选5个。
- V1四个dry-run manifest SHA256依次为：clean
  `44dd1af59a71171570980027640c4c3722438a3bc3e30a57b15428f62efd9a80`；LIE z=0.25
  `049b063b4bfe22565d8134770a4784c0cd44892c5b792462a6cdf888b7862e98`；seed42
  Multi-Krum `76819b66e3a881a1c4ff15946a4cf8f56f59338a2f8afb079254085942e56f07`；其余
  seeds的LIE z=0.5/DBA `53fe8fedf3a4b2ffb03ca096594662944a3978c32f8cf69c2cedf6eeeae87c0f`。
  PowerShell 5.1重复dry-run通过，四目录status/rounds/gates均为0，确认未启动训练。
- V1脚本/分析器SHA256分别为
  `b9a173bfff9a28ead8add99540a081238e113c9d2cd513e836d87a42fff320c4`与
  `c431bacfca6c07f5f656a93aa894291edcdf4df71c976ef4b4649874c3c9a81c`。分析器要求
  24格完成、全部质量门通过、pair内trial-plan和attack-implementation hash一致，
  并输出三seed配对t检验与95%置信区间，以及全部权重、impact、mass、误伤和首次
  机制分叉指标。三seed检验仅有2个自由度，最终报告必须明确宽置信区间和探索性限制。
- 当前动作：旧五seed人工任务已按用户要求在第一格round15后停止，没有完成的V1格；
  旧目录已标记`UserInterrupted`且不复用。当前已重新到达三seed V1人工GPU训练边界；
  不得自行启动训练。唯一人工命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_v1_multiseed_validation.ps1' -Execute`
- 2026-09-04 第一次B4训练边界续行复核：重新完整读取本计划并核对两个manifest、
  训练产物、关键代码hash与一次性进程快照。LIE基线/候选manifest SHA256仍分别为
  `2a7897330049d5eb7af162a2138872ca85a11599aede8f37bd79dc52aeb53dff`与
  `fd0ea03b17171b5448fe2d3edb707822f9aad11094f0b9b49812d26498af85af`；执行脚本、
  分析器与RTC实现hash也均未漂移。两个目录仍分别只有3/4个dry-run文件，
  status/rounds/quality gates/raw训练产物全部为0；未发现Python或Ray训练进程。
  `nvidia-smi`仅列出桌面应用和少数权限不足的系统进程，不能证明B4训练正在运行。
  继续等待人工执行同一唯一命令，不自行启动训练、重复dry-run或提前进入B5。
- 2026-09-04 第二次B4训练边界续行复核：再次完整读取计划、两个manifest及B4当前
  实现，LIE仍为z=0.5、恶意比例仍为0.3，两个manifest与执行脚本、分析器、RTC实现
  SHA256全部未漂移。LIE基线目录仍只有3个dry-run文件，候选目录仍只有4个；
  status/rounds/quality gates/raw训练产物继续全部为0，且未发现Python或Ray训练
  进程。自B4准备完成并到达人工训练边界起，同一阻塞条件已连续三个目标回合成立，
  当前没有不越过“不得自行训练”授权边界即可继续的工作，目标模式正式blocked。
  仅在人工完成3个训练单元并恢复后继续验收，不得提前进入B5。
- 2026-09-03 第一次B3R-F0.51训练边界复核：manifest SHA256仍为
  `4ca5a8e4415de4dbfb8389d53748e53911b98685d9b7b0ed8dd0154518f42af2`；
  执行脚本、分析器与anchor实现SHA256仍分别为
  `cb9d4f5e3459d2ca9483fa0d915148969c57674280ac3971126d4ed6c77e3754`、
  `ae1dd8b63de425de9aa46ae78984aed826adea821590060a4295a8e2aa7eae7f`、
  `1aed752347757a0e69eb6165c86eae25cd2a3af5cdac8f0ad159aa72181039d8`。
  目录仍只有dry-run manifest、matrix与两份trial plan，status/rounds均为0，
  quality gates和raw训练日志不存在；继续等待人工执行唯一命令，不启动或持续轮询。
- 2026-09-03 第二次B3R-F0.51训练边界复核：manifest、执行脚本、分析器与anchor
  实现hash均与第一次复核一致；目录文件数仍为4，仅含dry-run manifest、matrix和
  两份trial plan。status/rounds/quality gates/raw日志继续不存在，没有可验收结果；
  保持同一人工训练边界，不自行启动实验或持续轮询。
- 2026-09-03 第三次B3R-F0.51训练边界复核：manifest与三份关键代码hash继续
  完全一致，目录仍只有同一4个dry-run文件；status/rounds/quality gates/raw日志
  计数连续第三次均为0。当前已无可在不越过人工训练授权前提下推进的工作，目标模式
  正式blocked；仅在人工完成两个实验单元并恢复后继续验收，不得提前进入B4。
- 2026-09-03 第一次B3R-F0.5训练边界复核：manifest SHA256仍为
  `eaadb61ce7b56b74487e9b76a7940d071e580c55a64ad8017d60d9b16701e4c7`，内嵌spec
  hash仍为`55c1f58dd5759813d970ba574606f7772906a484174c79110924e76621a6a18b`；
  两个candidate spec均保持fraction=0.5、weighting=accepted及原trial-plan hash。
  status/rounds均为0，quality gates、运行汇总与raw训练日志均不存在，没有已确认的
  运行句柄或可验收结果；继续等待人工执行唯一命令，不自行启动或持续观察。
- 2026-09-03 第二次B3R-F0.5训练边界复核：manifest、内嵌spec、两个trial-plan
  hash以及fraction=0.5/weighting=accepted均未变化；执行脚本SHA256为
  `7f85b0997426563c22e63b2e06b5c941fdc06bec52c16b0e8d8bef3716130072`，分析器
  SHA256为`ae1dd8b63de425de9aa46ae78984aed826adea821590060a4295a8e2aa7eae7f`。
  status/rounds仍为0，quality gates、运行汇总与raw训练日志仍不存在；训练边界
  未变化，继续等待人工执行唯一命令，不自行启动或持续轮询。
- 2026-09-03 第三次B3R-F0.5训练边界复核：重新读取计划、
  `experiment_manifest.json`、matrix、完整目录产物和anchor实现；manifest SHA256
  仍为`eaadb61ce7b56b74487e9b76a7940d071e580c55a64ad8017d60d9b16701e4c7`，
  两个spec仍为seed42、恶意比例0.3、fraction=0.5、weighting=accepted，并保持原
  trial-plan/attack-contract hash。`defenses/rtc/v3.py` SHA256为
  `1aed752347757a0e69eb6165c86eae25cd2a3af5cdac8f0ad159aa72181039d8`，执行脚本
  与分析器hash也与第二次复核一致。目录仍只有dry-run manifest、matrix和两份
  trial plan，status/rounds均为0，quality gates、运行汇总与raw训练日志不存在。
  相同人工训练阻塞已连续确认三次，目标模式正式blocked；不得启动实验或进入B4，
  仅在人工执行唯一命令并恢复后继续验收。
- 2026-09-03 首次人工执行B3R-F0.5时，在训练runner启动前因Windows PowerShell
  5.1不支持`ConvertFrom-Json -Depth`而终止；现场复核确认status/rounds均为0，
  quality gates和raw日志不存在，因此没有实验单元实际启动。已只移除该不兼容参数，
  未改变manifest、RTC机制或实验参数；使用Windows PowerShell 5.1执行不带
  `-Execute`的完整dry-run返回exit code 0。manifest SHA256仍为
  `eaadb61ce7b56b74487e9b76a7940d071e580c55a64ad8017d60d9b16701e4c7`，修复后
  执行脚本SHA256为`7eaee61b3c3fc43b93ea48211bf01f73f7c5f4a293ab440c00103073ff13b52f`；
  重新到达同一人工训练边界，仍使用原唯一命令运行两个candidate-only单元。
- 2026-09-03 第一次B3R训练边界续行复核：重新读取计划、manifest、matrix与当前
  anchor实现；manifest SHA256仍为
  `f7de91c35d85f3a05143639983e3e91de4e95746230b8c7d642af572c755abe8`，两个候选
  单元、floor=0.5、q power=1、recycle fraction=1.0、weighting=accepted及两组
  trial-plan hash均未变化。status/rounds/gates/raw log计数仍为0；进程命令行检查
  因当前Windows权限被拒绝，且没有其他可轮询运行句柄。训练尚无可验证进展，继续
  等待同一人工命令，不重复dry-run或改动候选。
- 2026-09-03 第二次B3R训练边界续行复核：manifest文件SHA256仍为
  `f7de91c35d85f3a05143639983e3e91de4e95746230b8c7d642af572c755abe8`，内嵌spec
  hash为`f0e4cc5a45c809d8af6fe2038d9a80c90b6aeb2b56463a53f66bfffdcd529677`；
  两个candidate spec、参数与trial-plan hash均未变化。status/rounds/gates、运行
  汇总和raw log仍全部不存在，没有可验证训练进展或可轮询句柄；继续等待同一命令，
  不修改候选或进入B4。
- 2026-09-02 第一次B3训练边界续行复核：重新读取计划、manifest与当前代码；
  matrix仍为DBA/scaling×B2/B3四单元，floor=0.5、linear power=1、anchor
  fraction=1.0及两组trial-plan hash均未变化。`status/`、`rounds/`、
  `quality_gates.csv`和raw训练日志均不存在，没有可轮询的运行句柄；正式训练尚未启动。
- 2026-09-02 第二次B3训练边界续行复核：manifest hash仍为
  `23be498407e760566fcbb0de06d22bb312fd5d6d7b987ce82441576584b38d66`，4个spec、
  两组trial-plan、候选唯一anchor参数与分析器契约均未变化；status/rounds/gates/log
  计数仍为0。阻塞条件与第一次相同，继续等待人工训练，不重复dry-run或修改候选。
- 2026-09-02 第三次B3训练边界续行复核：manifest hash、4个spec、floor=0.5、
  power=1、candidate anchor=1.0与代码契约仍未变化；status/rounds/gates/log计数
  连续第三次均为0。当前已无可在不越过人工训练授权前提下推进的工作，目标模式
  正式blocked；仅在人工执行唯一B3命令并恢复后继续验收。
- 2026-09-02 续行复核：B2 manifest仍为同一两个dry-run单元，floor=0.5、
  linear power=1、trial-plan hash均未变化；`status/`、`rounds/`和
  `quality_gates.csv`仍不存在，正式训练尚未启动。
- 2026-09-02 第三次B2训练边界复核：仍无status、rounds或quality-gate产物，
  且代码、参数和配对证据均已就绪；当前已无可在不启动训练的前提下继续完成的
  工作，目标模式正式blocked，等待人工执行唯一B2命令后恢复。

## 阶段交接记录

### 2026-09-02 / B1R floor=0.10 完成，转 floor=0.50 单参数重试

- 实验目录：`logs/rtc_v3_semantic_risk_gated_seed42_mf03`。
- 完成性：LIE 与 DBA 均 60 轮完成，exit code=0；11/11 质量门通过。
- 配对证据：LIE trial-plan hash
  `de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`；
  DBA trial-plan hash
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`。
- LIE：active ACC 0.771174→0.771858（+0.0684 pp），final ACC
  0.8192→0.8246（+0.54 pp）；恶意权重 28.5556%→28.5362%，恶意 impact
  31.4735%→31.4187%，良性权重 66.9199%→67.2665%，zero mass
  4.5245%→4.1973%。首次分叉为 round26 良性 cid18，risk=0.0590，权重
  0.09410→0.1。
- DBA：active ACC 0.839266→0.839052（-0.0214 pp），final ACC
  0.8737→0.8736；active ASR 2.2536%→2.2302%，恶意权重
  0.1083%→0.1068%，zero mass 33.4469%→33.5519%。首次分叉为 round28
  良性 cid13，risk=0.0474，权重 0.09526→0.1。
- 判定：拒绝 floor=0.10；安全护栏全部通过，但 LIE active ACC 增益未达到
  +0.3 pp。
- 下一唯一动作：运行 `rtc_semantic_restricted_only` 的 2 单元最小筛选，参数
  `semantic_intervention_risk_floor=0.50`；命令见“当前状态”，随后继续等待人工恢复。
- 交接加固：通用分析器已增加 status 终态验证，并在已完成的 floor=0.10 两单元
  上重放通过；完整测试套件 405 passed，`git diff --check` 无错误。

### 2026-09-02 / B1R floor=0.50 通过，进入 B2 线性 q cap

- 实验目录：`logs/rtc_v3_semantic_restricted_only_seed42_mf03`。
- 完成性：LIE与DBA均completed、exit code=0、last round=60、各61行；
  11/11质量门通过。
- 配对证据：LIE和DBA trial-plan hash分别为
  `de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`、
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`。
- LIE：active ACC 0.771174→0.775966（+0.4792 pp），final ACC
  0.8192→0.8407（+2.15 pp）；恶意权重28.5556%→28.5775%，良性权重
  66.9199%→71.4%，恶意impact31.4735%→30.2719%，zero mass
  4.5245%→0.0225%。首次分叉round26良性cid18，risk=0.0590，权重
  0.09410→0.1。
- DBA：active ACC 0.839266→0.839258（-0.0008 pp），final ACC
  0.8737→0.8750；active ASR 2.2536%→2.3753%，恶意权重
  0.1083%→0.2540%，恶意impact0.1828%→0.4310%，zero mass
  33.4469%→31.5460%。首次分叉round11两名恶意客户端，仍为quarantined。
- 判定：接受并冻结floor=0.50；全部B1R效用、安全、完成性和质量门通过。
- 下一唯一动作：B2只加入`cumulative_q_cap_power=1`，运行LIE z=0.5与DBA
  两个候选单元；脚本、命令和验收条件见“当前状态”，随后等待人工恢复。

### 2026-09-02 / B2 线性 cumulative-q cap 通过，进入 B3 准备

- 实验目录：`logs/rtc_v3_cumulative_q_cap_seed42_mf03`。
- 完成性：LIE与DBA均completed、exit code=0、last round=60、各61行；11/11质量门通过。
- 配对证据：LIE和DBA trial-plan hash分别为
  `de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`、
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`，
  与B1R完全一致。
- LIE：active ACC 0.775966→0.783058（+0.7092 pp），final ACC
  0.8407→0.8638；恶意权重0.285775→0.253088（-3.2687 pp），良性权重保持
  0.714，恶意impact0.302719→0.277523，zero mass0.000225→0.032912。
- DBA：active ACC、final ACC、active ASR、恶意权重、恶意impact和zero mass
  均与B1R完全一致，active ASR=2.3753%、恶意权重=0.2540%。
- 误伤与实现审计：LIE cap-active 61个客户端行全部为恶意，良性为0；两种攻击
  各500行均满足`weight <= nominal × client_q_cap`。首次分叉round34恶意cid7。
- 判定：接受并冻结linear q cap；无需q²重试。B2解决的是恶意质量过高，而不是
  再次牺牲良性质量。
- 下一唯一动作：B3只加入coordinate-median anchor recycle，先完成代码、测试、
  dry-run、严格配对基线检查和分析器，再给出人工训练命令。

### 2026-09-02 / B3 代码与dry-run就绪，等待人工训练

- 候选：`rtc_cumulative_q_cap_anchor`，相对B2只增加
  `anchor_recycle_fraction=1.0`；已有coordinate-median聚合实现与硬预算约束不改。
- 代码验证：新增组合别名单元测试；全量`413 passed`，`git diff --check`无错误。
- dry-run目录：`logs/rtc_v3_anchor_recycle_b3_seed42_mf03`；无status、rounds或
  quality-gates训练产物，确认未启动训练。
- matrix：DBA×{B2,B3}与scaling×{B2,B3}共4单元。每对候选只增加anchor参数；
  trial-plan与attack contract均相同。scaling补跑B2 baseline是严格配对所必需。
- 冻结参数：seed42、mf=0.3、IID、60轮、攻击从round11开始；DBA/scaling均为
  strong、replacement gain=1、poison fraction=0.3。
- 分析器：`analysis/rtc_anchor_recycle_b3/analyze_results.py`已完成，验收ACC、ASR、
  恶意/良性权重、impact、zero/effective mass、anchor target/actual/budget limit、
  误伤率、首次anchor激活和首次客户端权重分叉。
- 唯一人工训练命令与恢复后分析命令见“当前状态”。达到训练边界后停止，不能
  自行执行`-Execute`。
- 第一次续行复核仍无任何训练产物或运行句柄；manifest、配对hash与候选唯一参数
  均保持不变。继续等待同一人工命令，不得通过重复dry-run代替正式实验。
- 第二次续行复核仍无训练产物或运行句柄；manifest hash为
  `23be498407e760566fcbb0de06d22bb312fd5d6d7b987ce82441576584b38d66`，所有冻结证据
  保持不变。当前无可在不启动训练前提下继续完成的工作。
- 第三次续行复核仍无训练产物或运行句柄；相同阻塞条件已连续出现三次，目标模式
  正式blocked。人工训练完成并恢复后，必须先核对四个status、各61行轮次（其中
  50个攻击活跃轮）、
  exit code、quality gates和两组严格配对，再运行B3分析器。

### 2026-09-03 / B3 完成但安全门失败，转 B3R 单机制重试准备

- 实验目录：`logs/rtc_v3_anchor_recycle_b3_seed42_mf03`。
- 完成性：DBA/scaling×B2/B3四单元均completed、exit code=0、last round=60、
  各61行；13/13 quality gates通过。
- 配对证据：DBA trial-plan hash
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`；scaling
  trial-plan hash `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`；
  同攻击的B2/B3使用同一attack contract、初始模型、采样序列与恶意身份。
- DBA：active ACC 0.839258→0.844402（+0.5144 pp），final ACC
  0.8750→0.8875；active ASR 0.023753→0.521673（+49.7920 pp），peak ASR
  0.029778→0.897778；恶意权重0.002540→0.002477，良性权重
  0.682000→0.675787，恶意impact 0.004310→0.004319，zero mass
  0.315460→0，anchor recycle mass均值0.321736。良性watch/restricted率分别增加
  3.6683/1.2381 pp，quarantine率保持0。
- Scaling：active ACC 0.835866→0.837596（+0.1730 pp），final ACC
  0.8731→0.8829；active ASR 0.017264→0.272500（+25.5236 pp），peak ASR
  0.024111→0.678444；恶意权重0.001958→0.002237，良性权重
  0.705907→0.670705，恶意impact 0.003541→0.004127，zero mass
  0.292135→0，anchor recycle mass均值0.327058。良性watch/restricted率分别增加
  6.4952/3.1659 pp，quarantine率保持0。
- 机制证据：两种攻击的anchor均在round11首次激活，客户端权重在round12首次分叉。
  实现先按`nominal[positive]`对所有正质量客户端的clipped update求coordinate median，
  后由solver产生最终客户端权重，再将缺失质量完整回填到该预先构造的anchor；回填项
  仅计server anchor/total exposure，没有客户端、principal residual或semantic归因。
  这解释了“恶意客户端权重/impact近乎不变但ASR暴涨”的代理指标失真。
- 判定：拒绝B3，两个ASR门均失败；ACC提升与zero mass改善不能覆盖安全退化。
- 下一唯一动作：B3R只把recycle anchor改为按solver最终accepted weights构造，并
  用该新anchor重新计算server anchor/total exposure；不得回退到nominal anchor，
  不同时改recycle fraction、q power、residual rank-cap或clipping。完成代码、测试、
  dry-run和分析器后再给出人工训练命令。

### 2026-09-03 / B3R accepted-weight anchor 就绪，等待人工训练

- 候选：`rtc_cumulative_q_cap_accepted_anchor`；底层defense type仍为`rtc_full`。
  相对失败B3唯一增加`anchor_recycle_weighting=accepted`；floor=0.5、linear q cap、
  anchor recycle fraction=1.0均保持不变。
- 实现：nominal anchor继续用于原有residual分解与客户端QP约束；仅recycle anchor
  在solver后按最终accepted weights重算，并以其真实norm重算server anchor/total
  hard-budget系数。accepted总质量为0时不使用nominal fallback。
- 代码验证：阶段新增测试全部通过；全量`418 passed`；`git diff --check`仅有既有
  LF/CRLF提示，无空白错误。
- dry-run目录：`logs/rtc_v3_anchor_recycle_b3r_seed42_mf03`；manifest hash
  `f7de91c35d85f3a05143639983e3e91de4e95746230b8c7d642af572c755abe8`；仅DBA和
  scaling两个候选单元。当前无status/rounds/quality-gate训练产物。
- 配对证据：DBA/scaling分别复用B3目录内B2 baseline的trial-plan hash
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`与
  `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`；
  attack contract也逐项相同。
- 验收：每种攻击分别检查ACC、ASR、恶意/良性权重、impact、zero/effective mass、
  良性watch/restricted/quarantine、anchor实际质量、source availability、首次anchor/
  客户端权重/ASR分叉；安全与效用阈值沿用B3预注册门。
- 唯一人工执行命令与恢复后分析命令见“当前状态”。达到训练边界后停止；不得自行
  执行`-Execute`。
- 第一次续行复核仍无status、rounds、quality gates或raw训练日志；manifest、两组
  配对hash和唯一参数差异保持不变。当前没有可分析结果或可轮询运行句柄，继续等待
  同一人工命令。
- 第二次续行复核仍只有dry-run manifest/matrix/trial plans；训练status、rounds、
  quality gates、运行汇总与raw log均为0。manifest文件与内嵌hash、当前代码契约和
  唯一命令均未漂移；相同人工训练边界继续成立。

### 2026-09-03 / B3R full recycle 完成但 DBA ASR 门失败，转 fraction=0.5 单参数重试

- 实验目录：`logs/rtc_v3_anchor_recycle_b3r_seed42_mf03`。
- 完成性：DBA由完整61行cache复用；scaling首次在round53训练期间被Windows外部
  终止，随后以同一run id、trial-plan hash和输出目录从round0完整重跑。最终两个
  单元均60轮、exit code=0、各61行，11/11质量门通过。
- 配对证据：DBA/scaling trial-plan hash分别为
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`与
  `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`；与B2/B3
  相同，attack contract、初始模型、采样序列和恶意身份均通过质量门。
- DBA：active ACC 0.839258→0.850794（+1.1536 pp），final ACC
  0.8750→0.8869；active ASR 0.023753→0.030647（+0.6893 pp），peak ASR
  0.029778→0.046889；恶意/良性权重0.002540/0.682000→0.002692/0.682000；
  恶意impact 0.004310→0.004802；zero mass 0.315460→0，anchor mass 0.315308。
  良性watch +0.0635 pp，restricted/quarantine均不变。
- Scaling：active ACC 0.835866→0.846768（+1.0902 pp），final ACC
  0.8731→0.8798；active ASR 0.017264→0.016880（-0.0384 pp），peak ASR
  0.024111→0.026667；恶意/良性权重0.001958/0.705907→0.002234/0.704369；
  恶意impact 0.003541→0.004187；zero mass 0.292135→0，anchor mass 0.293397。
  良性watch +0.9167 pp，restricted/quarantine均不变。
- 机制证据：两种攻击均在round11首次启用accepted anchor，round12首次客户端权重
  分叉且均发生在恶意客户端；DBA在round19首次出现逐轮ASR增幅>0.5 pp。运行时
  accepted weighting与source availability均为100%。
- 判定：拒绝full-recycle B3R；仅DBA active ASR门失败，且超限0.1893 pp。ACC和
  zero mass收益不能覆盖安全门失败。
- 下一唯一动作：B3R-F0.5只把accepted anchor的recycle fraction从1.0降到0.5；
  运行strong DBA与strong scaling两个seed42候选单元，命令与门槛见“当前状态”，
  随后等待人工恢复。

### 2026-09-03 / B3R-F0.5 代码与dry-run就绪，等待人工训练

- 候选继续使用`rtc_cumulative_q_cap_accepted_anchor`；相对B3R唯一变化是
  `anchor_recycle_fraction=0.5`。manifest、trial-plan与attack-contract守卫已加入
  专用脚本，未暴露全量rerun开关。
- dry-run目录：`logs/rtc_v3_anchor_recycle_b3r_f05_seed42_mf03`；2个candidate-only
  单元，manifest SHA256
  `eaadb61ce7b56b74487e9b76a7940d071e580c55a64ad8017d60d9b16701e4c7`。
- 分析器已参数化fraction-only reference检查，并会同时核对运行时fraction、accepted
  weighting、完成性、质量门、严格配对、ACC、ASR、恶意/良性权重、impact、zero
  mass、误伤率及首次分叉。
- 验证：全量测试`424 passed`，`git diff --check`无空白错误；再次dry-run后仍为
  status=0、rounds=0，且无quality gates或raw训练日志，确认未误启动训练。
- 唯一人工执行命令与恢复后分析命令见“当前状态”；到达训练边界后停止，不得自行
  执行`-Execute`。
- 三次独立恢复均重新核验manifest、matrix、产物目录与关键实现；配置和hash未漂移，
  且始终只有dry-run文件，没有任何训练status、rounds、quality gates、运行汇总或
  raw日志。当前已无不越过人工训练边界即可继续完成的工作，目标模式正式blocked。
  人工训练完成并恢复后，必须先核对两个status终态、各61行轮次、exit code、全部
  quality gates、两组trial-plan/attack-contract严格配对及运行时fraction/weighting，
  再运行B3R-F0.5分析器决定接受、回退或继续B3；不得直接进入B4。
- 首次人工执行没有进入训练runner：专用脚本的manifest守卫使用了Windows PowerShell
  5.1不存在的`ConvertFrom-Json -Depth`参数。该兼容性问题已修复，并由同版本
  PowerShell的完整dry-run验证通过；训练配置与manifest hash不变，目录中仍无训练
  产物。下一动作仍是人工重新执行原唯一命令，两个单元均需正式运行。

### 2026-09-03 / B3R-F0.5 完成但 scaling zero-mass 门失败，转 fraction=0.51

- 实验目录：`logs/rtc_v3_anchor_recycle_b3r_f05_seed42_mf03`。DBA与scaling均
  completed、exit code=0、last round=60、各61行；11/11 quality gates通过。
- 配对证据：DBA/scaling trial-plan hash分别为
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`与
  `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`；
  attack-contract hash与B2/B3/B3R一致，运行时fraction=0.5、weighting=accepted。
- DBA：active ACC 0.839258→0.846106（+0.6848 pp），final ACC
  0.8750→0.8817；active ASR 0.023753→0.026338（+0.2584 pp）；恶意/良性权重
  0.002540/0.682000→0.002596/0.682000；恶意impact 0.004310→0.004521；
  zero mass 0.315460→0.157702（ratio=0.499911），anchor mass=0.157702。
- Scaling：active ACC 0.835866→0.842050（+0.6184 pp），final ACC
  0.8731→0.8765；active ASR 0.017264→0.016811（-0.0453 pp）；恶意/良性权重
  0.001958/0.705907→0.002126/0.705410；恶意impact 0.003541→0.003899；
  zero mass 0.292135→0.146232（ratio=0.500562），anchor mass=0.146232。
- 机制证据：anchor均在round11首次激活，round12首次客户端权重分叉且均为恶意
  客户端；DBA在round56首次出现单轮ASR增幅>0.5 pp。良性watch仅小幅增加，
  restricted/quarantine不变，source availability=100%。
- 判定：拒绝B3R-F0.5。唯一失败为scaling zero mass ratio超过0.5约0.000562；
  不事后放宽门槛，也不把其余安全与ACC收益用于覆盖该失败。
- 下一唯一动作：B3R-F0.51只将fraction从0.50调到0.51；DBA与scaling两个
  candidate-only单元，其他配置与配对hash冻结不变。脚本、Windows PowerShell 5.1
  dry-run、全量测试、manifest守卫和分析说明均已就绪；执行命令与门见“当前状态”，
  随后等待人工恢复。
- 第一次训练边界复核未发现status、rounds、quality gates或raw训练日志；manifest、
  脚本、分析器和anchor实现hash均未漂移，没有可验收结果或可继续分析的数据。
- 第二次训练边界复核结果不变：仍仅有4个dry-run文件，全部配置与代码hash保持
  冻结；继续等待同一人工命令，尚未达到可做结果验收或进入B4的条件。
- 第三次训练边界复核仍无任何训练产物，且配置与实现未漂移；相同人工训练阻塞已
  连续出现三次，目标模式正式blocked。恢复后先验证两个status、各61行rounds、
  exit code、全部quality gates和严格配对，再运行B3R-F0.51分析器。

### 2026-09-04 / B3R-F0.51 通过，冻结 accepted anchor 并进入 B4 准备

- 实验目录：`logs/rtc_v3_anchor_recycle_b3r_f051_seed42_mf03`；DBA与scaling两个
  candidate-only运行均completed、exit code=0、last round=60、各61行，11/11质量门通过。
- 配对证据：DBA/scaling trial-plan hash分别为
  `3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`与
  `d27d846ab6310cc6bda82b3d0efd88a1f8e2fc58495e3b73858bfa03dab7a359`；attack contract、
  初始模型、采样序列和恶意身份与B2/B3/B3R一致。运行时只将fraction由0.50调为0.51。
- 核心结果：DBA active/final ACC=84.6014%/88.03%，active ASR=2.5711%，恶意/
  良性权重=0.2572%/68.2000%，恶意impact=0.4532%，zero mass=15.4560%；
  scaling active/final ACC=84.2148%/87.80%，active ASR=1.6871%，恶意/良性权重
  =0.2112%/70.5140%，恶意impact=0.3863%，zero mass=14.3447%。两攻击active/final
  mean ACC=84.4081%/87.9150%。
- 机制证据：anchor均在round11首次激活，round12首次权重分叉且仅涉及恶意客户端；
  DBA的单轮ASR增幅在round58首次超过0.5 pp，但攻击活跃期平均增幅仅0.1958 pp。
  两攻击zero-mass ratio分别为0.489950/0.491028；良性watch率下降，restricted/
  quarantine不变，accepted source availability=100%。
- 判定：接受B3R-F0.51并冻结；全17项预注册检查通过。高ACC的full-recycle B3R
  仍因DBA active ASR增幅0.6893 pp失败，不能当作最优可接受候选。
- 下一唯一动作：在冻结的B3R-F0.51上准备B4 residual rank-cap的单机制改动；
  不同时改q power、floor或clipping，并在人工训练边界前完成代码、测试、dry-run与分析器。

### 2026-09-04 / B4 residual rank-cap 就绪，等待人工训练

- 候选：`rtc_b4_residual_rank_cap`；相对B3R-F0.51只加top-2 full-residual rank cap×0.5。
  原有缺失质量仍按0.51回填，rank-cap实际新移除质量按1.0转给accepted anchor；
  实现使用同轮reference QP精确分解两类缺失质量，并重用现有anchor hard-budget限制。
- 实验：唯一PowerShell脚本串行生成/执行两个matrix，共3个单元：LIE B3R-F0.51
  基线、LIE B4候选、DBA B4候选；DBA基线严格复用已完成B3R-F0.51。LIE固定z=0.5，
  恶意比固定0.3，两攻击trial-plan/attack-contract hash均与现有基线一致。
- 验证：新增实现与分析契约测试，全量`440 passed`；Windows PowerShell 5.1 dry-run
  exit code=0，再次执行后两个manifest hash稳定，status/rounds/gates/raw均未生成。
- 分析器：验证完成性、配对和单机制差异；同时输出ACC、ASR、恶意/良性权重、
  impact、zero/effective mass、良性状态误伤、rank-cap良性/恶意命中率、reference QP、
  rank-recycle质量分解和首次权重分叉。预注册门见“当前状态”和分析README。
- 唯一人工命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_residual_rank_cap_b4_ablation.ps1' -Execute`
- 恢复后先验证3个status、各61行rounds、exit code、两个新目录的全部quality gates、
  两组trial-plan/attack-contract严格配对和运行时rank契约；再运行唯一分析命令，
  决定接受B4进入B5或拒绝并回退B3R-F0.51。
- 阻塞审计：到达本阶段人工训练边界后的两次独立续行均重新核验计划、manifest、
  训练产物和当前代码；连同最初到达边界的回合，同一人工训练阻塞已连续三个目标
  回合成立。两个目录始终只有dry-run文件，没有status、rounds、quality gates或raw
  训练日志，也没有可轮询训练句柄；本阶段现正式blocked，等待人工执行上述唯一命令。

### 2026-09-04 / B4 完成但效用门失败，回退并进入 B5 准备

- 完成性：3个新单元均完成60轮、exit code=0、各61行；LIE基线10/10、候选11/11
  quality gates通过。LIE/DBA trial-plan hash分别为`de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`
  与`3e951f0860377e0fbfd592b19a98c3f9cfbf44ab1d95f3d19b69db7830957a67`，
  attack-contract和单机制差异检查通过。
- 核心结果：LIE active/final ACC为78.0796%/86.03%→76.8336%/86.10%，恶意权重
  25.1071%→20.8158%、恶意impact 27.6117%→24.9253%；DBA active/final ACC为
  84.6014%/88.03%→84.5716%/88.54%，active ASR 2.5711%→2.4960%。
- 机制证据：两攻击每个活跃轮均精确选中top-2，reference QP有效率100%，移除质量与
  rank recycle一致且单轮不超过10%。LIE的100次rank-cap命中含32次良性，首次分叉
  已在攻击前round1出现；DBA仅6次良性命中。说明该排序在LIE下区分度不足。
- 判定：拒绝B4，恢复冻结B3R-F0.51。失败检查为LIE/DBA active ACC不低于基线以及
  LIE恶意权重≤15.5%；不得事后降低门槛或继续调top-k/factor。
- 当前可比最优：B3R-F0.51在DBA+scaling的预注册宏平均active/final ACC为
  84.4081%/87.9150%；同配置补入LIE后，三攻击seed42宏平均为82.2986%/87.2867%。
- 下一唯一动作：只在冻结B3R-F0.51上准备B5 `norm_clip_mad_k=2.25`；2.0仅在2.25
  独立失败且机制证据支持时另开阶段，不与本阶段同时运行。

### 2026-09-04 / B5 MAD=2.25 就绪，等待人工训练

- 候选：`rtc_b5_clip_mad_225`；底层defense仍为`rtc_full`，相对冻结B3R-F0.51
  只显式增加`norm_clip_mad_k=2.25`，默认基线为2.5，且不包含B4 rank-cap参数。
- 实验：唯一脚本只生成/执行LIE z=0.5和strong DBA两个candidate单元；恶意比0.3、
  seed42、IID、60轮、round11起攻击、20客户端、participation=0.5。两条基线均复用，
  round1–10作为各自严格配对的clean clipping窗口。
- 验证：新增参数化、运行时日志、非法值和matrix单机制测试；全量`447 passed`。
  Windows PowerShell 5.1 dry-run两次通过，manifest SHA256稳定为
  `79e88dda7cb4f00a5ae4089def6dac84b227469cf671ac046deadc05ffaeeb0a`，未生成训练产物。
- 分析器：输出ACC/ASR/权重/impact/zero mass/良性状态、preattack clean与active
  良性/恶意clipping rate、首次clipping分叉；同时检查运行时k=2.25、完成性、质量门、
  trial-plan/attack-contract和仅单参数差异。预注册门见“当前状态”和分析README。
- 唯一人工命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_clip_mad_b5_225_ablation.ps1' -Execute`
- 恢复后先核验2个status、各61行rounds、exit code、全部quality gates和manifest hash，
  再运行`analysis/rtc_clip_mad_b5_225/README.md`中的唯一分析命令。不得直接运行2.0。
- 2026-09-04 第一次B5训练边界续行复核：重新读取完整计划、B5 manifest和当前实现；
  manifest、执行脚本、分析器与RTC实现SHA256分别仍为
  `79e88dda7cb4f00a5ae4089def6dac84b227469cf671ac046deadc05ffaeeb0a`、
  `b391d3e7b4958acdd8a51fd4c28db072636eb389ad6dfbc5e7791eaee4896ebd`、
  `e19e83fe0935613e5a01da6a3f1bd651444121d2228bac7e9f8d0378389c21d6`、
  `a88c3e4fcd5e1e1321580049af6c14dca8c9c6a7c10d4265798b049eb068b2f9`，
  均未漂移。实验目录仍只有manifest、matrix和两份trial plan，status/rounds/raw/
  quality gates均为0；`Get-Process`未发现python/pythonw/raylet进程。当前没有可验收
  的B5训练结果，也没有已确认的运行句柄；继续停在同一人工GPU训练边界。
- 2026-09-04 第二次B5训练边界续行复核：完整计划共741行，manifest仍严格包含
  LIE z=0.5与strong DBA两个seed42、mf=0.3、`norm_clip_mad_k=2.25`单元；两组
  trial-plan/attack-contract hash及manifest、脚本、分析器、RTC实现hash均未漂移。
  status/rounds/raw/quality gates再次全部为0，且未发现python/pythonw/raylet进程。
  连同最初到达B5人工边界和第一次续行复核，同一外部训练阻塞已连续三个目标回合
  成立；当前目标模式正式blocked。人工执行下述唯一命令并恢复后，必须先做完成性、
  质量门和严格配对验收，再运行B5分析器；不得直接进入2.0或V1：
  `& 'D:\workspace\FL2\experiments\run_rtc_clip_mad_b5_225_ablation.ps1' -Execute`
- 2026-09-04 LIE mean ACC口径纠偏：历史目录
  `logs/rtc_v3_lie_z05_seed42_mf03/lie_z05_analysis/summary.csv`中的RTC active mean ACC
  79.0324%计算正确，但目录名误导；manifest实际为mf=0.2、trial-plan hash
  `d2accf167adbf5528ca210bc5f84b55468a28ab315a31642efaffe95a57ac2f5`。正式mf=0.3
  严格配对的原始`rtc_full`为77.1174%，trial-plan hash
  `de965e3cccfcfe99daaf9610b1f3e3357ff585d7fafa3ca644b356c9c4c8663c`；当前冻结
  B3R-F0.51为78.0796%，相对同mf、同trial-plan基线提高0.9622 pp，而非倒退。
  旧mf=0.2结果高于当前0.9528 pp只反映不同攻击者比例/恶意身份与trial plan的混合
  差异，不能归因于微调。以后任何“mf03”结论必须以manifest为准，并自动检查目录
  标签与`malicious_fraction`一致性。

### 2026-09-04 / B5 MAD=2.25 完成但效用门失败，冻结 B3R-F0.51

- 完成性：LIE z=0.5与strong DBA两个候选均completed、exit code=0、last round=60、
  各61行；全部quality gates、trial-plan/attack-implementation pairing和运行时
  `norm_clip_mad_k=2.25`检查通过。
- 结果：LIE active ACC 78.0796%→78.0406%（-0.0390 pp），恶意权重/impact
  分别下降0.2979/0.3299 pp；DBA active ACC 84.6014%→84.3576%
  （-0.2438 pp），active ASR下降0.1533 pp，但DBA攻击前clean clipping仍恰好8%。
- 判定：拒绝2.25。失败项为LIE/DBA active ACC不低于基线、LIE active ACC严格
  提高、DBA preattack clean clipping严格<8%。不事后放宽门槛。
- 机制：LIE多裁剪恶意客户端却没有utility增益；DBA恶意客户端基线已100%被裁剪，
  候选只增加良性裁剪。继续收紧到2.0不具备修复这些失败项的机制证据，因此不运行。
- 产物：`analysis/rtc_clip_mad_b5_225/`包含comparison、clipping、deltas、首次分叉、
  decision以及通过validator并渲染的技术报告。最终冻结候选恢复为B3R-F0.51/k=2.5。
- 下一唯一动作：准备V1五seed正式RTC/Multi-Krum验证，不再新增微调参数。

### 2026-09-04 / V1 多seed验证就绪，等待人工训练

- 设计：5 seeds × 4 conditions × 2 defenses=40格。条件为显式clean、LIE z=0.25、
  LIE z=0.5、strong DBA；防御为冻结B3R-F0.51和Multi-Krum。固定mf=0.3、IID、
  60轮、20客户端、每轮10个，Multi-Krum使用f=3/select=5。
- 最小化：严格复用当前hash一致的seed42 RTC LIE z=0.5与DBA两格；四个新矩阵共
  38格。旧Multi-Krum结果因attack implementation hash过期而不能复用。
- 代码：runner新增显式`attack=none` clean支持，仅用于V1验证，不改变RTC算法；两份
  V1 attack freeze分别锁定LIE z=0.25以及LIE z=0.5/strong DBA。分析器会汇总40格，
  验证所有完成性/质量门/参数/配对hash并计算五seed配对t检验和95%置信区间。
- 验证：全量`452 passed`、`git diff --check`无空白错误；Windows PowerShell 5.1
  dry-run两次通过，四个manifest hash稳定，status/rounds/quality gates均未生成。
- 唯一人工命令：
  `& 'D:\workspace\FL2\experiments\run_rtc_v1_multiseed_validation.ps1' -Execute`
- 恢复后：先逐目录核验38个新status、每个61行rounds、exit code和全部quality
  gates，再运行`analysis/rtc_v1_multiseed/README.md`中的唯一分析命令。只有40格
  （含2格复用）全部通过后，才能生成最终RTC/Multi-Krum报告并完成目标。
- 已到人工GPU训练边界；本回合不启动训练。

### 2026-09-04 / V1 按用户决定缩减为3 seeds，重新等待人工训练

- 停止证据：旧5-seed脚本的首个clean/RTC单元在round15后按用户要求终止；根进程
  PID 25516及其Python/Ray子树已全部退出，复核时`python/pythonw/raylet`进程数为0。
  该单元没有rounds终态文件或quality gates，不属于已完成实验；旧status已如实标记
  `state=interrupted`、`error_type=UserInterrupted`、`last_round=15`，旧5-seed目录
  不进入后续分析，也不从半轮次续跑。
- 新设计：固定seeds 42/46/47，仍保持clean、LIE z=0.25、LIE z=0.5、strong DBA
  和RTC/Multi-Krum严格配对，共24格。继续复用当前hash一致的seed42 RTC LIE z=0.5
  与DBA两格，因此新训练为22格；没有已完成的新V1格需要重跑。
- 新目录：clean与LIE z=0.25分别为
  `logs/rtc_v3_v1_clean_seeds42_46_47_mf03`、
  `logs/rtc_v3_v1_lie_z025_seeds42_46_47_mf03`；seed42 Multi-Krum目录保持不变；
  seeds46/47的LIE z=0.5/DBA目录为
  `logs/rtc_v3_v1_lie_z05_dba_seeds46_47_mf03`。四目录均只有dry-run产物，
  status/rounds/quality gates/raw训练日志均为0。
- 四个manifest SHA256依次为
  `44dd1af59a71171570980027640c4c3722438a3bc3e30a57b15428f62efd9a80`、
  `049b063b4bfe22565d8134770a4784c0cd44892c5b792462a6cdf888b7862e98`、
  `76819b66e3a881a1c4ff15946a4cf8f56f59338a2f8afb079254085942e56f07`、
  `53fe8fedf3a4b2ffb03ca096594662944a3978c32f8cf69c2cedf6eeeae87c0f`；
  Windows PowerShell 5.1重复dry-run通过且hash稳定。
- 分析契约：V1分析器现要求24格/12个严格配对，除ACC和DBA ASR外同时输出恶意/
  良性权重、impact、zero/effective mass、anchor mass、身份与行为口径误伤率及首次
  机制分叉。三seed配对t检验仅有2个自由度，最终报告必须说明置信区间明显更宽，
  不得将“不显著”解释为“两者等效”。
- 验证：全量测试`454 passed`，分析器`py_compile`与`git diff --check`通过；仅有
  既有LF/CRLF提示。执行脚本/分析器SHA256分别为
  `b9a173bfff9a28ead8add99540a081238e113c9d2cd513e836d87a42fff320c4`、
  `c431bacfca6c07f5f656a93aa894291edcdf4df71c976ef4b4649874c3c9a81c`；RTC、runner
  和两份attack freeze均未改动。
- 下一唯一动作：由人工执行
  `& 'D:\workspace\FL2\experiments\run_rtc_v1_multiseed_validation.ps1' -Execute`。
  恢复后先核验22个新status终态、各61行rounds、exit code、全部quality gates及
  12对trial-plan/attack-implementation hash，再运行分析器并生成最终对照报告。
  本回合不自行启动训练。
- 第一次3-seed训练边界续行复核：重新完整读取869行计划、四个manifest、当前执行
  脚本、分析器、两份attack freeze、runner与RTC参数实现。四个manifest和执行脚本
  hash均未漂移；四个新目录的status/rounds/raw/quality gates仍全部为0，且没有
  `python/pythonw/raylet`进程或其他可轮询训练句柄。分析器进一步把
  `exploratory_three_seed_paired`、自由度2、未执行等效性检验和“非显著不证明等效”
  同时写入`statistical_tests.csv`与`decision.json`；全量`454 passed`、`py_compile`
  通过，新分析器SHA256为
  `c431bacfca6c07f5f656a93aa894291edcdf4df71c976ef4b4649874c3c9a81c`。
  继续停在同一人工训练边界，不自行执行训练命令。
- 第二次3-seed训练边界续行复核：重新完整读取878行计划，并再次核对四个manifest、
  执行脚本、分析器、两份attack freeze、runner与RTC实现。manifest SHA256仍依次为
  `44dd1af59a71171570980027640c4c3722438a3bc3e30a57b15428f62efd9a80`、
  `049b063b4bfe22565d8134770a4784c0cd44892c5b792462a6cdf888b7862e98`、
  `76819b66e3a881a1c4ff15946a4cf8f56f59338a2f8afb079254085942e56f07`、
  `53fe8fedf3a4b2ffb03ca096594662944a3978c32f8cf69c2cedf6eeeae87c0f`；
  脚本/分析器SHA256仍为
  `b9a173bfff9a28ead8add99540a081238e113c9d2cd513e836d87a42fff320c4`、
  `c431bacfca6c07f5f656a93aa894291edcdf4df71c976ef4b4649874c3c9a81c`。
  四个新目录的status/rounds/raw/quality gates仍全部为0，且实际
  `python/pythonw/raylet`进程数为0，没有运行句柄或终态结果可验收。连同3-seed
  设计完成并首次到达边界的回合、第一次续行复核，同一人工训练阻塞已连续三个目标
  回合成立；当前目标模式正式blocked。仅在人工完成22个新单元并恢复后继续完成性、
  质量门和12对严格配对验收，不得自行启动训练或提前生成最终报告。

### 2026-09-06 / V1 三 seed 验证完成，冻结最终 RTC 候选并结束本轮微调

- 恢复与完整性：本次恢复先重新完整读取本计划和四个V1 manifest，并核对当前
  status、rounds、quality gates、执行脚本、分析器、两份attack freeze、runner及
  RTC实现。人工已完成全部22个新增单元；连同2个当前契约复用单元，24/24格均为
  `completed`/exit code 0/last round 60，每份rounds均恰有61行且覆盖0--60。
  四目录quality gates分别为9/15/11/17行，共52/52通过；12个RTC/Multi-Krum对的
  trial-plan与attack-implementation hash逐对一致。LIE参数确认精确为z=0.25和0.5，
  mf=0.3；不存在z=1调参。四个manifest及训练代码hash均未漂移，复核时亦无
  `python/pythonw/raylet`进程。本回合没有启动训练。
- 分析与独立复核：`analysis/rtc_v1_multiseed/analyze_results.py`成功生成
  `runs.csv`、`paired_by_seed.csv`、`summary_by_condition.csv`、
  `statistical_tests.csv`、`first_mechanism_divergence.csv`和`decision.json`。
  新增的`independent_verify.py`不导入主分析器，使用Python标准库直接从24份原始
  rounds/client CSV重算所有ACC、ASR、恶意/良性权重、impact、zero/effective/
  anchor mass与身份/行为误伤率；24个run和12个pair逐项完全一致。两份新增脚本
  `py_compile`、独立复核及`git diff --check`均通过。
- 效用结果（RTC B3R-F0.51对Multi-Krum，RTC-MK）：clean rounds1--60平均ACC
  80.8249% vs 80.1046%，+0.7203 pp，最终89.2667% vs 87.7067%，+1.5600 pp；
  LIE z=0.25 active rounds11--60为82.7939% vs 81.5391%，+1.2548 pp，最终
  +0.3167 pp；LIE z=0.5为77.9207% vs 77.5773%，+0.3433 pp，但最终
  86.6367% vs 87.3833%，-0.7467 pp；strong DBA为84.4373% vs 84.3372%，
  +0.1001 pp，最终+0.2633 pp。四条件等权宏平均ACC为81.4942% vs 80.8895%，
  RTC高0.6046 pp，说明本轮微调已消除“RTC平均ACC一直略低”的点估计现象。
- 安全结果：strong DBA active平均ASR为RTC 2.3956% vs Multi-Krum 1.7947%，
  RTC高0.6010 pp；跨seed峰值为3.3333% vs 2.7889%。三seed配对95% CI为
  [0.0190, 1.1829] pp，双侧p=0.0471。该安全退化不能被DBA平均ACC仅+0.1001 pp
  覆盖，因此严格的“相对Multi-Krum提升效用且安全不退化”目标没有完全实现。
- 机制结果：RTC的恶意权重/恶意impact相对Multi-Krum差值在LIE z=0.25为
  +8.1942/+10.0097 pp，在LIE z=0.5为+9.3106/+11.0764 pp；这表明剩余问题不只是
  “错误损失良性质量”，还包括对LIE恶意更新抑制不足。RTC良性clipping率在
  clean/LIE .25/LIE .5/DBA分别为6.0089%/2.3371%/5.0693%/0.2878%，良性watch
  为0/0/3.9280%/1.8371%；restricted/quarantined良性率均为0。RTC zero mass为
  0.0400%/2.0963%/1.8733%/14.8428%，accepted-anchor mass为
  0.0416%/2.1819%/1.9498%/15.4486%。LIE攻击者clipping为23.1559%/29.0733%，
  DBA攻击者clipping与quarantine均为100%；但DBA仍有更高ASR，说明客户端恶意
  权重/impact代理没有完整刻画后门方向及anchor recycle的几何安全性。
- 首次分叉：全部12个condition/seed配对从round1即因聚合选择/权重不同而分叉；
  三个攻击条件的指标窗口首次客户端级分叉均在攻击起始round11。完整CID、身份、
  attack-active状态和字段差异已写入`first_mechanism_divergence.csv`。
- 统计边界：正式范围为三seed探索性配对检验，n=3、df=2，未做等价性检验。
  所有条件的平均ACC差95% CI均跨0；尤其LIE z=0.5为[-1.9330, 2.6196] pp。
  因此三seed足以结束当前工程微调和选择候选，但不足以支持等价性、论文或强泛化
  声明；若需要后者，应在预注册主要终点后补到至少五seed。
- 最终判定：冻结B3R-F0.51（floor=.5、linear cumulative-q cap、accepted anchor
  fraction=.51、默认MAD k=2.5）为当前最优RTC研究候选。它完成了缩小/逆转ACC
  差距的工程目标，但不应宣称全面优于Multi-Krum，DBA安全优先场景仍优先
  Multi-Krum。若开启下一项独立研究，应优先处理LIE恶意权重/impact和DBA方向敏感
  anchor安全门，不能在相同三seed上继续未预注册参数搜索。
- 最终报告：规范化产物位于`analysis/rtc_v1_multiseed/artifact.json`，由
  `build_report_artifact.py`从已复核CSV构建；包含两张原生图、四张审计表、来源路径
  与可运行DuckDB查询。Data Analytics artifact validator已返回`ok=true`，随后
  report renderer返回`ok=true`。本轮B0→B1→B1R→B2→B3→B4→B5→V1计划与最终
  RTC/Multi-Krum报告均已完成；不再安排GPU实验。

## 阶段交接记录模板

每次恢复后在本节追加一条：

```text
日期/阶段：
实验目录与 run_id：
完成性：轮数、exit code、质量门
配对证据：trial-plan hash
核心结果：ACC、ASR、恶意/良性权重、impact、zero mass、误伤
机制证据：首次分叉轮次与客户端
判定：接受 / 拒绝 / 需单参数重试
下一唯一动作：
若需实验：脚本、参数、单元数、执行命令；随后 blocked
```
