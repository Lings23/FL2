# M3 / RTC 单seed全攻击评估：人工交接

2026-09-18。用户指定仅运行RTC，不重新运行已有其他防御。本批不以“已全面优于Multi-Krum”为前提：M3的LIE仍有已知差距，完整表用于检验表现。

## 冻结版本与12项矩阵

M3=M2+H2b；运行别名`rtc_i12_eligibility_confirmed`。M2包含B0+R1c+R2；H2b加入参考主体资格门与confirmed历史。
保留已接受的累计约束和0.51 accepted回填，不加入G1/G2低尾cap，不重新校准。
协议保存H2b已接受的完整custom_params和三个校准JSON的规范化哈希，避免Windows/Linux换行差异导致误报。

| 条件 | 固定参数 |
|---|---|
| clean | 无攻击，仍只运行M3 |
| Gaussian | mean=0，std=0.1 |
| Random-v2 | trainable-only Rademacher，scale=10，保留buffers；强度未验证 |
| Sign-flip | scale=1 |
| LIE .25 / .5 | 两个独立单元，all_updates |
| Min-Max / Min-Sum | inverse_sign，gamma_init=.001，tolerance=1e-6，max_iterations=64，gamma_fraction=1 |
| Targeted label flip | source=5，target=3，poison_fraction=1 |
| All-reverse label flip | y→9-y，poison_fraction=1 |
| Scaling backdoor | aggregation-aware，replacement_gain=1，poison_fraction=.3 |
| DBA | aggregation-aware，gain=1，poison_fraction=.3；paper_cifar_1x6_2x2，normalized_white |

合计12新训练单元，全部为RTC；其他防御0，训练结果复用0。ALIE是LIE别名，不重复统计。
seed42与旧Linux基线脚本对齐，属于已有开发seed；不称为独立留出，不给单seed显著性结论。

CIFAR10、ResNet18无预训练、IID20客户端每轮10人，60轮，本地5epoch、batch96。
SGD learning_rate=.01、momentum=.9、weight_decay=.0001、cosine；mf=.3，攻击11—60轮。
严格TrialPlan与确定性客户端训练；每client Ray CPU1/GPU.125，object store3072MiB、最低可用内存10240MiB、等待120秒、自动重试0。
脚本同时固定CLI和runtime配置的batch96/GPU.125，无需修改主config或共享bridge。
抽样可能出现单轮攻击者超过f=3，表中报告次数，不将随机压力实验称为每轮≤f保证。

Random-v2本地仅有准备产物，没有完成的两seed FedAvg强度筛选。用户本批只跑RTC，因此新增显式`--byzantine-evaluation-only`固定强度评估路径：
freeze必须声明`purpose=fixed_strength_evaluation_only`、v2/trainable范围及`unverified_evaluation_only`，manifest保留未验证标签。
原正式freeze入口仍要求FedAvg强度证据；该评估不证明攻击对无防御有效，不使用RTC结果选强度，不与v1旧表直接合并。
Gaussian参数沿用旧协议，RTC平滑本身不证明攻击无效或防御已覆盖其他噪声设计。

## 质量核验与结果表

全部12项完整、exit0、0—60共61轮、600客户端、runner质量门、攻击执行证据、源码/配置/TrialPlan/数据哈希和初始化记录核验。
复用H2b机制验证：谱参考与R1c、R2、资格/confirmed历史、累计重放、预算和回填守恒。
整数CSV接受`2.0`，拒绝`2.5`、NaN、Infinity；不修改原始CSV。
完整性/质量失败时分析拒收，不悄悄删除失败单元或自动重跑；较差ACC/ASR仍是应报告结果，不以效用门筛掉。

输出`analysis/rtc_m3_results.csv`和`analysis/rtc_m3_table.md`：
- ACC：clean用1—60轮，攻击用11—60轮均值；最终和51—60末10轮均值。
- ASR：两种标签翻转、DBA和Scaling输出攻击期均值、峰值、末10轮及最终；其他条件N/A。
- ASR分母：targeted标签翻转为真实source=5样本；all-reverse为全测试集的y→9-y映射；后门为带触发器的非target样本。
- 同时保留联合良性标记率、恶意聚合权重和超过f=3的轮数。CSV比例为0—1，Markdown显示百分数。

本地旧目录`logs/rtc_v3_formal_all_attacks_two_seed_mf03`中的49份已完成配置均batch48/GPU.25。
对应Linux seed42/batch96的`logs/rtc_v3_formal_attacks_seed42_rtx5090_b96_gpu0125_mf03`尚未复制到本地，故现在不自动合并其他防御。
收到服务器原日志后，核对相同seed、训练/模型/数据与验证口径、攻击版本/参数、TrialPlan/初始化/客户端随机流及环境。
满足合同的基线可进入配对表；不满足的只能单列历史参考或缺项，不因都叫seed42而宣称严格公平。
该旧Linux脚本不包含Multi-Krum、clean/Gaussian/Random；这些列还需对应的实际日志，不能由其他防御代填。

## G2处置

G2服务器32/32完成，98质量门通过；独立核验271份源码、60产物及19200谱参考，最大cosine误差1.1650784503824951e-11，与服务器判定一致。
候选60门58通过：LIE对M3平均ACC +4.0156pp、最终+1.20pp，Gaussian误标相对G1-R从9/6降至0/0。
Sign201良性7/351（1.9943%）超过1%，平均ACC−.2058pp低于−.2pp。本批保持拒绝，M3不变。
有价值机制完整归档至`analysis/rtc_retained_candidates/M3-G2`；374项证据包含服务器源码。M2-G1和M2-H2原记录保持。
先按用户要求完成M3全攻击表，再继续G2的Sign误伤修复；不静默舍弃收益，也不提前晋升。

## 准备验证

9项M3合成检查、9项G2合成回归、7项Random正式强度证据检查通过；未执行真实训练测试。
PowerShell语法、Bash -n及Windows默认脚本12项dry-run通过；Linux实际dry-run由用户执行。
缺失训练结果时分析正确拒收。最终冻结293份源码/63份产物，status/rounds/raw训练文件0。
锁SHA256：`3b11a920d04b1178be4f59785a77905f4e7f416e6d0dd235a83034337584242b`。
源码zip SHA256：`5897bb0bc9b4af13297b47b5cd4a9cb6f83b61334788ff09a69eeef8f0689743`。
检查记录：`analysis/rtc_m3_all_attacks_preparation/checks.json`。
初次仅准备的产物因receipt哈希改为跨平台规范化JSON，被保留至`logs/rtc_m3_all_attacks_seed42_preflight_receipt_bytes`；无训练结果。实际入口使用重新完整dry-run后的默认目录。

## 手动Git同步

未自动commit/push。其他现有修改不加入本次暂存清单。大体积audit_bundle.zip仅本地，Git忽略。

```powershell
Set-Location 'D:\workspace\FL2'
git add -- experiments/rtc_m3_all_attacks.py experiments/run_rtc_m3_all_attacks.ps1 experiments/run_rtc_m3_all_attacks.sh experiments/run.py experiments/rtc_v3/byzantine.py config/rtc_m3_all_attacks_protocol.json tests/test_rtc_m3_all_attacks.py analysis/rtc_g2_integration_review analysis/rtc_m3_all_attacks_preparation analysis/rtc_retained_candidates/README.md analysis/rtc_retained_candidates/M3-G2
git add -f -- docs/RTC_M3_ALL_ATTACKS_HANDOFF_20260918.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_GOAL_PROMPT.md docs/RTC_BYZANTINE_IMPROVEMENT_PLAN_20260911.md
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare M3-only full-attack evaluation and retain G2 findings"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

服务器拉取前保留自身未提交改动；不要覆盖旧日志，不将本地Windows准备锁复制成Linux可执行锁。Linux在自己的环境准备新输出。

## 人工运行

两个平台二选一。脚本默认仅准备/dry-run；真实训练必须由用户显式执行。
Linux更新代码并激活fl2后先准备（不训练）：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_m3_all_attacks.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data
```

上一步成功生成12项锁后，用户手动启动：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_m3_all_attacks.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --execute
```

全部完成后在原Linux宿主分析：

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_m3_all_attacks.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --analyze
```

Linux输出：`/home/jia_zhang/hqr/FL2/logs/rtc_m3_all_attacks_seed42`。
Windows本地默认入口dry-run完成后的唯一启动命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_m3_all_attacks.ps1' -Execute
```

Windows恢复后分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_m3_all_attacks.ps1' -Analyze
```

Windows输出：`D:\workspace\FL2\logs\rtc_m3_all_attacks_seed42`。
GPU配额当前由阶段脚本固定.125；准备后不可只改主config或锁文件。确需换资源，应事先修改注册协议和入口、重新测试，在新目录准备，并记录与旧基线资源偏差。
完整单元经runner缓存验证可跳过；failed/incomplete/orphan先报错供人工检查，不支持轮内checkpoint续训，也不自动重跑。

阶段`WAITING_FOR_MANUAL_EXPERIMENT`，新边界计数1；未启动任何训练。按宿主规则，同一阻塞连续三个目标回合才正式标记blocked。
本批是M3表现评估，不等于全目标完成。独立多seed/非IID、消融和保留候选修复仍待完成。
