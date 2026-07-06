# FreqFed、FoolsGold、FLTrust 防御方法补充设计

## 1. 目标与现状

在现有 `BaseDefense -> FedSecStrategy.aggregate_fit()` 扩展链路上补齐三种防御，保持客户端训练协议不变，并统一输出客户端信任度、最终聚合权重和防御运行指标。

仓库现状：

- FLTrust 已有基础实现、可信根数据集和服务端更新回调；
- FoolsGold 已有按稳定客户端 ID 累积历史更新的基础实现；
- FreqFed 尚未实现；
- FLTrust/FoolsGold 当前缺少可配置参数、完整异常处理、专项指标和覆盖主要边界的测试。

本次建议采用“一个方法一个文件”的结构，保留 `defense_base.py` 中的公共接口和工具函数，避免继续扩大单文件。

## 2. 统一数据约定

所有方法接收 `updates: List[(local_params, num_examples)]`，并在聚合前由策略调用：

```python
defense.set_context(server_round, client_ids, global_params)
```

统一约定：

- `local_params` 是客户端训练后的完整模型；
- `delta_i = local_params_i - global_params`；
- 相似度、距离和范数仅使用浮点张量，整数/布尔 buffer 不参与检测；
- 返回完整的新全局模型，非浮点 buffer 默认从上一轮全局模型复制；
- 防御只依据更新、稳定客户端 ID 和可信根数据，不读取 `is_malicious` 或 `attack_active` 标签；
- 所有除法使用 `eps=1e-12`，输入先做 NaN/Inf 检查；异常客户端权重置零，并记录原因。

统一可观测字段：

- `last_client_trusts[cid]`：方法自身产生的 `[0, 1]` 分数；
- `last_client_weights[cid]`：归一化前有效权重；
- `last_client_aggregation_weights[cid]`：最终归一化权重；
- `last_round_metrics`：选择数量、拒绝数量、全零降级、算法耗时等；
- 客户端标签只在实验结束后用于计算 TPR/FPR，不得进入防御决策。

## 3. FreqFed

### 3.1 算法流程

FreqFed 使用本轮客户端的**完整模型权重**构造频域指纹，而不是使用模型增量；这是与论文定义保持一致的复现路径。

1. 对每个客户端、每个浮点权重张量执行正交归一化 DCT-II。
2. 张量维度处理：
   - `ndim >= 2`：将前 `ndim-1` 维展平为行、最后一维为列，执行二维 DCT；
   - `ndim == 1`：执行一维 DCT；
   - 标量和非浮点 buffer 跳过。
3. 从每层频谱左上低频区域提取系数。首版采用每个轴前 `ceil(size * low_frequency_ratio)` 个系数；各层指纹先做 L2 归一化再拼接，避免大层完全支配距离。
4. 对指纹做可选的确定性随机投影，控制高维模型内存占用。
5. 使用 cosine distance 构造样本距离，对本轮客户端运行 HDBSCAN。
6. 丢弃噪声标签 `-1`，选择成员最多的簇；并列时依次选择平均成员概率更高、簇内平均距离更小、标签更小的簇，确保可复现。
7. 仅对入选客户端的原始完整模型做按 `num_examples` 加权 FedAvg。

FreqFed 是多数簇假设：若攻击者控制本轮多数客户端，它不能提供保证。实验配置必须显式记录每轮恶意参与比例。

### 3.2 小样本与失败降级

HDBSCAN 至少需要足够的本轮样本。规则应显式且可审计：

- `n < min_cluster_size`：不做聚类，回退 FedAvg，设置 `freqfed_fallback=1`；
- 全部为噪声或最大簇小于 `min_selected_clients`：默认回退 coordinate-wise median，而非任意选择一个客户端；
- 指纹包含 NaN/Inf：对应客户端拒绝；若剩余数量不足，再按上述规则降级；
- 不使用真实恶意客户端数量设置聚类参数。

### 3.3 配置

```yaml
security:
  defense:
    type: freqfed
    custom_params:
      low_frequency_ratio: 0.20
      min_cluster_size: 3
      min_samples: 2
      min_selected_clients: 2
      cluster_selection_method: eom
      projection_dim: 4096       # 0 表示不投影
      fallback: median           # median | fedavg | keep_global
      seed: 42
```

依赖优先使用 `scipy.fft.dctn`。HDBSCAN 建议显式加入 `hdbscan` 依赖；若不希望增加依赖，可使用新版 scikit-learn 的 HDBSCAN，但必须锁定版本，不能用 DBSCAN 冒充论文算法。

### 3.4 指标

`freqfed_selected_clients`、`freqfed_rejected_clients`、`freqfed_noise_clients`、`freqfed_cluster_count`、`freqfed_selected_ratio`、`freqfed_fallback`。

## 4. FoolsGold

### 4.1 算法流程

1. 计算客户端本轮浮点模型增量 `delta_i`。
2. 按稳定 `cid` 累积历史贡献 `H_i <- decay * H_i + delta_i`。
3. 对本轮参与客户端的历史向量计算两两 cosine similarity，主对角置零。
4. 执行 pardoning：相似度最大值较小的客户端，不因与高相似度客户端的单边相似而受到同等惩罚。
5. 计算 `alpha_i = 1 - max_j(cs_ij)`，按最大值归一化，再执行论文中的 logit 变换并裁剪到 `[0,1]`。
6. 推荐最终权重为 `alpha_i * num_examples_i`，使防御权重与本项目 FedAvg 语义一致；提供 `use_num_examples=false` 作为严格复现实验开关。
7. 对完整客户端模型进行显式权重聚合。

### 4.2 状态管理

- 历史必须按 `cid` 保存，不能按本轮结果下标保存；现有实现满足这一点；
- 增加 `history_decay`，默认 `1.0` 对应原始累积历史，非平稳/间歇攻击实验可设为 `<1`；
- 增加 `history_max_idle_rounds`，长期离线客户端历史可清理，防止状态无限增长；
- 模型结构变化时清空历史并记录 warning；
- 零向量与单客户端轮次赋予权重 1，不应产生 NaN；
- 全部权重为零时回退 FedAvg，并设置 `foolsgold_fallback=1`。

### 4.3 配置与指标

```yaml
custom_params:
  history_decay: 1.0
  history_max_idle_rounds: 50
  use_num_examples: true
  logit_offset: 0.5
  similarity_eps: 1.0e-12
```

指标：`foolsgold_mean_trust`、`foolsgold_min_trust`、`foolsgold_zero_weight_clients`、`foolsgold_history_clients`、`foolsgold_fallback`。

适用边界：主要防御多个 Sybil/串谋客户端产生方向高度相似的更新；独立攻击者、仅一个恶意客户端，或高度 non-IID 导致诚实客户端天然相似时，识别能力有限。

## 5. FLTrust

### 5.1 算法流程

1. 从与客户端训练集严格不相交的可信根数据上，从当前全局模型出发训练服务端模型，得到 `delta_server`。
2. 对每个客户端计算 `delta_i` 和信任分数：

   `TS_i = ReLU(cos(delta_i, delta_server))`

3. 将每个非零客户端增量归一到服务端增量的范数：

   `delta_i_norm = delta_i * ||delta_server|| / ||delta_i||`

4. 使用 `TS_i` 归一化加权：

   `delta_global = sum_i(TS_i * delta_i_norm) / sum_i(TS_i)`

5. 返回 `global_params + server_lr * delta_global`。

严格复现时不再乘 `num_examples`，避免样本量绕过信任分；如需工程对照，单独提供 `use_num_examples` 开关，默认关闭。

### 5.2 根数据与生命周期

- 当前 `main.py` 已预留并从联邦训练数据中剔除 root subset，应保留该隔离；
- 仅启用 FLTrust 时预留 root subset，其他防御不应无故减少客户端训练数据；这是现有数据流水线需要修正的一点；
- 根数据应按类别分层抽样；若数据集无标签接口，则确定性随机抽样并记录类别不可知；
- 服务端 root optimizer 参数单独配置，不复用客户端 optimizer；默认 SGD、1 epoch；
- 每轮必须从当轮全局模型重新计算服务端更新，不跨轮缓存；
- `delta_server` 近零或所有 `TS_i=0` 时默认采用 `delta_server` 更新，而不是静默保持全局模型，以保留可信训练信号；可通过配置切换为 `keep_global`。

### 5.3 配置与指标

```yaml
security:
  defense:
    type: fltrust
    root_dataset_size: 100
    custom_params:
      root_epochs: 1
      root_batch_size: 32
      root_optimizer: sgd
      root_learning_rate: 0.01
      root_momentum: 0.0
      server_lr: 1.0
      use_num_examples: false
      zero_trust_fallback: server_update  # server_update | keep_global
```

指标：`fltrust_mean_trust`、`fltrust_positive_trust_clients`、`fltrust_zero_trust_clients`、`fltrust_server_delta_norm`、`fltrust_fallback`。

适用边界：依赖小规模、干净且与任务分布基本匹配的可信根数据；根数据污染或严重分布偏移会直接影响信任方向。

## 6. 代码改造清单

1. 新增 `defenses/freqfed_defense.py`、`defenses/foolsgold_defense.py`、`defenses/fltrust_defense.py`。
2. `defenses/defense_base.py`：保留公共工具；注册三个类；增加统一的输入校验、降级聚合和指标复位辅助函数。
3. `config/config.yaml`：加入 `freqfed` 类型与三种方法的参数示例。
4. `config/config_loader.py`：参数继续放在 `custom_params`，避免为每个算法扩张顶层 dataclass；实现方法内部的类型/范围校验。
5. `strategies/fed_strategy.py`：将 `hasattr(set_server_update)` 改为显式能力接口（如 `requires_server_update=False`），防止其他类偶然同名触发 FLTrust 路径。
6. `main.py`、`server/fl_server.py`：仅 FLTrust 划分根数据；分层采样；将 root 训练参数与客户端参数解耦。
7. `requirements.txt`：加入锁定范围的 HDBSCAN 实现依赖。
8. `experiments/sweep.py`：`DEFAULT_DEFENSES` 加入 `freqfed`。
9. `README.md`：补充启动命令、前置条件和方法边界。

迁移时先移动现有 FoolsGold/FLTrust 代码并保持行为测试通过，再逐项增强，降低一次性回归风险。

## 7. 测试与验收

### 7.1 单元测试

- 通用：空更新、单客户端、整数 buffer、NaN/Inf、零范数、客户端顺序变化、不同样本数；
- FreqFed：DCT 指纹确定性、低频切片形状、明显离群频谱被排除、并列簇确定性、全噪声和小样本降级；
- FoolsGold：两个相同历史客户端降权、独立方向保留、跨轮 ID 重排、衰减与离线清理、全零历史；
- FLTrust：同向/反向/正交更新、范数归一、零服务端更新、全零 trust、根训练每轮以当前全局参数为起点。

### 7.2 集成测试

- Flower `aggregate_fit` 能正确透传稳定 client ID；
- 三种方法均写出客户端权重记录和 round metrics；
- FLTrust 启用时存在且仅存在一份不相交 root subset；
- FreqFed/FoolsGold 启用时不预留 root subset；
- checkpoint、FedAvg/FedProx 正常；FedAdam/FedYogi 与 robust aggregator 的组合先标为实验性，因为二次服务端优化会改变论文语义。

### 7.3 实验矩阵

建议至少使用 3 个随机种子，比较 `none / freqfed / foolsgold / fltrust / time_consistency`，覆盖：

- 攻击：label flip、Gaussian/Byzantine、backdoor、DBA、model replacement；
- 分布：IID、Dirichlet `alpha=0.5`、`alpha=0.1`；
- 恶意比例：10%、20%、40%；
- 指标：clean accuracy、ASR、收敛轮次、TPR/FPR、聚合耗时、峰值内存、降级次数。

验收门槛不写死单一准确率，而采用相对基线：无攻击时 clean accuracy 下降不超过预设容差；攻击时 ASR/accuracy degradation 显著优于 FedAvg；所有结果报告均同时给出均值和标准差。

## 8. 实施顺序

1. 重构但不改变现有 FLTrust/FoolsGold 行为，补齐回归测试；
2. 修正 FLTrust 根数据生命周期和参数解耦；
3. 完成 FoolsGold 状态、权重和指标增强；
4. 引入依赖并实现 FreqFed；
5. 跑 smoke test、完整 sweep 和消融实验；
6. 根据误报率调参，不使用测试攻击标签在线调阈值。

## 9. 参考

- FreqFed, NDSS 2024: https://www.ndss-symposium.org/wp-content/uploads/2024-620-paper.pdf
- FoolsGold: https://arxiv.org/abs/1808.04866
- FLTrust: https://arxiv.org/abs/2012.13995
