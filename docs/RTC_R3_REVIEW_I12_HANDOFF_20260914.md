# R3观测验收与I12集成桥接交接

日期：2026-09-14。R3观测质量已接受；I12仅完成准备，未训练、未验收。总体目标未完成。

## R3完成性与诊断

三个计划run均completed/exit0/round60，分别61个唯一轮号、600条客户端记录，20/20 runner质量门通过。
原分析器experiments/rtc_r3_observation.py --analyze成功；analysis/rtc_r3_review/verify.py独立从CSV重算
ACC、权重、谱参考与三分之二计票、raw范数比、累计q和质量守恒，结论一致；最大参考误差<1.8e-15，累计q误差0。
严格配对核验trial plan、数据、初始化、恶意身份、实际参与序列和随机流；两个新模块均observe。

| LIE z=.5，seed42，round11--60 | RTC B0 | Multi-Krum |
|---|---:|---:|
| 平均ACC | 78.0796% | 76.9732% |
| 最终ACC | 86.03% | 87.09% |
| 每轮恶意权重 | 25.1071% | 18.0000% |
| 方向/幅度攻击标记 | 均0/143 | 均0/143 |
| 方向/幅度良性标记 | 均0/357 | 均0/357 |

RTC累计q首次在round34下降，143条攻击记录中63条q<1，约63.7270%的恶意权重发生于q仍为1时。
clean平均81.0985%、最终89.47%，两个观测均0/600标记。观察通过不能解释为新增防御通过。
这支持后续继续研究慢攻击延迟，不能假设R1c或R2已覆盖LIE；也不能把单seed平均ACC较高当泛化结论。

R3原锁SHA256：0687c4c25aeb33b3f44a290a731d9154137a23221f2144693cdb5594b2f6d87f。
源码归档analysis/rtc_r3_review/r3_sources.zip，SHA256：53125c3d3a15c3c621a24572c77f94a4f51a85235e67f20fdf108a44482acee3。
新增I12源码后，原R3主分析器的全工作区来源检查会按设计拒绝；历史验收使用归档与独立verify.py复核，不能改写旧锁。

## I12候选与矩阵

父版本D=B0+R1c方向cap，候选C=D+固定R2 raw cap；阈值、累计参数、MAD和回填不调参。
辅助组B0和B0+N用于四组归因。R2旧批次+0.2pp效用门失败保持，新问题是安全收益与效用非劣。

| 每个seed（固定44、45） | RTC四组 | 挑战者 | 单元 |
|---|---|---|---:|
| clean | B0、D、N、C | 无 | 4 |
| Sign-flip，trainable v2 scale1 | 同上 | MK | 5 |
| Gaussian，mean0/std.1 | 同上 | MK、RFA | 6 |
| LIE，z.5/all_updates | 同上 | MK | 5 |

共40个新训练单元，复用旧基线0。四条件均每seed独立配对，不跨平台复用基线。
两seed全批结果返回才判定，不因中途好看就停止或换seed；n=2只作工程筛选。
DBA/scaling进入targeted阶段前另做转移；Random-v2、其他慢攻击、label reversal及最终新seed/非IID仍待验证。

固定合同：CIFAR10、ResNet18无预训练、IID20客户端每轮10人、60轮，本地5轮、batch48、
SGD lr.01/momentum.9/weight decay.0001/cosine；攻击round11--60、恶意比例.3；strict/principal_uniform/确定性训练。
RTC floor=.5、累计power1、accepted回填.51、MADk2.5。MK f3选5；RFA迭代3、smoothing1e-6、按样本数加权。
资源两端默认CPU1/GPU.25每客户端，object store3072MiB、最低可用内存10240MiB、等待120s、重试0。
L20不能为提高并发静默改变batch size或训练参数；任何资源调整须另登记配置并重新准备独立目录。

## 预登记验收

权威数值见config/rtc_i12_protocol.json，训练前已冻结：

- 每个条件、每个seed，C相对D及D相对B0，active/final ACC退化各不超过0.2个百分点。
- Sign-flip中D和C均保留相对B0至少1个百分点平均ACC增益，final不下降。
- Gaussian中C恶意权重不超过D的一半；D已经为零时C须保持≤1e-8。
- 攻击条件下C恶意权重相对D增加不超过0.001；D/C的联合良性cap误标率分别≤1%。
- 全部适用完整性、冻结合同、严格配对、预算≤1e-8、几何重构相对误差≤1e-4、累计重放误差≤1e-9。
- N单独的安全/效用结果另判，不把单项效用失败自动等同于组合失败；但各组质量数据必须有效。

报告C-D增量、C-B0总效果、四组交互，MK/RFA比较、两seed分别结果、首次权重/sketch分叉及配对可用负向投影代理。
每客户端LOO轴不同，投影和不称为净攻击向量。未检验targeted安全，ASR为不适用，不填0。
通过仅晋升上述范围内的研究版本；不是全面优于MK，也不是最终目标完成。

## 验证与冻结

11项I12合成测试通过，包括phase6多轮组合cap、标签独立性、拒收损坏q/权重/累计证据、
独立N去留与组合判定分开、旧攻击退化拒收、缺单元拒收和禁止自动重跑失败单元。
11项已有raw模块合成测试通过。PowerShell语法检查和40单元dry-run通过，Bash -n通过。
实际Linux训练环境dry-run尚未执行，不能称为Linux已验收。

Windows最终输出D:\workspace\FL2\logs\rtc_i12_bridge_v2；准备后status为0，无训练启动。
冻结230份源码/测试/依赖文件、66份计划产物；锁SHA256：
1669371a3c4ea30d735a1eb03d91ae03ecd170553ec91fa9b7c103262ec0073a。
logs/rtc_i12_bridge是之前只做dry-run的草稿，保留不复用；当前唯一交接目录为rtc_i12_bridge_v2。

Windows环境：Python3.11.9、torch2.12.1+cu126、numpy2.4.4、CUDA12.6、RTX4060。
Linux按REMOTE_L20.md的/root/FL2与/root/FL2/.venv独立prepare，记录本机环境、源码与攻击哈希。
校准内容用canonical JSON hash跨平台核验，避免CRLF/LF差异造成伪漂移；各自实际字节hash仍进入本地锁。

## 人工入口

GitHub同步本批提交并核验后，Windows启动：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_i12_bridge.ps1' -Execute
```

Windows恢复分析：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_i12_bridge.ps1' -Analyze
```

Linux先在/root/FL2更新codex/periodic-attack-defenses分支，然后仅验证，不训练：

```bash
bash /root/FL2/experiments/run_rtc_i12_bridge.sh
```

待用户返回Linux dry-run结果、环境和锁后，核验再交接Linux真实训练命令，不能以Windows验证代替。
Linux对应分析入口为同一脚本加--analyze；没有训练结果时它应拒绝验收。
Windows/Linux是同一研究的替代执行环境，二选一，不需要两边各跑40项。

两脚本可覆盖Python路径及输出目录；数据目录仅prepare时可指定，必须在对应环境重新生成合同。
数据集和虚拟环境须按REMOTE_L20.md准备，不提交到Git；运行本批不需要拷贝旧训练日志或源码zip。
历史R3独立复核需要其本地logs与r3_sources.zip；这些历史证据不作为Linux训练的运行时依赖。

同一批次再次执行时，已完成单元由runner验证后跳过；失败、未完成、孤儿产物会先拒绝并要求人工复核。
不支持从中断轮次checkpoint续训；不得把跳过已完成单元说成训练断点恢复。
脚本不会自动扫参、扩大seed、串接下一候选。本批未验收前不开发依赖其结果的R3算法。

## 状态

阶段WAITING_FOR_MANUAL_EXPERIMENT。交接后的同一人工实验阻塞计数从1开始；
宿主规则要求连续三个目标回合后才可正式blocked，不能提前声称已blocked。
本次属于progress：完成R3独立验收/归档、实现I12桥接与双平台入口、必要测试及Windows合同冻结。
