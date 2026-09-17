# RTC 拜占庭鲁棒性目标状态

## 2026-09-17 / 最新：指定目录H2 batch96验收，H2b等待人工实验

用户指定logs/rtc_reference_eligibility；该目录现为Linux完成结果，旧Windows准备及configfix等待已过期。
按24个合同run_id核验全部completed/exit0/last_round60，每项61轮/600客户端；92项runner门、250源码/52产物通过。
独立14400条谱参考最大cosine差8.8913e-12；原机制、资格记忆、累计/预算、有限性和严格配对通过。
CSV的2.0/3.0整数字段仅作数值无损导入，1920项有完整记录，拒绝小数与非有限值，原文件不改。
审计analysis/rtc_reference_eligibility_review/verify.py、review.json；本地31门与服务器一致，29通过2失败。
锁65de3cb6680acff790ab3f4a97a6a0d34ac94cdcbd880c18d410c3a324fa436e；
源码zip 7638f68db62e8327bc54cae37e2691332b97d01fb458d966574202dd53402dca。

实际全部batch96、每client GPU.1（用户训练前选择96），仅按本批父子配对，不混用旧batch48。
Sign seed201 M2→H2：平均ACC80.1984→79.9716（-.2268pp，低于-.2门），最终84.20→84.52；
恶意每轮平均权重5.3510%→5.6019%（+.2509pp，高于+.1门），误伤4/351→0/351。
seed204 Sign及其他三条件父子无分叉。故H2不晋升，但消除误伤的实测价值保留。
round52误伤被撤销后原标记仍写入历史，导致round53三条正确攻击拒绝被撤销；逐客户端来源已核查。
M2-H2独立保留登记retained_for_memory_repair，302份审计输入及原配置/源码快照，包97097261字节，
SHA256 5422938b0157db17f1fe66718ae3066891b7c565b78757d7211bcb8f5cd1e91f；大包仅本地、Git忽略。
M2-G1原11项哈希再次一致，继续保留；batch96下不能直接继承其batch48校准泛化或收益结论。

H2b只改reference_eligibility_memory=confirmed：资格弃权保留旧历史，当前R2拒绝优先；
无弃权的正常判断照旧更新，同主体去重，成功聚合后同时提交。不调三分之二、R1c/R2、累计或回填。
原H2保留作未接受研究对照，M2为主线对照，H2b为rtc_i12_eligibility_confirmed。
旧轨迹重放保留四条误伤修复，恢复round53三条攻击拒绝；不是闭环ACC证明。
32新单元、复用0：seeds201开发/205新工程筛选；每seed clean3/Sign4/Gaussian5/LIE.5四项，含M2/H2/H2b及挑战者。
205注册前无本地运行产物，201抽样与本次原结果一致；固定batch96、GPU.1、60轮、本地5轮，其他训练/攻击参数不变。
每seed各条件对M2和H2分别验收active/final差≥-.002、恶意平均权重增加≤.001；误伤≤1%且不超过H2，
201 Sign相对M2误伤严格减少、相对H2撤销恶意标记数严格减少；全部完成性/质量/配对/预算及状态重放须通过。
19项纯合成检查、PS语法、Bash -n、Windows32项默认dry-run通过；Linux实机dry-run待返回，未训练。
新目录logs/rtc_reference_memory，274源码/60产物，训练产物0；缺结果时分析器正确拒绝。
锁84f3ce261ec6a2f771cde455a5a5e51cd8f32ba31aeb6eb115df2ef7894a833f；
源码zip ff19717b7479ba38bc232caeb9b0ffb26ada711585436fefc71a7df5cfbf5f03。
完整参数/预登记门/手动Git及双平台命令docs/RTC_H2_REVIEW_H2B_HANDOFF_20260917.md。
Windows人工启动：powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_memory.ps1' -Execute
分析同入口改-Analyze；Linux仅交接实机dry-run。状态WAITING_FOR_MANUAL_EXPERIMENT。
本回合progress，实际结果解除旧阻塞，完成验收/保留/诊断和H2b准备；新人工边界计数1，正式blocked须连续3回合。
未自动训练、Git提交/推送或后台等待。H2b之后必须返回G1修复/集成；全攻击、targeted、新seed/非IID最终目标未完成。

H2b首次自动续行复核（2026-09-17）：同一Windows锁84f3ce261ec6a2f771cde455a5a5e51cd8f32ba31aeb6eb115df2ef7894a833f保持。
按32个计划run_id核验，status/rounds/raw文件均0，本机python/pythonw/raylet进程0；未返回Linux实机验证或训练产物。
没有已确认属于本批的远端活动句柄，不推断外部服务器是否运行。上一回合为progress，本回合no progress，不属于verified wait。
同一人工执行边界连续计数2，尚未达到正式blocked条件。保持WAITING_FOR_MANUAL_EXPERIMENT。
未训练、后台等待、重复测试/dry-run或推进依赖H2b结果的新机制；M2-H2/M2-G1保留义务与完整目标不变。

H2b第二次自动续行复核（2026-09-17）：同一Windows准备锁保持，32项计划status/rounds/raw文件仍各0，
本机python/pythonw/raylet进程0，无已确认的远端活动句柄；不推断服务器训练是否运行。
上一回合与本回合均为no progress，不属于verified wait。同一人工执行边界已连续3个目标回合成立，
准备工作已完成且无可独立推进的必要工作，按宿主规则正式将目标标记blocked。
阶段保持WAITING_FOR_MANUAL_EXPERIMENT，等待人工返回实机验证或实验产物；完整目标未完成。
未训练、后台等待、重复测试/dry-run或推进下一候选。M2-H2及M2-G1继续保留，Git仍由用户手动提交/推送。

---

## H2最新执行修复：Linux准备参数不一致，等待重新dry-run

用户返回Linux prepare的client字典AssertionError；这是无训练准备失败，不是训练结果或候选判定。
原实现继承服务器主config/config.yaml中的客户端参数，只由命令行固定batch_size；现有堆栈不足以确定具体差异字段。
现已在H2运行配置显式固定原预登记5epochs/batch48/SGD/lr.01/momentum.9/weight_decay.0001/cosine，
保存原client配置与固定值到preparation_client_parameters.json，改为逐字段expected/actual/type的ValueError。
不修改主配置或GPU资源配额、不放宽验收、不改变H2机制或24项矩阵。9项纯合成检查通过，无真实训练。
原服务器失败目录和本地旧锁均保留；新目录logs/rtc_reference_eligibility_configfix。
最新人工无训练命令及手动Git同步见docs/RTC_H2_DRY_RUN_CONFIG_FIX.md；服务器必须先重新dry-run并返回新锁。
后续执行和分析均须带新输出路径。阶段WAITING_FOR_MANUAL_EXPERIMENT，当前等待实机准备验证；M2-G1及完整目标范围不变。
本地新目录24项dry-run通过，267源码/52产物匹配，训练产物0，所有client参数及trial-plan与旧本地准备一致。
新Windows锁e37374a699be1830ed4b489e4233303a685918ced990324633179c55a1cf8e55；
源码zip 98f10ad43db134e06be0000b91ab65d8409ed1602b0028d278f91af044460d12。旧目录和锁保持不变。
本回合为progress，修复实际准备失败并形成可执行交接；没有启动实验或自动Git提交/推送。

---

## 2026-09-16 / 最新：H1闭环拒绝，H2参考主体资格等待人工实验

本节优先于下方历史记录。本次恢复发现H1目录已是Linux终态结果，解除旧人工阻塞。
按manifest指定status/<run_id>.json和rounds/<run_id>.csv核验24/24 completed、exit0、last_round60，
每项61轮/600客户端；92项runner质量门、242份源码、51份产物、严格配对/预算/累计/历史状态全部通过。
独立重建14400条谱参考，最大cosine误差3.3611e-12；本地与服务器31项候选门一致，23通过、8失败。
初始按文件名包含status/rounds的计数不适用于该结构，已纠正为合同路径，不能将其当作未执行证据。
H1锁130303998e35f04d8f4eb41439e750dd14140b9f4b85c1936a900eb0b7cb7d51；
源码zip 9a8c45ca708af20bced412c595e34f84c41c35518808acc0ceef58356f8d811c。
审计analysis/rtc_reference_guard_review/verify.py、review.json；原服务器产物与判定不改。

Sign-flip平均/最终ACC：seed201 M2 83.7182/87.41→H1 82.4322/86.87，seed203 84.0108/87.26→83.2910/86.30；
攻击期每轮平均恶意权重4.6326%→9.2095%、2.0881%→7.5629%；误伤仍4/351、0/363。
clean/Gaussian/LIE父子轨迹不变。H1从round31/32撤销正确攻击拒绝，改变轨迹后round52污染参考变成历史正向，
seed201仍误伤4个良性客户端。故拒绝H1，不继承、不改阈值求通过；历史M2接受范围保持但不能外推新seed安全。

H2唯一变化：原R1c cap需要至少ceil(2*参考主体数/3)个主体在其上次参与未被原R1c/R2标记。
未知有资格，缺席保留、正常参与清除；同主体只计一票，当前轮只读旧状态，成功聚合后同时提交原判断。
不重算谱参考、不翻转模型、不撤销R2/累计/预算；不读攻击标签。H1关闭，G1暂未启用。
父版rtc_i12_eligibility_observe，候选rtc_i12_eligibility_cap，唯一差异reference_eligibility_mode observe→cap。
固定规则旧轨迹重放seed201误伤4→1，保留115/115攻击标记，seed203保留121/121；仅开发证据，非闭环收益。
设计及重放analysis/rtc_reference_guard_review/h2_design.json、replay_h2.py和h2_offline.json。

24个新单元、复用0：seeds201开发/204新工程筛选，每seed clean2/Sign3/Gaussian4/LIE.5三项；
注册前未发现seed204产物，201四条件trial-plan与H1相同，不消除多数攻击轮。
每seed/条件联合良性误伤≤1%，201 Sign误伤数严格减少；active/final ACC差≥-.002，
各攻击的攻击期每轮平均恶意权重增加≤.001；全部完成性/质量/配对/预算/资格状态及原机制重放须通过。
完整参数及双平台命令docs/RTC_H1_REVIEW_H2_HANDOFF_20260916.md；协议config/rtc_reference_eligibility_protocol.json。
13项纯合成检查通过，Windows24项默认dry-run、PS语法、Bash -n/LF、Python编译通过；Linux实机dry-run待返回。
缺status时分析器按预期拒绝，无提前接受结果。新目录logs/rtc_reference_eligibility的status/rounds/raw均0。
冻结267份源码/51份产物；锁4dd7b1ca353493feb633323e803dfe1dde19a26160060a98c3cb4c1914f22c1a；
源码zip f8a4accc5b154b7d662eda810b46c47d21e3f2f5408cffdaadc899d36e06648d。

M2-G1的11项原归档哈希再次一致，继续保留待修正并重新集成。H2不代表已保存G1组合收益；
其结论后必须返回低尾参考污染修复与LIE收益/安全回归验证，或提交明确替代证据，不得跳过至R4/R5。
Windows人工启动：powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_eligibility.ps1' -Execute
完成后将-Execute改为-Analyze；Linux仅交接已知服务器路径的实机无训练dry-run。
本回合progress：H1终态解除旧阻塞，完成闭环验收/诊断及H2准备。新人工边界计数1，正式blocked须连续3回合。
阶段WAITING_FOR_MANUAL_EXPERIMENT；未训练、不后台等待、不推进依赖H2结果的后续候选；总体目标未完成。
Git按用户要求手动提交/推送，本地HEAD ba54ee2，本轮未自动同步；全部攻击/targeted/最终新seed与非IID范围不缩减。

H2首次自动续行复核（2026-09-16）：Windows锁仍为4dd7b1ca353493feb633323e803dfe1dde19a26160060a98c3cb4c1914f22c1a。
按24个计划run_id核查，status文件0、rounds文件0、raw文件0，本机python/pythonw/raylet进程0。
没有返回Linux实机验证或训练产物，也没有已确认的远端活动句柄；不能推断外部服务器是否运行。
上一回合为progress，本回合为no progress，不属于verified wait。同一人工执行边界连续计数2，尚不正式blocked。
保持WAITING_FOR_MANUAL_EXPERIMENT；不训练、不后台等待、不重复测试/dry-run、不推进依赖H2结果的新机制。
M2-G1保留与后续重新集成义务不变，完整目标尚未完成。

H2第二次自动续行复核（2026-09-16）：同一Windows准备锁保持；按24个计划run_id核查，
status、rounds、raw文件仍均0，本机python/pythonw/raylet进程0，无已确认的远端活动句柄。
上一回合与本回合均为no progress，不属于verified wait；不据此推断外部服务器训练状态。
同一人工执行阻塞连续3个目标回合成立，准备工作已完成且无可独立推进的必要工作，按宿主规则正式标记blocked。
阶段保持WAITING_FOR_MANUAL_EXPERIMENT，等待人工实机验证或实验产物；完整目标未完成，M2-G1仍保留待修正和重新集成。
未启动训练、后台等待、重复测试/dry-run或推进依赖H2结果的新机制；Git仍按用户要求手动提交/推送。

---

## 2026-09-16 / 最新：参考历史观测验收，M2-H1等待人工实验

本节优先于下方历史状态。五项Linux参考历史观测已completed/exit0/round60，21项质量门通过；
234份源码、14份计划产物与3000条独立谱参考核验通过，父/观测模型轨迹、权重及原cap完全一致。
审计analysis/rtc_reference_history_review/verify.py及review.json；只接受观测质量，不接受新防御。
服务器锁909fb289c50a5db76f642c432f12c070f0a97baa76ddb4040abd5ab8b58c456e；
源码zip 1097accb5c93dd739efcf0d397a6165e85c99ac5ecd9c631d177714811edb241。

Sign-flip seed201原误伤4/351，其中round52三条参考被6名攻击者占据，历史cosine约-.0278；
round60另一误伤的历史cosine为+.03685。旧轨迹若要求历史同向，良性标记4→1，但攻击标记115→79。
clean589/600条历史参考有效，其中172条非正，说明历史更新不是可信oracle，必须检验误伤和防御损失的取舍。
M2-G1保持独立归档和待修正/重新集成状态；此阶段不启用G1，也不声称保留其LIE收益的组合已接受。

新H1唯一改动：原R1c危险标记还需有效参考与前一实际trainable聚合cosine严格>0才施加方向cap；
无历史/零范数/非正时只对该R1c判断弃权，R2、累计、裁剪和accepted回填不变，不使用攻击身份。
阈值0为固定同半空间确认，不以攻击数值搜索；上一实际聚合含anchor且仅在成功聚合后更新。
父版rtc_i12_guard_observe，候选rtc_i12_guard_cap；单一差异reference_guard_mode observe→cap。
24个新单元、复用0：seeds201（已知开发）/203（无既有产物的独立工程筛选），每seed clean2/Sign3/Gaussian4/LIE.5三项。
seed201原抽样不变；父版已知误伤失败继续报告，新协议验收候选修复，绝不追改原24项判定。

每seed各条件候选联合误伤≤1%；201 Sign误伤条数严格减少；所有条件active/final ACC差≥-.002；
所有攻击每轮恶意权重增加≤.001。全部24项质量/配对/预算/原机制与guard重放须通过；不按平均掩盖失败。
19项不同纯合成测试通过，含600条合成日志完整验证器和伪造字段拒收；未训练。
PowerShell语法/24项默认dry-run、Bash -n/LF检查通过；Linux实际dry-run待返回。
分析入口对缺少status按预期拒绝；未提前生成接受结果。
新输出logs/rtc_reference_guard，259份源码/51份计划产物，status/raw为0。
锁45b9faca8f973702e522d49707aa53c9c49573a295053a59864b832d949f6af2；
zip957119e049b47c1f7de338468936b07067ad1b9f193709f85ae36376b2a670c5。
完整参数、门、Git清单和双平台命令docs/RTC_REFERENCE_HISTORY_REVIEW_H1_HANDOFF_20260916.md。
Windows人工启动：powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_guard.ps1' -Execute
完成后将-Execute改为-Analyze；Linux先执行文档内实机dry-run，验收后再交接其真实训练命令。

本回合为progress：真实终态产物解除旧五项等待，完成观测验收和H1准备。新人工边界计数1，
正式blocked须连续三回合。阶段WAITING_FOR_MANUAL_EXPERIMENT；未启动训练、不后台等待、不推进依赖H1结果的新机制。
总体范围与G1收益保留义务不变，目标未完成。按用户指示手动Git提交/推送，本轮未自动同步GitHub。

H1首次自动续行复核（2026-09-16）：Windows准备锁仍为45b9faca8f973702e522d49707aa53c9c49573a295053a59864b832d949f6af2，
计划24项，status文件、rounds文件及raw目录均0；未收到Linux实机dry-run或训练产物。
本机发现1个python/pythonw/raylet类进程，但读取命令行的CIM接口拒绝访问，无法确认其是否属于本实验；
不把它计为训练进展，也不声称本机或外部服务器训练已停止。没有已确认属于H1的活动句柄，故不属于verified wait。
上一回合仅复核保留归档并说明已有结论，未改变下一动作，分类no progress；本回合同样no progress。
以原H1交接与本次明确复核保守计数2；尚不正式blocked。保持WAITING_FOR_MANUAL_EXPERIMENT，
等待人工实验或实机验证产物，不重复测试/dry-run、不启动训练、不推进依赖H1结果的下一机制。
M2-G1继续保留待修正并重新集成；全部攻击、targeted及非IID最终验收仍待完成。

H1第二次自动续行复核（2026-09-16）：同一Windows冻结锁保持，24项计划的status/rounds/raw仍各0，
尚无Linux实机验证或训练结果返回。本机仍有1个未确认归属的Python类进程，没有已确认属于H1的活动句柄；
不推断外部训练状态，不停止或重启任何进程。上一回合与本回合均为no progress，不属于verified wait。
同一人工执行边界已连续3个目标回合成立，准备工作已完成，无可独立推进的必要工作；按宿主规则正式标记blocked。
阶段保持WAITING_FOR_MANUAL_EXPERIMENT，等待人工返回验证或实验产物；完整目标未完成，M2-G1保留义务不变。
未训练、未后台等待、未重复测试/dry-run、未修改冻结合同或推进下一候选。Git仍由用户手动提交/推送。

---

## 2026-09-15 / 保留候选登记：M2-G1不可静默舍弃

按用户要求，M2+持续低尾cap正式登记为M2-G1，研究状态retained_for_repair_and_reintegration。
原批次仍rejected、未晋升主线；全部ACC效用门通过，失败是3项良性误伤门，不能描述成没有效果。
LIE两seed平均ACC +5.1524pp、最终+.45pp和恶意权重下降已保存为必须追踪的实测收益。
登记analysis/rtc_retained_candidates/M2-G1/candidate.json，阅读入口analysis/rtc_retained_candidates/README.md。
独立保存完整配对配置、校准/协议、原锁、review，以及301个审计输入的归档（含服务器源码）。
audit_bundle.zip SHA256 e709490a3749b16976dd076a4c760aad5b577aa199109d721e8d5d63c4b29766，
逐文件校验通过；仅本地保存，大包不随Git提交。后续须明确其修正、重新集成或有证据替代的去向。
本次没有训练或改动当前5项观测冻结合同；WAITING_FOR_MANUAL_EXPERIMENT与目标blocked保持。
Git按用户既有指示留待手动提交/推送，未声称远端同步。

## 2026-09-15 / 最新：低尾cap有LIE收益但回归失败，父版本参考可靠性等待人工观测

本节优先于下方历史状态。logs/rtc_r3_lower_tail已经是外部Linux完成结果：24/24 completed/exit0/round60，
每项61轮/600客户端，92项runner质量门通过，228份源码和51份计划产物匹配；独立14400条谱参考、
低尾状态、累计、预算与严格配对均通过，本地审计与服务器decision完全一致。38个验收门有3个失败。
审计analysis/rtc_r3_lower_tail_review/verify.py及review.json，原锁a727c6d498f0e2f75c8ccbf022b0807623e51064df7cdc4a66d2f88865d3f966。
原zip 56cf30ef2e81a3a77708dffe425b7f0977be83c2960395803550ba7ad0e9a1e3，服务器原始文件保持不变。

LIE平均ACC两seed由77.5934%升至82.7458%（+5.1524pp），最终87.505%→87.955%；
恶意权重分别25.1162%→3.6%、26.8754%→1.2%，良性误标1/357、0/353。
但Gaussian seed201候选良性误标8/354=2.2599%；Sign-flip seed201父版和候选均4/351=1.1396%。
因此低尾G未接受，保留为有闭环收益、待修正的研究模块。M2仍仅具有历史I12已接受范围，seed201安全转移失败。
Sign-flip round52有6/10攻击者，错误谱参考全由攻击者组成；Gaussian已拒绝的大残差仍污染低尾中位数。
Gaussian seed201 round35同样有6攻击者使raw参考失效，当轮攻击权重31.3449%；不得再外推所有seed权重为零。

下一步先诊断M2参考可靠性：当前Gram不能还原跨轮方向，新增只读全trainable前一聚合位移点积/参考夹角/hash链。
父版rtc_i12_combined，观测版rtc_i12_reference_observe，唯一变化reference_history_observe_only=True；
原R1c/R2与累计/裁剪/回填不变，不增加新cap。原低尾G保持未接受，后续单独修正并重做集成，不丢弃其LIE收益。
固定开发seed201，clean父版/观测2项、Sign-flip父版/观测/MK3项，共5新单元、复用0。
两份trial-plan与旧seed201相同，保留原随机压力；不改抽样使失败轮消失。所有攻击/targeted/非IID最终范围保持。

12项纯合成测试通过；PS语法和5项默认dry-run通过；Bash -n通过，Linux实际dry-run待返回。
新目录logs/rtc_reference_history_observation：251份源码、14份计划产物、5格、status/raw均0。
锁c04a971db980be0d6e92a1b4bb94e950d18527de08745b3a63fdf51089211903；
zip 8f7958244fdb525e4749d90ea9aa0cc43ec87bc42a0161e697b5a396c21c2606。
协议config/rtc_reference_history_observation.json；详细参数、质量门、双平台命令、Git清单和恢复语义：
docs/RTC_LOWER_TAIL_REVIEW_REFERENCE_HISTORY_HANDOFF_20260915.md。
人工Windows启动：powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_history_observation.ps1' -Execute
完成分析：同一命令将-Execute换成-Analyze。Linux仅交接真实服务器路径的无训练dry-run。

本回合为progress：返回的24项终态结果解除旧阻塞，完成拒绝验收/根因诊断及必要新观测准备。
不是verified wait，无已确认远端活动句柄；新人工执行边界连续计数1，宿主要求连续3回合才可正式blocked。
阶段WAITING_FOR_MANUAL_EXPERIMENT。未启动任何训练、不后台等待、不推进依赖新观测的候选。
按用户最新指示由用户手动git提交/推送，命令在交接文档；未宣称GitHub已同步。总体目标未完成。

参考历史观测首次自动续行复核：锁仍为c04a971d…9211903，Windows准备环境，5项计划的status/rounds/raw均0；
本机python/pythonw/raylet进程0，未返回Linux实机验证或训练产物，不能推断外部服务器正在运行。
上一回合为progress，本回合为no progress，无可轮询活动句柄，不记为verified wait。
同一人工执行边界连续计数2，尚未达到正式blocked条件。保持WAITING_FOR_MANUAL_EXPERIMENT；
未重复测试/dry-run、未启动训练、未改冻结合同或推进依赖结果的下一候选。

参考历史观测第二次自动续行复核：同一锁及Windows环境保持，5项计划status/rounds/raw仍各0，
本机python/pythonw/raylet进程0，未收到Linux验证或训练产物；没有已确认的活动句柄。
上一回合与本回合均为no progress，不属于verified wait。同一人工执行阻塞连续3个目标回合成立，
准备工作已完成且没有可独立推进的必要工作，按宿主规则正式将目标标记blocked。
阶段保持WAITING_FOR_MANUAL_EXPERIMENT，等待用户返回验证或实验产物；总体目标未完成。
未启动训练、后台等待、重复测试/dry-run或推进下一候选。

---

## 2026-09-15 / 最新：R3跨轮观测已验收，持续小残差cap等待人工实验

本节优先于下方历史状态。外部Linux服务器10项R3-temporal均completed/exit0/round60，42项质量门通过，
220份源码/25份计划产物完整核验，四对父版本/观测版的模型轨迹、权重、原cap完全一致。
审计analysis/rtc_r3_temporal_review/verify.py及review.json；观测接受，新防御仍未接受，M2保持主线。
原高coherence假设未获支持：部分LIE残差为零导致方向无定义；攻击有效方向持续性反而低于clean。
新线索是攻击的trainable残差相对其他principal显著偏小；按预先写入的clean-only规则校准，不用LIE数据拟合阈值。

候选M2+lower-tail cap：ratio<0.8738694595117251且连续两次参与低尾后q=0；正常/无效观测重置。
参考排除自身principal、至少9个其他正名义质量principal，残差中位数含零值且须>1e-12；重复客户端不重复计数。
clean103校准、clean104验证均0/600标记；旧LIE44/45离线标记163/179、133/150，良性0/321、0/350，首标记round12。
生产模块与独立重放一致；这些是旧轨迹筛选证据，不是闭环ACC收益，非IID/合法中心更新误伤仍待验证。
原R1c/R2 cap、累计参数、裁剪和回填全部保留；父版本仅新增inert低尾observe，候选只将其改为cap。

新批次固定seeds201/202，clean各2项、Sign-flip各3项、Gaussian各4项、LIE .5各3项，共24个新单元，旧基线复用0。
23项合成测试、PowerShell语法/最终dry-run、Bash语法通过；Linux实际dry-run待人工验证，未启动训练。
Windows目录logs/rtc_r3_lower_tail，245份源码/51份冻结产物、status/raw均0，含r3l_sources.zip。
锁SHA256：3d70d7b038c755eaf749a3ea045de16264221f042e58e03a6dba6e6e2ebee807。
源码zip SHA256：39519784664172a5601109cd542946b871e8704422cabd5625f14b0bb8c878e9。
校准canonical hash：666ef4c6c31726fdf206deeed88947c802b433cbf997ba0493a164c82d64ba71。
协议config/rtc_r3_lower_tail_protocol.json；完整参数、门、Git、双平台命令见docs/RTC_R3_TEMPORAL_REVIEW_LOWER_TAIL_HANDOFF_20260915.md。

本回合为progress：完成服务器观测验收、独立clean筛选与新候选准备，旧人工阻塞解除。
阶段WAITING_FOR_MANUAL_EXPERIMENT，新人工边界计数1，尚未达到正式blocked条件。
按用户最新指示留待手动git提交/推送，未宣称已同步；不后台等待，不启动训练，不提前推进下一候选或声称目标完成。

首次自动续行复核：lower-tail锁仍为3d70d7b0…ebee807、Windows准备环境，24项计划的status/rounds/raw均0；
本机未发现python/pythonw/raylet进程，也未返回Linux实机验证证据。
上一回合为progress，本回合为no progress；无可轮询活动句柄，不记为verified wait。
同一人工执行阻塞连续计数2，尚未满足正式blocked条件；未重复测试/dry-run、未训练或推进下一候选。

第二次自动续行复核：lower-tail锁、Windows准备环境保持不变，status/rounds/raw仍各0；本机无python/pythonw/raylet进程，
未返回Linux实机验证或训练证据。上一回合与本回合均为no progress，无活动句柄，不属于verified wait。
同一人工执行阻塞已连续3个目标回合成立，必要准备已完成且没有可独立推进的工作；按宿主规则正式blocked。
阶段保持WAITING_FOR_MANUAL_EXPERIMENT，等待用户返回Linux验证或实验产物；目标未完成，不自动训练或推进下一候选。

---

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
