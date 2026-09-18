# M3-G2：保留待修正

2026-09-18独立复核32项Linux实验，质量通过，60候选门58通过。
LIE两seed攻击期平均ACC对M3提升4.0156pp，最终提升1.20pp；Gaussian低尾良性误标相对G1-R从9/6降至0/0。
Sign-flip seed201良性误标7/351（1.9943%）超过1%，平均ACC差−.2058pp低于−.2pp门，因此本批未接受、未晋升M4。

该收益和失败一并保存，不删除机制、不追改门槛。按用户最新要求，先完成冻结M3的单seed全攻击结果表；之后再研究Sign误伤的修复。
M3仍为M2+H2b confirmed参考历史；本次RTC全攻击批次不启用G2。

candidate.json记录来源与快照哈希。audit_bundle.zip包含374项冻结及JSON/CSV证据（含服务器源码zip），只在本地，Git忽略；同步Git不等于备份该大包。
审计入口：analysis/rtc_g2_integration_review/verify.py；审计输出：review.json。
