# H2服务器dry-run客户端参数不一致修复

用户在Linux准备H2时，prepare中的完整client字典断言失败；这发生在dry-run，不能解释为训练失败或候选失败。
堆栈只能确定客户端配置不符合预登记，尚不能确定服务器具体哪一字段不同。
num_gpus不在该client字典内，本次修复不更改GPU资源配额。

原prepare从主config/config.yaml继承local_epochs、optimizer、learning_rate、momentum、weight_decay和lr_scheduler，
命令行只固定batch_size，因此服务器主配置与本地不同时会失败，而且裸断言不显示具体差异。
现在H2显式生成自己的client运行配置：local_epochs5、batch_size48、sgd、lr.01、momentum.9、weight_decay.0001、cosine。
这些是已预登记参数，不是更改实验假设或放宽验收。主config/config.yaml不修改，旧结果/源码锁不覆盖。
preparation_client_parameters.json保存原主配置client值与固定值；运行配置写入后先做校验，逐spec仍校验。
若仍有差异，ValueError打印字段名、expected、actual及实际类型，不能删除检查绕过。

9项纯合成测试通过，包括主配置故意改成adam/不同epochs、lr等仍恢复预登记参数，
GPU配置保留、错误详情、失败准备目录保留、完整聚合与600条合成日志重放；没有真实训练。
旧失败目录通常已有runtime_config.yaml和部分manifest，但还没有r1e_lock.json；直接在原目录重试会被拒绝。
请保留它诊断，使用新的rtc_reference_eligibility_configfix目录，不删除或覆盖旧日志。

## 手动同步

本地修改只有H2准备脚本、相关测试及交接文档；按用户既有要求由用户手动提交和推送。
先确认此前H2交接的全部代码已提交，且暂存区没有其他工作。

```powershell
Set-Location 'D:\workspace\FL2'
git add -- experiments/rtc_reference_eligibility_stage.py tests/test_rtc_reference_eligibility_stage.py
git add -f -- docs/RTC_H2_DRY_RUN_CONFIG_FIX.md docs/RTC_H1_REVIEW_H2_HANDOFF_20260916.md docs/RTC_BYZANTINE_GOAL_STATE.md docs/RTC_BYZANTINE_GOAL_PROMPT.md
git diff --cached --stat
git diff --cached --check
git commit -m "Pin H2 client training recipe independently of host defaults"
git -c http.proxy=http://127.0.0.1:17897 -c http.sslBackend=openssl push origin codex/periodic-attack-defenses
```

服务器手动更新当前分支，若有本地GPU等修改，先检查差异并保留，不使用reset/强制覆盖：

```bash
git -C /home/jia_zhang/hqr/FL2 pull --ff-only origin codex/periodic-attack-defenses
```

## 服务器重新准备（不训练）

```bash
bash /home/jia_zhang/hqr/FL2/experiments/run_rtc_reference_eligibility.sh --python /home/jia_zhang/miniconda3/envs/fl2/bin/python3.11 --data-dir /home/jia_zhang/hqr/FL2/data --output /home/jia_zhang/hqr/FL2/logs/rtc_reference_eligibility_configfix
```

本次仍为原H2的24项、seeds201/204、原攻击与验收门，不是额外一批算法实验。
预期最后输出stage=reference-eligibility、training_units=24，并生成新目录下r1e_lock.json。
返回实机dry-run输出及新锁后再核验并交接Linux真实启动命令；此处不授权或启动训练。
后续执行和分析必须继续使用同一个--output；不能省略后回到旧失败目录。

Windows等价无训练准备入口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\workspace\FL2\experiments\run_rtc_reference_eligibility.ps1' -Output 'D:\workspace\FL2\logs\rtc_reference_eligibility_configfix'
```

当前WAITING_FOR_MANUAL_EXPERIMENT，具体为等待Linux实机准备通过。M2-G1保留义务与最终目标范围不变。

本地验证结果：上述Windows入口24项dry-run通过，267份源码/52份冻结产物全部核验，训练产物0。
24项client参数及trial-plan与原本地准备完全一致；旧锁4dd7b1ca…22c1a保持原样。
新Windows锁e37374a699be1830ed4b489e4233303a685918ced990324633179c55a1cf8e55；
源码zip 98f10ad43db134e06be0000b91ab65d8409ed1602b0028d278f91af044460d12。
Linux实机尚待用户验证，不复用Windows锁。
