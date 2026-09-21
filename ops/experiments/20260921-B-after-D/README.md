# B授权接续与检查范围

2026-09-21。用户明确“允许，请在d完成训练后加入b”。本文件替代旧文件中“不得追加B”及“C/D完成就暂停检查”的限制；E和其他种子仍未授权。

## 当前安排

B已实现、通过本地及服务器13项CPU检查，已部署独立候选版本 `f401807975d9ad050c61f795e09a5b4b7e27f2ef`。GPU验收尚待D结束后执行，不与D竞争GPU。

2026-09-21 22:51已启动服务器接续程序，外层PID26997，协调程序PID26998，核对命令身份与状态为 `waiting_for_D`。每20秒检查一次D；D成功完成、进程退出、C/D协调器完成、GPU空闲后直接进入B验收，无需用户夜间守候。

只执行一个B：Boxing、seed3710、100K从头训练；原轻量架构，3个额外训练标量；不沿用C/D权重。公式与限制见30号文件及仓库ops/B-CONTROL.md。保持图像、奖励、KL原始目标与detach；相同学习率，标量无weight decay，加入共同梯度裁剪。没有安装依赖、修改D运行版本。

## 范围与终止条件

1. D必须为原5829fc8代码、正确任务和种子、成功100K、完整10回合终点评测。D失败或到原7小时上限后仍未完成，B不启动。
2. D完成后先保存其核心文件大小与SHA256至B协调目录的D-completed-evidence.json；小时检查负责独立Release归档和公开下载恢复，不阻塞GPU接续。
3. B GPU/1800步短验收合计最多15分钟：模型初始化/优化器覆盖、梯度有限、并行与缓存、真实决策、CUDA Graph想象、标量更新和保存、实际训练耗时。1200–1700步耗时比原A短训练不超过1.15才通过。使用同一冻结代码启动正式训练。
4. B正式硬上限7小时，合作上限6.9小时。任何环节失败均保留日志并停止，不调整阈值、不重启、不追加种子。
5. 接续进程总外层16小时限额包含等待；实际GPU预算仍只有15分钟验收+7小时正式。进程结束不会停止实例计费。当前单价未知。

## 唯一路径与连接

远端根目录 `/root/autodl-tmp/drama-research-20260919`。

- D：`runs/boxing-h1-routed-s3710-100k-20260921`，代码 `candidates/5829fc831febdbd844506f9d87906e6c0b348cbb`。
- B候选代码：`candidates/f401807975d9ad050c61f795e09a5b4b7e27f2ef`。
- B协调状态：`runs/b-after-d-s3710-20260921/status.json`，外层日志 `exports/B-after-D-console.log`。
- B验收：协调目录 `preflight/`，结论 `preflight/gate.json`，GPU记录 `preflight/gpu.json`，短训练 `preflight/training/`。
- B正式：`runs/boxing-optimization-control-s3710-100k-20260922`。

只用本地 `.ssh/drama-research/clone-20260921/connection.json`、其中指定私钥和clone known_hosts；BatchMode=yes、StrictHostKeyChecking=yes，不读写/展示密码或私钥。

## 提醒与归档

现有自动检查每15分钟执行 `environment-clone-20260921/h1_hourly_check.py`，该脚本只读C/D/B与协调状态，所有快照仍在h1/hourly-checks。正常每小时简报；阶段转换、完成、失败、明显停滞、磁盘不足立即在首次发现时通知。写h1/notification-state.json记录用户可见汇报时间。不能因C/D协调器completed就停掉B监控。

监控端不得另行启动B，服务器唯一接续程序负责；先核对协调状态和进程身份，禁止重复提交。如果前置程序退出或验收未过，保存全部证据并告知用户，不自行修改运行版本。

C已归档，不重复发布。D完成用现有 `archive_h1_routed.py build/publish/verify` 和 `finalize_h1_archive.py routed`。B完成用 `archive_h1_optimization.py build/publish/verify` 和 `finalize_h1_archive.py optimization`。这些本地脚本均位于environment-clone-20260921，不启动训练。下载缓慢可先用 `fetch_h1_parts_bounded.py routed` 或 `optimization`，再运行verify恢复校验。完整日志/指标/权重在独立Release，配置、评测、恢复清单提交研究分支；旧归档不变。

另外保存B协调日志、GPU验收及1800步记录至研究分支，核对其中代码提交。D和B均终态且证据保存后，才暂停检查。失败则保留实际步数和已有权重，不能声称完成或精确续训。

预计D于9月22日02:45–03:30结束；若验收通过，B随后运行，预计08:00–10:00完成。更新时间预估以实时速度为准，不是固定截止承诺。B比较A/D的100K终点、60K–100K均分、全程耗时，不因单个阶段好看宣称有效。
