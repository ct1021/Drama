# 已完成实验的保存与迁移

本说明对应 Boxing / seed 3710 / 100K（2026-09-19），公开配置世界模型
21,925,667 参数，终点 10 回合均分 32.9。候选架构没有实现或训练。

## 两层归档

- 分支 `research/atari100k-baseline`：代码、解析配置、运行状态、评测、控制台日志、
  小时检查摘要、校验清单与迁移工具。固定迁移版本标签为
  `boxing-s3710-100k-portable-v1`，后续分支变化不影响这个版本。
- [实验 Release](https://github.com/ct1021/Drama/releases/tag/boxing-s3710-100k-20260919)：
  完整标量日志、最终世界模型与策略权重、原始训练代码快照、实际依赖版本、
  环境验收记录和已编译的 SM120 卷积包。共 141,394,521 字节，分为 12 块，
  下载程序会校验、拼接；完整文件清单见实验目录 `archive-files.json`。

训练源代码固定为 `d4b2182c18bfb7a4d152f2fc76c16435cd0aa16e`。
Release 标签指向这一训练提交；迁移标签增加的只是归档工具和记录。
大文件放在 Release，分支中的 `archive.json` 记录精确地址、大小和 SHA256。
无需旧服务器开机，不使用旧 SSH 凭据。归档不含密码、私钥或 GitHub 令牌。

## 新实例：先恢复资料

数据恢复只需要 Git 和 Python 3.11+，不安装依赖、不启动训练。以下是在新 Linux
实例上的示例；目录可更换，但应放在空间充足的数据盘上。

```bash
git clone --branch research/atari100k-baseline https://github.com/ct1021/Drama.git Drama
cd Drama
git checkout boxing-s3710-100k-portable-v1

DRAMA_MIGRATION_ROOT=/root/autodl-tmp/drama-migrated
python ops/download_experiment.py \
  --manifest experiments/boxing/20260919-seed3710-100k/archive.json \
  --destination "$DRAMA_MIGRATION_ROOT/download"
python ops/restore_experiment.py \
  "$DRAMA_MIGRATION_ROOT/download/boxing-s3710-100k-d4b2182.tar.gz" \
  --sha256 32e8a991c7880b37bccb6d82ef5325351ba911da475c5bfbec7f9dc055fe3ea0 \
  --destination "$DRAMA_MIGRATION_ROOT/restored"
```

下载中断后可重跑下载命令，已经通过校验的分块会复用。解包目标必须为空；
成功后生成 `RESTORE-VERIFIED.json`。如果解包中断，请选择新的空目录重试。

恢复后的内容：

| 目录 | 用途 |
|---|---|
| `restored/experiment/` | 原始配置、状态、评测、完整 metrics.jsonl、console.log |
| `restored/experiment/ckpt/` | world_model-final.pth、agent-final.pth |
| `restored/code/` | 训练时全部已跟踪代码的精确快照 |
| `restored/environment/` | 实际版本清单、硬件与运行验收、预编译包 |

## 需要 GPU 运行时，再显式配置环境

已验证的硬件软件组合：RTX 5090，Linux x86_64 / Ubuntu 22.04，Python 3.12，
CUDA Toolkit 12.8，驱动 580.76.05；PyTorch 2.7.1+cu128、Triton 3.3.1、
Mamba 2.2.6.post3、causal-conv1d 1.5.2+sm120。其他 GPU/系统需另做兼容验收。
新镜像还需提供 Git、curl、g++ 和 Python venv。数据盘建议预留至少 20 GB。

在上述仓库目录中、确认使用 Python 3.12 后：

```bash
bash ops/install_restored_env.sh \
  "$DRAMA_MIGRATION_ROOT/restored" "$DRAMA_MIGRATION_ROOT/runtime"

export DRAMA_DATA_DIR="$DRAMA_MIGRATION_ROOT/runtime"
source ops/activate.sh
```

该入口只安装独立环境并进行运行检查，不启动训练。它会复用已编译的 SM120 包，
按记录的 SHA256 获取 PyTorch/Mamba 等官方 wheel，并以实际版本清单约束依赖。
不覆盖既有环境；安装失败后保留日志，先核对失败原因。

141 MB 实验包已实际完成公开下载和本地还原；新 GPU 实例的环境重建尚未实测。
这不是完整离线镜像：PyTorch、CUDA 依赖等数 GB 文件仍需联网下载，速度依网络而定。
版本清单包含原环境的可选包，安装入口保证所选运行路径的版本约束，不宣称复制整个系统。

## 已验证范围与续训边界

已从发布后的 GitHub Release 无凭据下载全部 12 块，校验整体 SHA256，
在本地新目录还原并逐文件核对 105 项。8 个核心实验文件还单独对照旧服务器
原始 SHA256；完整日志有 1,386,075 条标量记录，终点 100000，未见非有限值。
每 10K 的 10 回合评测共 100 回合均已保留。详见实验目录 `verification.json`。

保存的是最终模型权重，不含回放缓冲区、优化器、随机数状态或环境快照。
可迁移权重用于评测、分析及另行设计后续实验，不能声称精确断点续训。
训练中每 2K 的 latest 权重会覆盖，历史全部检查点并不存在；全部评测记录仍保留。
本次的模型、策略、日志和配置已有独立归档，不再依赖旧实例保留；
这项确认只覆盖本实验，不覆盖旧实例上可能存在的其他工作。
