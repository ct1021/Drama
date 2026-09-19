# Drama 可迁移环境

此分支只管理运行环境与环境验收，起点是上游提交 `a50bd54c34e77d1d13e988a031733a47817098e2`。不包含历史工作目录中未提交的研究改动，不自动启动训练。

## 放在哪里

- Git：环境脚本、版本约束、验收工具和不含机器身份信息的说明。
- 本地 `.ssh/drama-research/`：私钥、实例地址、端口、预算与主机指纹，不上传 Git。
- 服务器数据盘：虚拟环境、编译缓存、训练数据和检查点。
- 每台服务器单独保留验收记录；换服务器后重新验证，不凭镜像名称判断成功。
- GitHub 网络不可达时，可从可信本机传输代码和项目官方 Release wheel；源码仍按提交同步，wheel 留在服务器数据盘 `wheels/`，不提交二进制或凭据。

## 换服务器流程

1. 将 SSH 地址和端口保存到本地连接配置，使用研究专用公钥登录。
2. 在基础 Python 中运行 `python ops/probe.py`，检查显卡、开发工具、磁盘与已安装组件。该命令不会安装或训练。
3. 根据检查结果，在数据盘创建独立环境。候选脚本是 `bash ops/bootstrap.sh`，遵循上游 README 的 Python 3.12 / PyTorch 2.7.0 cu128 / Mamba 2.2.6.post3 组合；现有镜像环境保留。先完成服务器探测，再运行安装。`DRAMA_DATA_DIR`、`DRAMA_ENV_DIR`、`DRAMA_BASE_PYTHON` 可覆盖默认路径。
4. 在仓库根目录执行 `python ops/verify_runtime.py`。它只检查 GPU 扩展前反向、递归缓存、Drama 模块和 Atari 环境，不更新策略、不生成研究回报。
5. 保存 `pip freeze`、`pip check`、代码提交与验收输出；只有实测通过的组合才标记为已验证。

普通终端使用 `source ops/activate.sh` 激活环境。无人值守 SSH 的初始 PATH 可能不含镜像 Python/CUDA，可在安装前显式设置 `PATH=/root/miniconda3/bin:/usr/local/cuda-12.8/bin:$PATH` 和 `CUDA_HOME=/usr/local/cuda-12.8`。这些是本次 AutoDL 镜像实际核验路径，更换平台时先重新检查。

国内服务器可通过 `DRAMA_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` 使用清华镜像；默认 PyPI 官方源，PyTorch CUDA 包仍从 PyTorch 官方索引安装。源地址与实际版本应随验收记录保存。

## 当前状态

2026-09-19：已建立独立分支，安装组合和验收脚本均未在目标服务器验证。本文件不表示环境已配置完成。`constraints-cu128.txt` 为候选版本约束；安装后生成的完整 `pip freeze` 才记录实际解析出的依赖。

已修复上游 `train.py` 将实际 `envs/my_dmc.py` 写成 `env.my_dmc` 的路径问题，并只在选用 DMC 时导入。Atari 环境不再因可选的 DMC 依赖而无法导入训练入口。该改动不涉及模型或训练配方，DMC 本身不在此次验收范围。

Drama 包含自己的 `mamba_ssm/` Python 源码，其中有 `MambaWrapperModel` 等定制接口。因此验收必须从仓库根目录执行，核对模块来源；仅成功导入 PyPI 的 Mamba 不够。

## 验收范围

环境验收还包含一个小批量 Drama 想象调用，使用默认 CUDA Graph，且不执行优化器更新。通过只说明被检查的计算路径可用，不代表完整训练协议、所有批量/精度、torch.compile 或研究假说已验证。正式训练另行确定配置和预算。
