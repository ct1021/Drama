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
3. 根据检查结果，在数据盘创建独立环境。安装脚本为 `bash ops/bootstrap.sh`，本分支使用 Python 3.12 / PyTorch 2.7.1 cu128 / Triton 3.3.1 / Mamba 2.2.6.post3；现有镜像环境保留。先完成服务器探测，再运行安装。`DRAMA_DATA_DIR`、`DRAMA_ENV_DIR`、`DRAMA_BASE_PYTHON` 可覆盖默认路径。可提前将已校验 wheel 放入数据盘 `wheels/`，对应来源与 SHA256 见 `wheels-cu128.json`；SM120 自编译 wheel 放在 `built-wheels/`。不要将这些目录提交 Git。
4. 在仓库根目录执行 `python ops/verify_runtime.py`。它只检查 GPU 扩展前反向、递归缓存、Drama 模块和 Atari 环境，不更新策略、不生成研究回报。
5. 保存 `pip freeze`、`pip check`、代码提交与验收输出；只有实测通过的组合才标记为已验证。

普通终端使用 `source ops/activate.sh` 激活环境。无人值守 SSH 的初始 PATH 可能不含镜像 Python/CUDA，可在安装前显式设置 `PATH=/root/miniconda3/bin:/usr/local/cuda-12.8/bin:$PATH` 和 `CUDA_HOME=/usr/local/cuda-12.8`。这些是本次 AutoDL 镜像实际核验路径，更换平台时先重新检查。

国内服务器可通过 `DRAMA_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` 使用清华镜像；默认 PyPI 官方源，PyTorch CUDA 包仍从 PyTorch 官方索引安装。源地址与实际版本应随验收记录保存。

## 当前状态

2026-09-19：已完成 RTX 5090 上的独立环境安装。验收结果、实际依赖版本与测试提交见 `validated/20260919-rtx5090/`。`constraints-cu128.txt` 控制主要版本；实测完整版本清单记录在 `versions.txt`，其中 `causal-conv1d==1.5.2+sm120` 需要本分支构建，不能直接从 PyPI 安装。新机器仍需重新验收。

已修复上游 `train.py` 将实际 `envs/my_dmc.py` 写成 `env.my_dmc` 的路径问题，并只在选用 DMC 时导入。Atari 环境不再因可选的 DMC 依赖而无法导入训练入口。该改动不涉及模型或训练配方，DMC 本身不在此次验收范围。

Drama 包含自己的 `mamba_ssm/` Python 源码，其中有 `MambaWrapperModel` 等定制接口。因此验收必须从仓库根目录执行，核对模块来源；仅成功导入 PyPI 的 Mamba 不够。

已为 `InferenceParams` 补齐可选的 `key_value_dtype=None` 字段：仓库中的 Mamba2 已经读取它，Drama 非 CUDA Graph 路径也已传入它，但原数据类缺少声明。字段追加在末尾，保留原有参数位置与默认 dtype 选择，不修改模型计算或研究架构。

## RTX 5090 的扩展兼容处理

实测 Triton 3.3.0 在该设备上报 `computeCapability not supported`。采用 PyTorch 2.7.1 对应的 Triton 3.3.1，避免强行拆开其依赖关系。上游 SM120 修复来源：[Triton PR #6771](https://github.com/triton-lang/triton/pull/6771)。因此本分支的 Torch 补丁版本与上游 README 的 2.7.0 不同；不是研究方法改动。

实测官方 causal-conv1d 1.5.2 预编译 wheel 在 SM 12.0 上报 `no kernel image`。其源码默认目标最高只包含 SM 10.0，单独设置 `TORCH_CUDA_ARCH_LIST` 无法覆盖源码中显式写出的编译目标。

`build_causal_sm120.sh` 使用已校验的 1.5.2 源码，将编译目标设置为 SM 12.0，并将本地版本标为 `1.5.2+sm120`，不修改 `csrc/` 计算代码。构建出的 wheel 和校验值保存在数据盘 `built-wheels/`。`bootstrap.sh` 在检测到 SM 12.0 时自动使用这一路径；其他设备仍需自行验收，不宣称覆盖全部 GPU。

官方 Release wheel 的文件名还包含 CUDA/Torch/ABI 后缀，而包内版本没有相同后缀。`prepare_wheels.py` 先核对原文件 SHA256，再为其创建与包内版本一致的文件名；原文件和二进制内容均保留。

## 验收范围

环境验收还包含一个小批量 Drama 想象调用，使用默认 CUDA Graph，且不执行优化器更新。通过只说明被检查的计算路径可用，不代表完整训练协议、所有批量/精度、torch.compile 或研究假说已验证。正式训练另行确定配置和预算。

递推一致性检查使用 8 帧预填充上下文，再进行 8 次单步缓存递推，对比 16 帧整段计算（float32，容差 1e-3）；默认与显式 float32 缓存构造均覆盖。测试模型维度使用 256：之前过小的 128 维测试模型只有 4 个头，使投影后的通道步幅不满足底层卷积的 8 元素对齐要求；这不是上下文长度导致的问题。DMC、完整优化器循环及其他精度/长度组合未验收。
