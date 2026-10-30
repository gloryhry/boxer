# Jetson Orin NX 部署

部署目录为 `ARM:/home/hks/glory/boxer`，连接命令为 `ssh ARM`。
源码来自 `/home/glory/Code/boxer/boxer` 中的 x-nav
地图适配和 ImGui 兼容修复。模型和地图直接复制到 Jetson，不依赖本机的符号链接。

## 环境和文件

设备为 16 GB Jetson Orin NX，Ubuntu 22.04.5，JetPack 6.2.1 / L4T 36.4.4，
系统 CUDA 12.6。Python 3.10 的独立环境位于 `.venv/`。

| 路径 | 内容 |
| --- | --- |
| `.venv/` | ARM64 Python 环境 |
| `.runtime/` | 项目专用动态库、ffmpeg、临时文件及运行缓存 |
| `ckpts/` | BoxerNet、DINOv3、OWLv2 权重 |
| `sample_data/xnav_img/` | 同步的 V6 x-nav 地图参考快照，68 个关键帧、136 张图像 |

## 运行

```bash
ssh ARM
cd /home/hks/glory/boxer

# 完整地图输入检查，不加载模型
bash scripts/run_jetson.sh --input sample_data/xnav_img --validate_input

# 使用 Jetson GPU，先运行三个关键帧
bash scripts/run_jetson.sh --input sample_data/xnav_img --max_n 3 --track

# 处理整张地图
bash scripts/run_jetson.sh --input /path/to/xnav/map --track

# 在 Jetson 桌面会话中回放已保存的结果
bash scripts/run_jetson.sh view_tracker.py --input sample_data/xnav_img \
  --autoplay --xnav_show_tentative
```

普通 SSH 会话可执行输入检查、检测、跟踪和视频导出。交互式窗口需要可用的
桌面显示会话。默认输出为 `output/<map_id>/<snapshot_id>/`；指定
`--output_dir` 可把不同运行的结果保存在不同位置。

`run_jetson.sh` 先执行 CUDA 矩阵乘法、注意力和卷积检查，再启动程序。
它默认设置 `TORCHDYNAMO_DISABLE=1`，使 OWLv2 在 CUDA 上直接推理，避免
ARM 环境依赖 Triton 编译器；可通过环境变量覆盖。CPU 线程默认限制为 4。

该 PyTorch wheel 的通用 CUDA 求逆会遇到系统 cuSOLVER 缺少
`cusolverDnXsyevBatched_bufferSize` 符号的问题。Boxer 的重力对齐原先对固定
正交旋转调用通用求逆，现改为数学等价的转置，避免加载该求解器。修复已同步到
本机与 Jetson。当前部署未验证
额外调用通用 CUDA 线性代数求解器的扩展用法。

地图输入格式、六种相机、V1/V2 标定、坐标系、缓存和输出约定详见
[XNAV_MAP.md](XNAV_MAP.md)。参考地图是室外场景；部署验收用于验证运行链路，
不构成检测准确率评测。默认跟踪确认需要三个不同关键帧的支持，候选框可以通过
`--xnav_show_tentative` 查看。

## 重建环境

使用 JetPack 6.2 / CUDA 12.6 和 Python 3.10，在项目根目录执行：

```bash
bash scripts/setup_jetson.sh
```

脚本只在项目目录安装依赖。精确版本记录在 `requirements-jetson.lock.txt`；
PyTorch 使用 Jetson AI Lab 的 `jp6/cu126` ARM64 wheel，固定下载地址和 SHA256。
该 wheel 所需的 cuDSS 0.7.1.4 来自 NVIDIA 的 Jetson aarch64 独立归档，
安装在 `.runtime/cudss/`，固定 SHA256，并由 `.runtime/lib/` 中的链接加载。
继续使用系统 cuBLAS 12.6；安装 cuDSS 的 Python wheel 会拉入较新的 cuBLAS，
本次实测会导致矩阵乘法报 `CUBLAS_STATUS_ALLOC_FAILED`，所以重建脚本使用独立归档。
不要用面向本机 x86_64 / CUDA 12.8 的 `requirements-local.lock.txt` 重建 Jetson。
`imgui-bundle`、`moderngl`、`glcontext` 在此平台可能需要源码编译；当前系统已具备
Python 开发头文件、C++ 编译器、CMake 以及 EGL/X11 开发库。

此部署面向 x-nav 地图以及无需 Project Aria SDK 的数据入口。
`projectaria-tools` 为 Aria 专用可选依赖，未列入此环境。

依赖来源：[NVIDIA Jetson PyTorch 安装说明](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)、
[Jetson AI Lab CUDA 12.6 wheel 索引](https://pypi.jetson-ai-lab.io/jp6/cu126/torch/)、
[NVIDIA cuDSS 0.7.1 归档校验清单](https://developer.download.nvidia.com/compute/cudss/redist/redistrib_0.7.1.json)。
