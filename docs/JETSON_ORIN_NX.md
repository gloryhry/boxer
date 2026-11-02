# Jetson Orin NX 部署

部署目录为 `ARM:/home/hks/glory/boxer`，连接命令为 `ssh ARM`。
源码来自 `/home/glory/Code/boxer/boxer` 的 `main`（`84db912`）及当前工作区，
包含补充测试、文档、查看器改动和 Jetson Python 3.10 兼容修复。
模型和地图保存在 Jetson 本地，不依赖本机的符号链接。

## 环境和文件

设备为 16 GB Jetson Orin NX，Ubuntu 22.04.5，JetPack 6.2.1 / L4T 36.4.4，
系统 CUDA 12.6。Python 3.10 的独立环境位于 `.venv/`。

| 路径 | 内容 |
| --- | --- |
| `.venv/` | ARM64 Python 环境 |
| `.runtime/` | 项目专用动态库、ffmpeg、临时文件及运行缓存 |
| `ckpts/` | BoxerNet、DINOv3、OWLv2 权重 |
| `sample_data/xnav_img/` | 同步的 V6 x-nav 地图参考快照，68 个关键帧、136 张图像 |
| `output/deployment/` | CUDA、地图读取、测试及推理验收日志 |

## 推理流水线与回退

`bash scripts/run_jetson.sh --input img2 --track` 在 CUDA 上默认使用 `auto` 流水线：
加载线程预取图像、点云与原 CPU 图像预处理结果，单推理线程独占模型，主线程有序
融合/跟踪，`spawn` 输出进程负责 CSV、预览、可靠发布与视频。
从开始推理到可靠提交确认最多两帧在途，提交确认前不释放额度。
运行中 schema 3、完成后 schema 2 和原有缓存参数保持兼容；执行配置单独保存在
`execution` 中。收尾导出失败时不会发布完成状态，已提交前缀仍可读取。

BoxerNet 固定逐相机 batch=1。OWLv2 支持 `--xnav_detector_batch_size 1` 或 `2`，
保持原图像预处理与逐图 NMS；CUDA OOM 自动退回逐图推理并记录回退次数。
默认 OWLv2 batch=1：batch=2 虽通过前 50 帧差分，但在完整 469 帧中
2D 置信度最大差异为 0.00743568，超过 0.002 上限，未通过默认启用门槛。
该探索运行的输出和失败报告完整保留，没有放宽阈值。
跟踪几何在少于 256 条轨迹时使用 CPU，更大规模使用 float32 GPU 投影/点支持，
框面附近的临界判断由原 CPU 实现复核。全局合并保留较早 ID、支持次数及标签更新顺序。

可按运行切换回原串行流程，结果另存以便对照：

```bash
bash scripts/run_jetson.sh --input img2 --track --xnav_execution serial \
  --output_dir output/serial-comparison
```

`--profile` 使用 CUDA events，结束时解析 stream 阶段时间，阶段边界不再全局同步。
单帧延迟、GPU 阶段、队列等待、可靠提交延迟和按提交间隔计算的吞吐分别报告。
关闭 profiling 时不增加 CUDA 计时或同步。正式性能验收关闭 profiling，包含输入检查、
初始化、逐帧输出、最终完整 JSON 与视频编码，按同机顺序运行测量。

源码冻结、完整基准结果和验收证据保存在 `.deployments/optimization-20261102/`。
部署只更新源码；`.venv`、`.runtime`、模型、地图及历史输出保留。

流水线回归包含服务器 350 项用例和 3 个子测试（完整套件 349 项通过，视频编码失败
用例加入 `.runtime/bin` 路径后另行通过）。覆盖六种相机、1000 条以上轨迹、图像/有效域
边缘、旋转框面、空点云、标签变化、队列背压、并发回放与后台进程死亡。
`--cache2d` 的五帧真实 Boxer 推理、`--cache3d` 的五十帧融合/跟踪均通过基准差分。
真实进程组 SIGINT 检查保留了 10 帧可读有序前缀，状态为 `interrupted`。
串行回退的真实推理及缓存重放通过；NVIDIA EGL 离屏回放八帧通过。
本机无 CUDA 的独立回归为 300 项通过、50 项跳过，另有 3 个子测试通过。

### 完整 img2 验收（469 帧，两次顺序运行）

保持 native、960 输入、1220 类、BF16、双相机、跟踪、CSV、预览与视频，关闭
profiling。两次串行基准使用冻结源码；两次优化测量使用相同 Python 源码 SHA256，
默认 OWLv2/BoxerNet 均为 batch=1。GPU load 每 100 ms 采样，比较第 2 帧可靠提交后
至最后一帧提交前的同一关键帧计算窗口；完整耗时包含初始化和视频编码。

| 运行 | 完整耗时（秒） | 平均 GPU 利用率 | GPU <10% 采样比例 | 进程树 RSS 峰值（MiB） |
| --- | ---: | ---: | ---: | ---: |
| 串行 1 | 1272.50 | 32.80% | 62.26% | 2502.30 |
| 串行 2 | 1272.60 | 32.86% | 62.23% | 2469.88 |
| 优化 1 | 904.70 | 58.30% | 33.49% | 2378.68 |
| 优化 2 | 907.40 | 57.89% | 34.14% | 2452.35 |

完整耗时中位数 **1272.55 → 906.05 秒，减少 28.80%**；GPU 低利用率采样比例中位数
**62.25% → 33.82%，下降 28.43 个百分点**，通过至少 10% / 10 个百分点的门槛。
RSS 为进程树驻留内存，不能视为 CUDA 分配峰值。优化运行中系统最低可用内存
为 9388.93 MiB；系统级 swap-in/out 分别为 89/0 和 99/4 页（每页 4096 字节），
只有分散的小量换页，没有持续换页压力。

两次优化结果分别与冻结参考结果检查 **13,254,135 个值**：469 帧全部通过，
2D/3D 坐标、尺寸及置信度差异为零；四元数角差仅有约 0.0000024° 的计算舍入。
逐帧类别、检测数量、轨迹 ID、关联、状态、支持次数和最终别名完全一致，
均为 **7213 个 2D 检测、4320 个融合后 3D 检测和 119 个确认物体**。
两个全局额度的最大实测在途量为 2。
两次运行各有 127 个框的临界点支持判断由 CPU 复核，批量 OOM 回退次数均为 0。
四次运行均生成 4 个 CSV 和 469 张预览图，与参考结果逐字节一致；每段视频均为
1920×960、5 FPS、469 帧，全部帧解码通过，证据为 `output-validation.json`。

逐次原始日志、10 Hz 采样及完整输出位于 `benchmarks/baseline-{1,2}` 和
`benchmarks/optimized-final-{1,2}`；汇总为 `performance-summary.json`，
逐帧差分为 `optimized-final-{1,2}-comparison.json`。
不通过的 batch=2 结果保存在 `benchmarks/optimized-batch2-1` 和
`exploratory-full-comparison.json`，没有用于默认部署的性能结论。
部署前源码另存为 `predeploy-source-*.tar.gz`，部署后的逐文件校验记录为
`deployment.json` 和 `candidate-manifest.json`。

## 同日初次代码更新（2026-11-02）

本次同步原生畸变相机、增量结果保存、批量记录转换、分块跟踪可见性和实时查看器。
默认使用 `native` 相机模式；原有虚拟针孔模式可通过
`--xnav_camera_mode virtual_pinhole` 选择。模型、地图、既有结果与 ARM64 环境保留。

服务器继续使用 Python 3.10。将新流程的 `sys.exception()` 改为兼容的
`sys.exc_info()[1]`，并为异常备注提供 3.10 回退，避免推理结束时出错或清理错误
覆盖原始异常。构造结果存储时同时触发发布与关闭失败的回归测试已先复现、后通过；
本机 Python 3.12 的原生异常备注行为也已验证。

| 项目 | 结果 |
| --- | --- |
| 相关回归测试 | 261 项通过，另有 3 个子测试通过；包含真实 CUDA 相机几何与重力对齐 |
| 完整地图输入检查 | `img2`：469 个关键帧、938 张图像；参考快照：68 个关键帧、136 张图像 |
| GPU 推理 | `img2 --max_n 5 --track --profile`：5 个关键帧、10 个原生相机观测；128 个 2D 检测、51 个融合后 3D 检测、1 个确认物体 |
| 输出检查 | 4 个 CSV、5 张预览图；1920×960、5 fps 的视频共 5 帧，全部解码通过 |
| 回放 | NVIDIA EGL 原生相机模式离屏渲染 8 帧通过，截图为 `viewer-smoke.png` |

本次验证包含五帧模型推理，未对全部 469 帧执行模型推理或进行检测精度评测。
完整验收日志、源码清单和更新前源码备份位于
`.deployments/20261102T092337+0800/`；推理结果位于
`output/deployment-20261102T092337+0800/inference/`。

在服务器上运行完整 `img2` 地图：

```bash
cd /home/hks/glory/boxer
bash scripts/run_jetson.sh --input img2 --track --profile
```

## 历史验收记录（2026-10-30）

| 项目 | 结果 |
| --- | --- |
| 模型与地图同步 | 三个主要权重 SHA256 一致，参考地图逐文件校验一致 |
| CUDA | PyTorch 2.10.0 / CUDA 12.6，Orin SM 8.7，bfloat16 矩阵乘法、注意力及卷积通过 |
| 输入检查 | 68 个关键帧、136 张保存图像、相机 0 和 1，完整地图检查通过 |
| 回归测试 | x-nav、跟踪、融合、文件读写、重力对齐及 ImGui，共 122 项通过 |
| GPU 推理 | `--start_n 2 --max_n 3 --track`：3 个关键帧、6 张图像、30 个虚拟视图；14 个 2D 检测、6 个融合后 3D 检测 |
| 跟踪输出 | 默认确认条件下，已确认物体为 0；回放可显示候选框 |
| 视频 | H.264，1920×960，5 fps，3 帧均能解码 |
| 回放 | NVIDIA EGL 离屏渲染 8 帧通过，地图、图像、候选框及 ImGui 显示正常 |

这次执行了完整地图输入校验和 3 个关键帧的模型推理，未在 Jetson 上执行
全部 68 个关键帧的模型推理，未验收桌面鼠标键盘交互或检测精度。
主要证据为 `output/deployment/validation.json`、`pytest_final.xml`、
`xnav_inference.log`、`video.json` 和 `xnav_viewer_smoke.png`。
源码快照、基准 Git 提交与文件 SHA256 保存在 `output/deployment/source_manifest.json`。

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
本机与 Jetson，并增加了真实 GPU 与 CPU 位姿结果对比测试。当前部署未验证
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
