# 本机 NVIDIA CUDA 运行环境

项目源码：`/home/glory/Code/boxer/boxer`。

本机配置：Ubuntu 24.04.5、Python 3.12.3、NVIDIA GeForce RTX 5060 Laptop GPU
（8151 MiB）、驱动 595.91.07。安装的是 `torch==2.10.0+cu128`，使用它自带的
CUDA 12.8 运行库，无需另外安装系统 CUDA Toolkit。

## 文件位置

安装开始时系统盘只剩约 7.4 GB，因此将较大的依赖和数据放在本机另一个已挂载分区。
项目内通过符号链接访问它们：

| 项目内路径 | 实际位置 |
| --- | --- |
| `.venv` | `/media/glory/Windows/boxer-runtime/.venv` |
| `ckpts` | `/media/glory/Windows/boxer-runtime/ckpts` |
| `sample_data` | `/media/glory/Windows/boxer-runtime/sample_data` |

运行时需保持 `/media/glory/Windows` 挂载。模型和数据的来源版本、文件大小及大文件
SHA256 保存在 `/media/glory/Windows/boxer-runtime/assets-manifest.json`。
已下载三个官方模型权重及 `nym10_gen1` 示例序列。
Python 环境约 8.3 GB，模型约 1.2 GB，示例和读取缓存约 117 MB。

## 启动

在本机终端执行：

```bash
cd /home/glory/Code/boxer/boxer

# GPU 运算检查：矩阵乘法、注意力、卷积
.venv/bin/python scripts/check_cuda.py

# 官方无窗口示例：前 90 帧，启用在线 3D 跟踪
bash scripts/run_local_cuda.sh --input nym10_gen1 --max_n 90 --track

# x-nav 地图边推理边显示 3D 窗口
bash scripts/run_local_cuda.sh \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img2 --track --live_viz

# 交互式检测和 3D 查看器，需要桌面显示环境
bash scripts/run_local_cuda.sh view_prompt.py --input nym10_gen1

# 查看上一次推理结果中的在线跟踪
bash scripts/run_local_cuda.sh view_tracker.py --input nym10_gen1 --autoplay
```

启动脚本会检查 CUDA 运算是否成功；不可用时直接报错。模型在此显卡上自动使用
bfloat16。首次运行会编译 OWLv2 的 PyTorch/Triton 内核，耗时比后续启动长。
编译缓存保存在 `/media/glory/Windows/boxer-runtime/cache`。
默认使用 4 个 CPU 运算线程、1 个编译线程，减少首次编译的内存压力；可通过
`OMP_NUM_THREADS`、`MKL_NUM_THREADS` 和 `TORCHINDUCTOR_COMPILE_THREADS` 覆盖。

默认结果目录为 `output/nym10_gen1/`：

- `owl_2dbbs.csv`：每帧 2D 检测框。
- `boxer_3dbbs.csv`：每帧 3D 检测框。
- `boxer_3dbbs_tracked.csv`：在线跟踪结果。
- `boxer_viz_current.jpg`：最后处理帧的可视化。
- `boxer_viz_final.mp4`：完整可视化视频。

## 重建 Python 环境

`requirements-local.lock.txt` 记录本次安装的精确依赖版本。
下面的命令安装到同一专用目录；已有环境可直接跳过创建步骤。

```bash
cd /home/glory/Code/boxer/boxer
runtime_dir=/media/glory/Windows/boxer-runtime
mkdir -p "$runtime_dir/tmp"
TMPDIR="$runtime_dir/tmp" uv --no-cache venv "$runtime_dir/.venv" --python /usr/bin/python3.12
TMPDIR="$runtime_dir/tmp" uv --no-cache pip install \
  --python "$runtime_dir/.venv/bin/python" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r requirements-local.lock.txt
```

如果从全新源码目录开始，先创建上述三个实际目录，再建立对应链接；项目自带的
`bash scripts/download_ckpts.sh` 和 `bash scripts/download_aria_data.sh nym10_gen1`
可下载模型和示例。当前目录已完成这些步骤。

## GPU 访问说明

在隔离的工具进程中可能看不到 `/dev/nvidia*`，此时即使宿主机驱动正常，
`nvidia-smi` 和 PyTorch 也会报 CUDA 不可用。本机终端应能直接访问显卡。
本次 GPU 验证使用了能访问宿主机 NVIDIA 设备的进程。

## 验证记录

2026-10-28 在本机实际验证：

| 检查 | 结果 |
| --- | --- |
| Python 依赖一致性 | `uv pip check` 通过，155 个包 |
| CUDA 实际运算 | bfloat16 矩阵乘法、注意力、卷积全部通过 |
| 显卡架构 | compute capability 12.0 |
| OpenGL 3.3 独立上下文 | NVIDIA GeForce RTX 5060 Laptop GPU，创建成功 |
| 项目入口和数据/图形依赖 | 四个入口脚本及 Aria、OpenCV、查看器依赖导入成功 |
| 官方样例 | `--input nym10_gen1 --max_n 90 --track`，退出码 0 |
| 模型设备和精度 | CUDA，OWLv2 使用 bfloat16，BoxerNet 权重 320/320 全部加载 |
| 帧数 | 90/90，2D 和 3D CSV 均覆盖 90 个时间戳 |
| 检测结果 | 2038 条 2D 检测、1848 条 3D 检测、60 条最终跟踪结果 |
| 结果数值 | 数值全部有限、置信度在 [0, 1]、3D 尺寸为正 |
| 视频 | H.264，2880×960，10 FPS，90 帧，9 秒 |

日志确认 `Using device cuda` 和 `Loaded OWLv2 on cuda ... precision=bfloat16`；
运行期间 `nvidia-smi` 也观察到本项目 Python 进程占用显卡。
帧处理循环约 19 秒，包含检测、3D 提升、跟踪和逐帧可视化；不包含首次模型加载、
内核编译和最后的视频编码时间。

完整日志：`output/local_cuda_demo.log`。
CSV 数值检查结果：`output/local_cuda_validation.json`。
查看器进一步完成 NVIDIA 桌面窗口验证：自动执行 `light` 提示，得到 1 个 2D 检测，
成功提升并接受 1 个 3D 框，实际绘制 6.0 和 3.0 像素线宽的矩形，连续渲染 8 帧后
正常关闭（退出码 0）。日志：`output/local_cuda_gui.log`；
结果：`output/gui_prompt_validation.json`；窗口图像：`output/gui_prompt_smoke.jpg`。
日志中的自动调优提示和 NumPy 弃用警告没有阻止此次样例完成。

## `add_rect` 界面兼容修复

本机安装的 `imgui-bundle==1.93.0` 使用新版 Dear ImGui 绘图接口。Dear ImGui 从
1.92.8 起将 `add_rect` 最后两个参数从 `flags, thickness` 调换为
`thickness, flags`。项目原适配层按旧顺序传递位置参数，使浮点线宽（如 `6.0`）
进入整数 `flags` 参数，触发 `TypeError`。CUDA 推理和 OpenGL 初始化均已成功，
此前无窗口推理没有经过这段界面绘制代码。

已在 `utils/imgui_compat.py` 中改用 `rounding=...`、`thickness=...`、`flags=...`
关键字传参，保留上层调用接口。依赖版本保持不变。
新增三个直接调用真实 imgui-bundle 绑定的回归测试，覆盖默认线宽、查看器使用的
浮点线宽，以及圆角标志和线宽对应的实际顶点/索引几何。修复前复现相同异常，
修复后全部通过：

```bash
.venv/bin/python -m unittest discover -s tests -p test_imgui_compat.py -v
```

接口变更说明：[Dear ImGui Draw List](https://github.com/ocornut/imgui/wiki/Draw-List#rectangles)。

参考：[Boxer 官方安装与示例](https://github.com/facebookresearch/boxer)、
[PyTorch 官方 CUDA 12.8 安装版本](https://pytorch.org/get-started/previous-versions/)。
