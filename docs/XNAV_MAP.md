# x-nav 地图包：3D 检测、静态物体关联与回放

程序直接读取 x-nav 导出的地图目录，无需转换为 ScanNet 或录制 ROS bag。
使用现有 OWLv2、BoxerNet 权重检测物体，并在地图坐标系中关联静态物体。
读取 x-nav 标定需要 PyYAML，本机环境已安装；其他数据集入口不增加此依赖。

## 快速使用

```bash
cd /home/glory/Code/boxer/boxer

# 只检查输入：不加载模型、不生成输出，不要求 CUDA
.venv/bin/python run_boxer.py \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img --validate_input

# 全部相机、原生图像检测、逐关键帧跟踪和视频
bash scripts/run_local_cuda.sh \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img --track

# 边推理边查看：自动打开 3D 窗口，跟随最新完成的关键帧
bash scripts/run_local_cuda.sh \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img2 --track --live_viz

# 回放保存的状态；前后跳帧不会重新分配 ID
bash scripts/run_local_cuda.sh view_tracker.py \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img --autoplay

# 同时查看尚未满足确认条件的候选物体
bash scripts/run_local_cuda.sh view_tracker.py \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img \
  --xnav_show_tentative
```

`--input` 自动根据地图清单识别 x-nav 数据。原有 Aria、ScanNet、CA-1M、
Omni3D 入口保持原来的处理方式。只读检查和 x-nav 回放也可直接使用
`.venv/bin/python`，无需经过 CUDA 启动检查脚本。

## 交互式文本提示与鼠标框选

无需先运行批处理，也无需 `results.json`，可以直接打开地图包：

```bash
bash scripts/run_local_cuda.sh view_prompt.py \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img

# 只启用指定相机；默认 all
bash scripts/run_local_cuda.sh view_prompt.py \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img --xnav_cameras 0

# 切回原有的多方向虚拟针孔视图
bash scripts/run_local_cuda.sh view_prompt.py \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img \
  --xnav_camera_mode virtual_pinhole
```

默认 `--xnav_camera_mode native`，图像面板保留原始畸变，每路相机只有一个
`native` 视图；左侧选择相机，底部切换关键帧。`virtual_pinhole` 模式额外提供虚拟视图选择器。
可输入英文类别并点击 `Detect`，或在图像内按住左键拖出 2D 框。
文字检测和框选只作用于当前视图，推理开始时暂停播放。
初始 2D 阈值为 0.2，3D 阈值为 0.6，可在界面调整。

推理几何来自当前关键帧的 `rgb`/`surf` 点云，按当前视图筛选后最多 10,000 点。
`global_map.pcd` 作为 3D 背景显示；深度点和深度块叠加显示的是实际推理点云。
切换关键帧、相机或视图时取消待处理提示，已接受的 3D 框在地图坐标系中保留，
并重新投影到当前图像。可用 `Undo Last` 撤销、`Clear All` 清空。
交互结果保留在当前窗口中，关闭后不恢复；不新增导出或自动跟踪功能。

索引中没有图像的关键帧显示空观测并禁用提示，仍可切换相机或关键帧。
两种模式的填充及无效区域均不能框选，检测框必须具有有效图像支持。
原生模式缓存缩放填充及有效掩码；虚拟针孔模式缓存重映射。

提示界面的 `--start_n` 从 0 计数，`--skip_n`/`--max_n` 按关键帧计数；
`--max_n 0` 表示不限制数量。原有 Aria、ScanNet 等入口保持可用。

## 输入约定

支持 `OMNI_KEYFRAME_MAP` V6 完整快照，使用这些文件：

```text
manifest.txt                 # format/schema_version/map_id/snapshot_id 等
poses_base.txt               # 每行 timestamp tx ty tz qx qy qz qw
poses_image_base.txt         # 新索引必需：前八列同上，末列 lidar_keyframe_id
camera_calibration.yaml      # 原始图像尺寸、相机模型、内外参
images/index.csv             # 新七列或旧六列索引，见下文
images/<image_frame>_cam<id>.jpg # 旧包使用雷达关键帧编号
rgb/<keyframe>.pcd           # 优先使用的彩色关键帧点云
surf/<keyframe>.pcd          # RGB 点云缺失或为空时的回退几何
global_map.pcd               # 回放背景；缺失时显示一个关键帧点云
config/map_config.yaml       # V1 标定转换所需的地图自身配置
```

新索引为 `image_frame_id,keyframe_id,camera_id,image_stamp_ns,file,source,status`。
`image_frame_id` 是从 0 连续递增的同步图像组编号；索引中的 `keyframe_id` 是
雷达锚点编号，对应 `poses_base.txt` 的行号及 `rgb/`、`surf/` 点云编号。
每个图像组在 `poses_image_base.txt` 中对应一行，末列必须与索引的雷达关联一致。
一个雷达帧可以关联多个图像组；使用已导出的优化后图像位姿，不做插值或时间近邻匹配。

旧六列索引 `keyframe_id,camera_id,image_stamp_ns,file,source,status` 仍按雷达关键帧
编号、位姿及原有空观测加载，无需迁移。七列索引缺少图像位姿文件，或六列索引混入
该文件时，均明确报错。两种格式保持 V6 manifest 不变。

图像使用索引中的原始纳秒时间戳；位姿时间戳按十进制解析，避免浮点换算丢失精度。
组估计时间与各相机原始时间允许不同。处理及提示界面导航按导出编号推进，
时间重复或回退不会合并、重排图像组；按时间戳查询遇到重复值时要求改用帧索引。
`no_image`、`disabled`、`legacy` 是合法空观测；不会使用历史图像填补。
索引重复、缺少已保存图像、图像解码失败、尺寸不符及非法标定会明确报错。

地图坐标为米制、Z 轴向上。点云已在关键帧 `base_link` 坐标系中，
只应用一次雷达锚点位姿。图像和点云分别变换：

```text
T_map_camera = T_map_image_base × T_base_camera
T_map_virtual = T_map_camera × T_camera_virtual
p_map_cloud = T_map_lidar_base × p_lidar_base
```

新包的 `--start_n`、`--skip_n`、`--max_n` 及跟踪命中次数均按图像组计数，
推理入口 `--start_n` 从 1 开始，提示和回放入口从 0 开始。默认命中阈值仍为 3。
旧包继续按雷达关键帧计数。点云缓存按雷达锚点复用，完整校验只检查每个雷达点云一次。

例如验证独立采样地图包：

```bash
python run_boxer.py --input /home/glory/Code/digitaltwins-x-nav/nav_map/img4 --validate_input
```

校验分别报告 `keyframes`（雷达帧数）、`image_frames`、`frames` 与 `selected_frames`。
该地图包应为 329 个雷达帧、1776 个图像组和 3552 张图像。

V2 标定的 `Rbc/Pbc` 表示 `p_base = Rbc*p_camera + Pbc`，矩阵按行存储。
V1 的 `Rcl/Pcl` 使用地图自身保存的 `input_transform` 在内存中转换；
不会改写原地图，也不借用当前运行的 ROS 参数。`vio_scale` 不再次应用于原始内参。

支持 Pinhole、PolynomialCamera、EquidistantCamera、OmniRadtanCamera、
EucmCamera、TripleSphereCamera。公式及可逆域以 x-nav 的 vikit 实现为依据。
PCD 支持地图规定的未压缩二进制 XYZRGB、PointType 布局，不需要 Open3D。

## 相机模式与跟踪

**默认模式已由虚拟针孔改为原生相机。** 批处理、提示窗口与回放统一使用
`--xnav_camera_mode native|virtual_pinhole`；模式在启动时确定，不在窗口内切换。
旧 `--pinhole` 仍用于原有数据集，x-nav 应使用新参数。

原生模式将六种模型精确接入 `CameraTW`，保留 Polynomial 的 `k2…k7/skew`、
TripleSphere 的 `b/c` 等参数；不拟合为 Fisheye624，也不改变网络结构或权重。
每路相机只处理一次原图，等比缩放并居中填充到模型尺寸，按像素中心约定更新标定。
几何变换、投影和反投影以 float32 计算，网络可继续使用 bfloat16。

原生有效域为源图边界、模型可逆域与 **Z>0 前半球** 的交集。
Z≤0 区域置黑并禁用提示；不再额外裁至 140°，因此可保留前半球中更靠边的射线。
这不等于支持超出前半球的全部视场。深度仍为相机 Z 深度，保持预训练模型约定。
3D 框每条边取 32 个点投影，在鱼眼图像中显示为分段曲线。

虚拟针孔模式每路相机生成前、后、左、右、上、下六个候选针孔视图，每面 100°，
仅保留与原始图像有效域有交集的面。重映射、内参和有效像素掩码会缓存。
2D 框中心必须有效，且框内至少 50% 像素来自有效原图区域。

点云先转到地图系，再按当前模式相机投影、筛选掩码及最近深度，
最后确定性采样，每个观测最多 10,000 点。图像和点云使用同一份观测相机标定。

各视图分别运行检测和 3D 提升。同一处理帧内以 0.3 的 3D IoU 和
0.7 的语义相似阈值融合重复检测，然后只更新一次跟踪器。
跨处理帧匹配的 3D IoU 默认 0.25。确认需要三个不同处理帧的支持，
同一帧的双目和重叠视图不会增加确认次数。

新任务默认使用 `--xnav_dedup conservative`。低 IoU 补充关联和去重共同检查语义相似度
（至少 0.7）、中心距离（不超过 0.30 m 和较小框水平对角线的 0.35 倍）及
90° 等价朝向下的尺寸比例（各轴不超过 1.5）。满足约束的轨迹以 0.30 的
3D IoU 合并；低重叠框只有在两个不同图像组、相机位置相隔至少 0.30 m，
且投影 IoU 至少 0.70、同时匹配同一个 2D 检测时才补充关联或合并。
两个图像组中分别匹配到独立检测框的候选禁止合并，避免吞并邻近物体。
`conservative` 版本 2 保持代表框的上述严格门槛。合并簇内的历史成员若因早期
测量偏差超过普通距离限制，必须取得上述两组多视角证据，才允许在 0.60 m 和
较小框水平对角线 0.70 倍的范围内通过检查；尺寸比例仍不超过 1.5。
所有跨簇成员逐对检查，独立物体的否决证据优先于合并证据。
常规关联保留原有 0.25 的 3D IoU 和 0.7 的语义门槛，并检查上述独立检测证据；
避免对已有身份额外施加绝对距离限制而因测量抖动重复建轨。
常规关联后、分配新 ID 前，会补查未匹配检测附近已有至少两个图像组支持的旧轨迹，
包含暂时投影到视野外的轨迹。该恢复路径要求严格的几何门槛及两组多视角证据，
以一对一分配恢复原 ID，不受当前点云稀疏或缺少点云支持的影响。
已经匹配到已有轨迹的图像组，如果还有符合两组多视角证据的同物体冗余框，
会吸收为同组观测，既不重复建轨，也不重复累加支持数或几何权重。

在线逐帧检查新增和更新轨迹的近邻。新增有效 2D 观测时，当前可投影轨迹也参与
近邻复查，即使本帧没有新的 3D 检测。合并簇内成员必须两两兼容，避免链式合并。
在线合并前双方各需两个图像组的支持，让独立物体的否决证据有机会建立。
最后一个处理帧提交前对全部确认轨迹执行全局去重，重复至没有新增合并，
包含已离开视野的目标。
最后一帧、最终 JSON、轨迹 CSV 和实时回放使用同一份结果。此前帧保留当时的
轨迹状态，通过 `id_aliases` 记录后续合并关系。这是静态地图去重，不估计运动速度。
`--xnav_dedup legacy` 保留旧版关联、每五帧合并和 0.50 的合并门槛。

跟踪匹配前刷新当前多视图可见性。投影视场约束常规匹配，点云支持约束漏检计数；
离开视野或无图观测不会累计可见漏检。稀疏点云不会阻止已有 ID 再关联。
候选轨迹在至少三次有几何支持的可见漏检后移除；已确认轨迹沿用 Boxer
默认 90 次漏检的耐心参数及置信度调整。轨迹合并保留较早 ID，记录别名，
支持数按不同处理帧去重；新包中共享雷达锚点的不同图像组分别计数。

这是静态物体的跨视角关联，不估计运动速度或运动轨迹。

## 参数与缓存

| 参数 | x-nav 行为 |
| --- | --- |
| `--xnav_camera_mode native` | 默认；原始畸变图像，六种模型精确适配，仅使用前半球 |
| `--xnav_camera_mode virtual_pinhole` | 原有六方向候选针孔视图，100° 视场 |
| `--xnav_cameras all` | 全部相机；也可指定 `0` 或 `0,1` |
| `--xnav_execution auto` | CUDA 使用有界流水线；CPU/MPS 串行执行。指定 `serial` 可回退对照 |
| `--xnav_detector_batch_size 1` | OWLv2 每批 1 或 2 个有序观测；BoxerNet 始终逐相机运行 |
| `--start_n 1 --skip_n 1 --max_n 68` | 从第一个处理帧开始；新包按图像组、旧包按雷达帧计数 |
| `--labels lvisplus` | 默认开放词汇类别表；可用带引号的逗号分隔文本提示 |
| `--thresh2d 0.25 --thresh3d 0.5` | 分别筛选 2D、3D 模型置信度；保存的框分数为二者均值 |
| `--track` | 静态物体关联与全局框融合 |
| `--xnav_dedup conservative` | 默认保守去重；`legacy` 用于旧策略对照，仅影响 x-nav 多视图跟踪 |
| `--live_viz` | 推理时自动打开 3D 窗口，跟随新完成的处理帧；需要桌面显示环境 |
| `--track_min_hits 3` | 确认物体所需的不同处理帧数 |
| `--no_sdp` | 仅图像推理，并按投影视场判断可见性 |
| `--skip_viz` | 不输出逐帧预览和视频 |
| `--cache2d` | 复用保存的各相机、各视图 2D 检测，重新运行 BoxerNet |
| `--cache3d` | 复用各视图的 3D 检测，重新融合和跟踪；仍提取原图 DINOv3 特征 |
| `--no_csv` | 跳过检测与跟踪 CSV，仍保存 JSON、DINOv3 特征和词表 |
| `--profile` | 输出 `performance.json`：分阶段时间、前中后段速度、轨迹数、保存大小及内存峰值；默认关闭 |

x-nav 自动进行处理帧内融合；全局融合使用 `--track`。`--fuse` 保持用于原有数据集。

去重配置保存在 `tracking.dedup`，不参与模型缓存失效判断。缺少该字段的旧结果
按 `legacy` 读取，回放不会重新去重。改变去重模式时可以使用 `--cache3d` 重放
已有各视图检测，不加载推理模型。缓存重放会写入所选输出目录；做对照时先将
旧 `results.json` 按原目录层级复制到独立输出根目录，再指定该根目录重放。

缓存以模式和 `(frame_id, camera_id, view_id)` 关联，区分共享雷达锚点的不同图像组，
以及不同相机的相同时间戳，并保存零检测的观测。
复用前检查地图元数据、图像/点云文件状态和推理参数，
不匹配或未覆盖选定处理帧时拒绝复用。图像位姿文件内容纳入来源签名，位姿修改后
缓存失效；旧包没有该文件时原签名不变。改变 `--track_min_hits` 可从 3D 缓存重建轨迹。

结果保留原有 schema 2 文档和 schema 3 运行中日志格式。新增 `frame_unit`（`image` 或
`lidar`）；帧和观测包含 `frame_id`、`image_frame_id`、`lidar_keyframe_id`。
结果中的 `keyframe_id` 是处理编号的兼容别名，与原始索引中表示雷达关联的同名字段
不同；`sources` 保留索引字段含义。旧结果缺少新增字段时仍可读取。
最终统计中的 `processed_frames` 是处理次数，`keyframes` 是涉及的不同雷达帧数，
`image_frames` 是处理的图像组数。

### 性能观测

CUDA 默认命令使用图像/点云预取、单模型推理线程和独立 CPU 输出进程。
主线程按处理帧顺序融合与跟踪；模型和跟踪使用独立 CUDA stream。
从开始推理到可靠提交确认占用两个全局额度，最多两帧在途；不会跳帧。
输出进程逐帧生成 CSV/预览，执行原有 fsync 和原子发布后才确认提交。
类别、分辨率、相机、深度输入、检测阈值、投影采样及关联顺序保持原设置。

OWLv2 可对同一处理帧的相邻两个观测批处理，沿用原 CPU resize/归一化和逐图 NMS。
默认 batch=1；Orin NX 的完整地图验证中，batch=2 的置信度误差超过 0.002，
因此没有进入默认路径。BoxerNet 固定 batch=1。批量推理遇到 CUDA OOM 时，仅当前组按单图重试，
本次运行后续使用单图；`execution` 元数据记录实际批量及回退次数。
CPU/MPS 和 `--xnav_execution serial` 使用原串行模型调用。
`--cache2d` 跳过 OWLv2，`--cache3d` 跳过 OWLv2 和 3D 预测；两者均提取原图 DINOv3 特征。
执行模式不参与缓存参数匹配。

```bash
bash scripts/run_local_cuda.sh --input /path/to/map --track --profile
```

结果目录的 `performance.json` 包含每个处理帧的阶段耗时、处理帧 ID、轨迹数和
累计持久化大小，另有 CPU RSS 与本进程 CUDA 分配/保留内存峰值。
`steady_stages` 和 `steady_frames` 排除开头两个预热帧；`segments` 将预热后的帧
分为前、中、后三段，分别报告 p50/p95。初始化、最终完整 JSON 导出和视频编码
放在 `other_stages`，总 `wall_seconds` 包含这些阶段。
`steady_cuda_stages` 使用 CUDA events 记录各 stream 的阶段区间，在执行线程停止后
统一解析；阶段边界不再全局同步。区间可能包含主机发起算子之间的空闲，不能视为
纯 kernel 活跃时间。关闭 profiling 时不创建计时事件或增加同步。
stream 与事件语义参见 [PyTorch CUDA 说明](https://docs.pytorch.org/docs/2.10/notes/cuda.html)。
流水线各阶段可以重叠，不能把阶段耗时相加当作整段耗时；`wall_ms` 是单帧延迟，
`steady_throughput_fps` 按可靠提交间隔计算吞吐。`queue.*` 分别记录额度、
数据、推理结果与输出队列等待；`commit_lag` 记录确认时的在途帧数。
`commit_latency_ms` 和 `steady_commit_latency` 单独报告计算完成到可靠提交的延迟。
`execution` 独立保存执行模式、批量和最大在途量，不改变 schema 或缓存约定。

每帧仍保存完整轨迹快照，历史文件大小会随帧数与轨迹数增长；每帧写入只处理本帧
内容，结束时再流式汇总一次。全局合并沿用原有间隔和顺序，语义判断使用每次合并后
更新的标签。可见性沿用 native 的每边 32 点曲边投影、有效域与严格框内判断，
未增加距离截断。少于 256 条轨迹时使用 CPU 几何，避免小规模 CUDA 调度开销；
更大规模使用 float32 GPU 投影/掩码和每批最多 256 框的点支持，缓存静态相机
模型与掩码积分图。框面附近可能改变支持阈值的浮点判断由原 CPU 路径复核。
全局合并只计算至少一条轨迹可见的候选对，保留较早 ID 优先及合并后重查标签的顺序。

2026-10-30 在本机 RTX 5060 Laptop GPU、PyTorch 2.10.0+cu128、4 个 CPU 运算线程下，
对 `img2` 的 469 个关键帧、938 张图像进行顺序对照；保持 native、1220 类、960 输入、
bfloat16、跟踪、CSV、预览和视频设置：

| 完整流程 | 耗时 |
| --- | ---: |
| 修改前冻结源码 | 2804.65 秒 |
| 优化后，开启 `--profile` | 426.31 秒 |

总时间包含初始化、最终完整 JSON 导出及视频编码，减少约 84.8%。
469 帧记录及非耗时元数据严格一致，4 个 CSV 和 469 张预览图逐字节一致；
视频均为 1920×960、5 FPS、469 帧。逐帧持久化 p95 为 34.6 ms。
本机存在其他任务，测试期间还执行了回归及中间结果读取，绝对时间会随负载变化。

2026-11-02 在 Jetson Orin NX 上按同机顺序各运行两次完整 `img2`，包含 CSV、
预览和视频且关闭 profiling：串行中位数 1272.55 秒，默认流水线中位数 906.05 秒，
减少 28.80%；相同关键帧计算窗口内 GPU <10% 采样比例从 62.25% 降至 33.82%。
两次优化结果分别通过全部 469 帧差分，离散字段、最终别名及置信度完全一致。
OWLv2 和 BoxerNet 默认均为 batch=1；batch=2 未通过置信度误差门槛。
详细采样、内存与验收证据见 [Jetson 部署文档](JETSON_ORIN_NX.md)。

### 数据准备与 GPU 输入复用（2026-11-06）

CLI、HTTP 和 Web 的 `xnav_prepare_workers` 默认为 `2`。反投影按 65,536 点分块，
由同一任务的两个 CPU 线程执行原有相机公式，按原顺序拼接。相机图像和点云准备也
复用这个线程池并行执行，按原相机顺序收集；OWL 图像预处理也复用该线程池。
相机任务内部直接执行反投影，避免嵌套
等待及额外线程。共享的 LiDAR 锚点在调度相机前读取；任务退出时关闭线程池。
相机坐标矩阵乘法使用任务内互斥锁，避免部分 OpenBLAS 版本在并发窄矩阵乘法中
出现数据损坏；耗时较多的相机反投影及图像操作仍并行执行。
设为 `1` 可使用原有单线程准备路径。模型、输入分辨率、词表、阈值、采样及精度沿用
原设置，OWLv2/BoxerNet 默认 batch 仍为 1。

CUDA 自动流水线在预取时填充锁页内存，通过独立 copy stream 和 CUDA events
上传输入。native 原图只上传一次，供 DINO 特征导出和 BoxerNet 共用；CPU 图像、
相机和点云继续供跟踪及输出使用。固定相机的 patch 射线使用最多 16 项缓存，
逐帧重新应用位姿；相机参数、裁剪、尺寸、设备或 dtype 改变时使用不同缓存项。
推理只返回所需的模型结果；OWL 检测与 DINO 描述符集中回传，并在交给输出进程前
等待对应事件完成。原有最多两帧未提交额度及逐帧可靠保存顺序保持不变。

```bash
# 新默认：双线程准备与 CUDA 自动流水线
python run_boxer.py --input /path/to/map --track --xnav_prepare_workers 2

# 对照：单线程准备与串行推理
python run_boxer.py --input /path/to/map --track \
  --xnav_prepare_workers 1 --xnav_execution serial --output_dir output/comparison
```

`--profile` 的 `performance.json` 增加 `data.decode`、`data.cloud`、
`data.unproject`、`data.project`、`data.image`、`data.owl_preprocess`、
`transfer.pin`、`transfer.h2d`、`transfer.descriptors` 和 Boxer 编码/深度/射线/
Transformer/查询阶段，以及 `cold_start` 的初始化和首帧可靠提交时间。
阶段可以嵌套，不应将所有阶段时间相加作为总耗时。CUDA events 在任务收尾解析，
阶段边界不做全局同步；正式吞吐验收应关闭 profiling，分别比较冷启动与常驻模型任务。

RTX 5060 Laptop 的三组完整地图对照：`img2` 中位数减少 10.27%，`img7` 减少
6.34%；六组结果逐值及 CSV 字节比较均通过。`img7` 尚未达到 10% 整图目标。
详见 [性能验收报告](XNAV_PERFORMANCE_20261106.zh-CN.md)。

## 输出与回放

结果目录为 `output/<map_id>/<snapshot_id>/<camera_mode>/`，两种模式互不覆盖：

| 文件 | 内容 |
| --- | --- |
| `results.json` | 完整帧索引、相机/视图、各视图 2D/3D 观测、逐帧跟踪状态、参数、别名及最终物体 |
| `owl_2dbbs.csv` | 带 `frame_id`、原图时间戳及 `cam<id>/<view>` 标识的 2D 框 |
| `boxer_3dbbs.csv` | 每个处理帧融合后的 3D 检测，时间戳来自最早来源原图 |
| `boxer_3dbbs_tracks.csv` | 各处理帧的已确认跟踪框，`instance` 为持久 ID |
| `boxer_3dbbs_tracked.csv` | 每个最终物体一行，包含全部关联 2D 检测和原图路径 |
| `boxer_3dbbs_observations.csv` | 每条 2D→融合 3D→轨迹关联一行，未确认物体的 `object_id` 为空 |
| `preview_frames/`、`boxer_viz_final.mp4` | 检测/跟踪预览；视频按每秒 5 个处理帧编码 |

`--write_name` 会替换 3D CSV 和视频的 `boxer` 前缀。

### 物体、2D 检测和原图关联

新结果的 `provenance_version` 为 `1`。原有 CSV 列顺序保持不变，新字段追加在末尾。
`owl_2dbbs.csv` 的 `detection_id` 唯一标识本次结果内的检测，格式为
`f<frame_id>:c<camera_id>:v<view_id>:d<index>`；融合框以 `f<frame_id>:b<index>`
形式的 `bbox3d_id` 标识。所有 CSV 通过 `object_id` 对应最终物体的 `instance`。
历史逐帧 `instance` 保留当时的轨迹 ID，完成时新增的 `object_id` 统一解析合并别名。
未归属最终确认物体的检测，其 `object_id` 为空。

最终物体 CSV 每行的 `observations_2d` 是 JSON 数组，包含该物体的全部可靠关联检测，
包括确认前观测和被合并轨迹的观测。每个元素包含对应 OWL CSV 行的完整字段，另附
`bbox3d_id`；其中 `image_file` 是相对地图根目录的原图路径，例如
`images/12_cam0.jpg`。按原图时间、帧、相机、视图和检测序号排序，按 `detection_id`
去重；同一原图的不同检测和视图分别保留。坐标、概率分别与 OWL CSV 的两位、六位
小数精度一致。`boxer_3dbbs_observations.csv` 提供展开明细，另保留实际匹配的 `track_id`。

| CSV | `time_ns` |
| --- | --- |
| OWL 2D | 当前原图时间 |
| 融合 3D | 该框全部来源中最早的原图时间 |
| 逐帧轨迹 | 本次匹配观测组中最早的原图时间；未匹配时保留上次实际观测时间 |
| 最终物体 | 全部关联检测中最早的原图时间 |

三个 3D CSV 的 `source_detection_id`、`image_file` 指向该行时间对应的检测与原图。
新增 `pose_time_ns` 保留位姿时间；逐帧 `frame_id` 表示当前处理帧/轨迹快照，最终物体的
`frame_id` 和 `pose_time_ns` 来自最早观测。JSON 帧级 `time_ns` 的位姿语义不变。
纳秒字段在关联 JSON 数组和 API 中使用十进制字符串。

读取完整物体历史示例（图片仍使用原地图包；ZIP 不复制 `images/`）：

```python
import csv
import json
import sys
from pathlib import Path

map_root = Path("/path/to/map")
csv.field_size_limit(sys.maxsize)  # 单个物体可能有大量观测
with (map_root / "3dbbox/boxer_3dbbs_tracked.csv").open(newline="") as stream:
    for obj in csv.DictReader(stream):
        observations = json.loads(obj["observations_2d"])
        assert int(obj["time_ns"]) == min(int(o["time_ns"]) for o in observations)
        for detection in observations:
            original_image = map_root / detection["image_file"]
            print(obj["instance"], detection["detection_id"], original_image,
                  detection["x1"], detection["y1"], detection["x2"], detection["y2"])
```

2D 坐标仍对应预处理视图，需结合 `camera_id`、`view_id` 和结果中的相机变换解释，
不能直接作为原图像素坐标。旧结果继续按旧语义读取，没有关联版本时不补造来源；
旧 2D 缓存可通过 `--cache2d` 重建关联，缺少来源索引的旧 `--cache3d` 会明确报错。

### 词表与图像特征

成功推理会将实际使用的完整词汇表保存到地图包的 `3dbbox/labels.csv`，
UTF-8 编码，表头为 `sem_id,label`。语义 ID 从 0 开始，对应解析后的 `parameters.labels`
顺序，保留未检测出的类别、原始大小写和文本提示；内置词表与自定义标签均适用。
命令行、Web、缓存推理以及 `--no_csv` 模式都会保存该表。
收尾时原子替换词表，写入失败使任务失败；重跑覆盖为本次成功推理的完整类别集合。
`results.json` 的 `vocabulary_file` 给出相对地图包根目录的路径 `3dbbox/labels.csv`。

DINOv3 特征默认汇总到地图包的 `3dbbox/dino_features.csv`，每张原图一行。
命令行与 Web 使用相同位置，两种相机模式共享该文件。重复推理按帧号和相机号更新，
保留其他帧和相机的历史行，按帧号、相机号升序保存。

文件使用 UTF-8，共十一列。原有四列保持不变，末尾追加七个位姿数值列，完整表头为：

```text
相机帧序号,相机序号,cls_token,patch_mean_pooling,base_link_x,base_link_y,base_link_z,base_link_qx,base_link_qy,base_link_qz,base_link_qw
```

相机帧序号使用 `image_frame_id`，旧地图使用处理帧 `frame_id`，与原图文件名编号一致；
相机序号使用原始 `camera_id`，均不重新编号。
两个向量各占一个单元格，编码为含 384 个数值的一维 JSON 数组，由标准 CSV 引号规则处理逗号。
计算仍使用 float32，写入数值可往返还原为原来的 `[1, 384]` 张量。

七个位姿列使用 float64：`base_link_x/y/z` 是该图像对应的 `base_link` 在全局 `map`
坐标系中的位置，单位为米；`base_link_qx/qy/qz/qw` 是从 base 到 map 的旋转四元数，
按 `qx,qy,qz,qw` 顺序保存并归一化。位姿直接来自处理帧的 `T_map_base`：新地图使用
`image_frame_id` 对应的 `poses_image_base.txt` 图像组位姿，旧地图使用 `poses_base.txt`
对应的关键帧位姿。同一图像组各相机的 base 位姿相同，不叠加相机或虚拟视图外参，
也不按各相机原始图像时间戳插值。

下次推理初始化特征存储时，自动识别旧四列 CSV，读取地图完整索引和位姿，为全部历史行
补齐七列后原子替换；不受本次选帧、选相机范围限制，保留已有特征值，无需重新提取。
空文件或仅有旧表头的文件直接升级表头。旧行无法匹配已保存图像、位姿缺失、数据非法或
写入失败时任务报错，保留原 CSV；升级成功后不重复迁移。

特征覆盖所选相机和帧范围内所有 `status=saved` 的原图，
包括没有检测框的图像。`--no_csv`、关闭预览、`--cache2d` 和 `--cache3d` 均不关闭特征保存；
`--cache3d` 仍加载/复用 BoxerNet 的 DINOv3 骨干，只跳过检测与 3D 预测。
两种相机模式的特征都使用 native 预处理（等比缩放、补黑边、相机有效域遮罩），
virtual_pinhole 下每张原图只生成一份特征，不为虚拟视图重复保存。
native 下复用相同 patch 特征进行 3D 预测，virtual_pinhole 下额外提取一次原图特征。

CLS 和 patch 均取模型最后一层经过 LayerNorm 的结果；均值使用 float32，排除 CLS 和
storage/register tokens，覆盖 native 输入全部 patch（含补边和遮罩区域），不额外做 L2 归一化。

```python
import csv
import json
from pathlib import Path
import numpy as np

map_root = Path("/path/to/xnav-map")
with (map_root / "3dbbox/dino_features.csv").open(encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream):
        frame_id, camera_id = int(row["相机帧序号"]), int(row["相机序号"])
        cls = np.asarray(json.loads(row["cls_token"]), dtype=np.float32).reshape(1, 384)
        mean = np.asarray(json.loads(row["patch_mean_pooling"]), dtype=np.float32).reshape(1, 384)
        position = np.asarray(
            [row[f"base_link_{axis}"] for axis in ("x", "y", "z")], dtype=np.float64
        )
        quaternion_xyzw = np.asarray(
            [row[f"base_link_q{axis}"] for axis in ("x", "y", "z", "w")],
            dtype=np.float64,
        )
        print(frame_id, camera_id, cls.shape, mean.shape, position, quaternion_xyzw)
```

每个处理帧批量合并一次，通过地图级文件锁保护读取、合并和发布；临时文件写入、同步并
原子替换 CSV 后才提交对应 JSON 帧记录。写入失败保留原 CSV 和已提交的回放前缀。
`results.json` 源图的 `dino_feature_file` 为地图内相对路径 `3dbbox/dino_features.csv`，
按原图帧号与相机号定位行。统计字段 `dino_feature_rows` 表示本次处理的特征行数，
CSV 还可能包含先前其他处理范围的行。

新流程不再写 PT，初始化保存时清理地图包 `3dbbox/dino_features/` 下的旧 PT，
不将旧 PT 转换到 CSV；目录清空后移除目录。历史 JSON 不改写，检测与跟踪回放继续可用。

通过 Web/API server 推理时，任务输出仍按任务隔离保存；成功完成后，另外将上述五个
检测/跟踪 CSV 覆盖保存到源地图包的 `3dbbox/`。勾选“不写 CSV”、失败、中断或导入
历史结果时不执行此导出。命令行入口的输出路径保持不变，详见 [Web 服务说明](WEB_SERVER.zh-CN.md)。
新版 JSON 使用 schema_version 2，包含模式、原始相机参数、图像变换、有效域策略和预处理版本。
运行中采用 schema_version 3 的小型 `results.json` 描述文件，静态信息只保存一次，
完整逐帧快照追加到本次运行独立的 JSONL。先同步帧记录，再原子发布已提交的帧数和
字节边界；回放与缓存只读取已提交范围，忽略未提交尾部。
完成后流式导出可独立读取的紧凑 schema_version 2 `results.json`，并清理本次侧文件。
流水线在所有请求的 CSV、预览与视频输出成功后才发布完成状态；收尾失败时保留
可读取的已提交前缀，并把异常传回主进程。
失败或 Ctrl+C 时保留侧文件与 `failed` / `interrupted` 状态，已提交帧仍可回放和用于
覆盖范围内的缓存复用；这不恢复跟踪器内部状态，不属于断点续推。
项目读取器兼容 schema 1、2、3。外部程序在推理完成后仍可直接读取完整 JSON；
运行中需要使用 `utils.xnav_pipeline.read_results()` 获取同样结构的已提交快照。
CSV 无法表示的空帧在 JSON 中保留。JSON 中的旋转四元数顺序为 `wxyz`，
包围框包含 `center`、`size`、`label`、`prob` 和 `track_id`。

查看器读取保存的快照，支持拖动帧滑块、自动播放、相机/虚拟视图切换、
地图点云、当前彩色点云，以及原始检测/跟踪/候选框切换。
普通回放固定打开时已提交的帧集合。使用 `--live_viz` 推理时自动打开实时窗口，
或在已有推理运行时用下面的命令连接同一结果目录：

```bash
bash scripts/run_local_cuda.sh view_tracker.py \
  --input /home/glory/Code/digitaltwins-x-nav/nav_map/img2 --live
```

实时窗口每完成一个处理帧就更新图像、3D 框和物体 ID；第一帧完成前显示等待状态。
`Follow latest result` 默认开启；拖动帧滑块或播放历史时停止跟随，重新勾选即可回到
最新结果。暂停查看、旋转和缩放不会暂停推理。关闭窗口也不会停止推理，推理完成后
窗口保留并显示 `complete`，可继续查看；中断和失败会显示对应状态。
窗口使用独立进程，增量读取已提交的帧；日志保存在结果目录的 `live_viewer.log`。
`--skip_viz` 只关闭图片和视频导出，可与 `--live_viz` 同时使用。

2026-10-30 在本机 CUDA 上实际运行 `img2` 前 90 个关键帧、180 张图像，验证了
推理期间的自动更新和完成后保留窗口。另通过真实 OpenGL 窗口检查首帧前等待、
暂停后保留历史帧及 ID、恢复跟随和相机切换。
日志与截图保存在 `output/live_viz_validation/`，回归日志为
`output/live_viz_regression.log`。

左键拖动平移，右键拖动旋转，滚轮缩放。候选框尚未满足确认条件，
默认隐藏，可用界面的 `Include tentative candidates` 或
`--xnav_show_tentative` 显示。没有确认物体时，最终跟踪 CSV 只有表头是合法结果。

旧目录下的 schema_version 1 结果按虚拟针孔解释。只有显式指定
`--xnav_camera_mode virtual_pinhole` 且新模式目录不存在时，才回退读取旧目录；
原生模式不会加载旧针孔结果。旧缓存可在虚拟针孔模式复用，重建结果写入新目录。
新模式目录已存在但缺少结果时直接报错，避免读取不对应的历史文件。

## 验证与已知限制

六种相机的固定参考向量由 `tests/reference_xnav_cameras.cpp` 调用实际 x-nav
C++ 实现生成，保存在 `tests/fixtures/`，同时记录源文件 SHA256。
普通测试无需安装 x-nav、ROS 或 C++ 编译器；先按 `tests/README.md` 安装 pytest：

```bash
.venv/bin/python -m pytest tests/test_xnav.py tests/test_xnav_native.py tests/test_xnav_prompt.py tests/test_track_3d_boxes.py \
  tests/test_fusion.py tests/test_file_io.py tests/test_imgui_compat.py -q
```

验证覆盖投影、有效域、视场覆盖、坐标变换、时间精度、PCD 读取、V1 转换、
空观测、去重计数、再关联、较早 ID 保留、缓存隔离及跳帧回放。

2026-10-29 的参考地图实际处理了 68 个关键帧、136 张源图像和 680 个虚拟视图。
CUDA/bfloat16 推理得到 264 个 2D 检测及 24 个融合后的 3D 检测。
最大跨关键帧支持数为 2，默认要求 3，因此最终已确认物体为 0。
可显示候选框检查结果；降低确认次数会放宽证据要求，不代表精度得到验证。
该地图为室外场景，而 Boxer 主要面向室内物体；当前结果存在错误类别预测，
没有人工标注或准确率评测。完整验收记录见 `output/xnav_validation.json`。

2026-10-30 补充验证了交互式提示入口：在没有批处理结果的独立输出目录中直接加载地图，
完成文本检测、真实鼠标拖框事件、相机/虚拟视图/关键帧切换、深度点及深度块叠加，
并检查 3D 视口实际绘制的包围框像素。样例 `flag` 预测置信度约为 0.519：
默认 0.6 阈值下拒绝，在界面调至 0.5 后接受；程序默认值保持 0.6。
原有 Aria `nym10_gen1` 的 `light` 提示也通过 CUDA/OpenGL 窗口验证。
验收记录：`output/xnav_prompt_validation.json`；截图：`output/xnav_prompt_smoke.png`。

2026-10-30 双模式验收：完整地图的原生模式为 136 个观测，虚拟针孔为 680 个观测。
原生模式使用 CUDA/bfloat16 完成全包推理，得到 209 个 2D 检测及 21 个关键帧融合
3D 检测；最大支持数仍为 2，默认三关键帧确认条件下已确认物体为 0。
虚拟针孔模式成功复用旧版 3D 缓存，分别生成独立的结果目录。
两种模式的文本提示、鼠标拖框、切换、深度叠加及回放跳帧保持 ID 均通过窗口验收；
原有 Aria `light` 提示入口也通过回归。原生 `flag` 单文本提示的 3D 置信度约 0.54，
默认 0.6 下拒绝，界面调至 0.5 后接受，默认阈值保持不变。
这些数量仅用于验收数据流，不是准确率指标。

最新记录：`output/xnav_dual_mode_acceptance.json`、`output/xnav_dual_validation.json`；
回归日志：`output/xnav_dual_regression.log`、`output/xnav_native_cuda_geometry.log`；
原生提示窗口截图：`output/xnav_native_prompt_smoke.png`。
