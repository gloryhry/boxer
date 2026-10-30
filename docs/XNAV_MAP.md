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
camera_calibration.yaml      # 原始图像尺寸、相机模型、内外参
images/index.csv             # keyframe_id,camera_id,image_stamp_ns,file,source,status
images/<keyframe>_cam<id>.jpg
rgb/<keyframe>.pcd           # 优先使用的彩色关键帧点云
surf/<keyframe>.pcd          # RGB 点云缺失或为空时的回退几何
global_map.pcd               # 回放背景；缺失时显示一个关键帧点云
config/map_config.yaml       # V1 标定转换所需的地图自身配置
```

位姿、点云编号和图像通过 `keyframe_id` 关联，不对图像时间戳做近邻匹配。
图像使用索引中的原始纳秒时间戳；位姿时间戳按十进制解析，避免浮点换算丢失精度。
`no_image`、`disabled`、`legacy` 是合法空观测；不会使用历史图像填补。
索引重复、缺少已保存图像、图像解码失败、尺寸不符及非法标定会明确报错。

地图坐标为米制、Z 轴向上。点云已在关键帧 `base_link` 坐标系中，
只应用一次关键帧位姿。相机位姿为：

```text
T_map_camera = T_map_base × T_base_camera
T_map_virtual = T_map_camera × T_camera_virtual
```

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

各视图分别运行检测和 3D 提升。同一关键帧内以 0.3 的 3D IoU 和
0.7 的语义相似阈值融合重复检测，然后只更新一次跟踪器。
跨关键帧匹配的 3D IoU 默认 0.25。确认需要三个不同关键帧的支持，
同一帧的双目和重叠视图不会增加确认次数。

跟踪匹配前刷新当前多视图可见性。投影视场约束匹配，点云支持约束漏检计数；
离开视野或无图观测不会累计可见漏检。稀疏点云不会阻止已有 ID 再关联。
候选轨迹在至少三次有几何支持的可见漏检后移除；已确认轨迹沿用 Boxer
默认 90 次漏检的耐心参数及置信度调整。轨迹合并保留较早 ID，记录别名，
支持数按不同关键帧去重。

这是静态物体的跨视角关联，不估计运动速度或运动轨迹。

## 参数与缓存

| 参数 | x-nav 行为 |
| --- | --- |
| `--xnav_camera_mode native` | 默认；原始畸变图像，六种模型精确适配，仅使用前半球 |
| `--xnav_camera_mode virtual_pinhole` | 原有六方向候选针孔视图，100° 视场 |
| `--xnav_cameras all` | 全部相机；也可指定 `0` 或 `0,1` |
| `--start_n 1 --skip_n 1 --max_n 68` | 从第一个关键帧开始，按关键帧计数，而非虚拟视图 |
| `--labels lvisplus` | 默认开放词汇类别表；可用带引号的逗号分隔文本提示 |
| `--thresh2d 0.25 --thresh3d 0.5` | 分别筛选 2D、3D 模型置信度；保存的框分数为二者均值 |
| `--track` | 静态物体关联与全局框融合 |
| `--live_viz` | 推理时自动打开 3D 窗口，跟随新完成的关键帧；需要桌面显示环境 |
| `--track_min_hits 3` | 确认物体所需的不同关键帧数 |
| `--no_sdp` | 仅图像推理，并按投影视场判断可见性 |
| `--skip_viz` | 不输出逐帧预览和视频 |
| `--cache2d` | 复用保存的各相机、各视图 2D 检测，重新运行 BoxerNet |
| `--cache3d` | 复用各视图的 3D 检测，重新融合和跟踪，不加载推理模型 |
| `--no_csv` | 保留 JSON 结果与回放，不导出 CSV |
| `--profile` | 输出 `performance.json`：分阶段时间、前中后段速度、轨迹数、保存大小及内存峰值；默认关闭 |

x-nav 自动进行关键帧内融合；全局融合使用 `--track`。`--fuse` 保持用于原有数据集。

缓存以模式和 `(keyframe_id, camera_id, view_id)` 关联，区分不同相机的相同时间戳，
并保存零检测的观测。复用前检查地图元数据、图像/点云文件状态和推理参数，
不匹配或未覆盖选定关键帧时拒绝复用。改变 `--track_min_hits` 可从 3D 缓存重建轨迹。

### 性能观测

默认命令已经使用增量保存、批量记录转换与等价的跟踪优化。类别、分辨率、相机、
深度输入、检测阈值、投影采样及关联顺序保持原设置。

```bash
bash scripts/run_local_cuda.sh --input /path/to/map --track --profile
```

结果目录的 `performance.json` 包含每个关键帧的阶段耗时、实际关键帧 ID、轨迹数和
累计持久化大小，另有 CPU RSS 与本进程 CUDA 分配/保留内存峰值。
`steady_stages` 和 `steady_frames` 排除开头两个预热帧；`segments` 将预热后的帧
分为前、中、后三段，分别报告 p50/p95。初始化、最终完整 JSON 导出和视频编码
放在 `other_stages`，总 `wall_seconds` 包含这些阶段。
只有 `--profile` 开启时才同步 GPU 进行阶段计时；它会带来测量开销。

每帧仍保存完整轨迹快照，历史文件大小会随帧数与轨迹数增长；每帧写入只处理本帧
内容，结束时再流式汇总一次。全局合并沿用原有间隔和顺序，语义判断使用每次合并后
更新的标签。可见性沿用 native 的 32 点曲边投影，点支持仅计算可见框，每批最多 256 框。

## 输出与回放

结果目录为 `output/<map_id>/<snapshot_id>/<camera_mode>/`，两种模式互不覆盖：

| 文件 | 内容 |
| --- | --- |
| `results.json` | 完整帧索引、相机/视图、各视图 2D/3D 观测、逐帧跟踪状态、参数、别名及最终物体 |
| `owl_2dbbs.csv` | 带 `frame_id`、原图时间戳及 `cam<id>/<view>` 标识的 2D 框 |
| `boxer_3dbbs.csv` | 每个关键帧融合后的 3D 检测，时间戳来自关键帧位姿 |
| `boxer_3dbbs_tracks.csv` | 各关键帧的已确认跟踪框，`instance` 为持久 ID |
| `boxer_3dbbs_tracked.csv` | 最终已确认全局物体，兼容 Boxer OBB CSV |
| `preview_frames/`、`boxer_viz_final.mp4` | 检测/跟踪预览；视频按每秒 5 个关键帧编码 |

`--write_name` 会替换 3D CSV 和视频的 `boxer` 前缀。
新版 JSON 使用 schema_version 2，包含模式、原始相机参数、图像变换、有效域策略和预处理版本。
运行中采用 schema_version 3 的小型 `results.json` 描述文件，静态信息只保存一次，
完整逐帧快照追加到本次运行独立的 JSONL。先同步帧记录，再原子发布已提交的帧数和
字节边界；回放与缓存只读取已提交范围，忽略未提交尾部。
完成后流式导出可独立读取的紧凑 schema_version 2 `results.json`，并清理本次侧文件。
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

实时窗口每完成一个关键帧就更新图像、3D 框和物体 ID；第一帧完成前显示等待状态。
`Follow latest result` 默认开启；拖动帧滑块或播放历史时停止跟随，重新勾选即可回到
最新结果。暂停查看、旋转和缩放不会暂停推理。关闭窗口也不会停止推理，推理完成后
窗口保留并显示 `complete`，可继续查看；中断和失败会显示对应状态。
窗口使用独立进程，增量读取已提交的帧；日志保存在结果目录的 `live_viewer.log`。
`--skip_viz` 只关闭图片和视频导出，可与 `--live_viz` 同时使用。

左键拖动平移，右键拖动旋转，滚轮缩放。候选框尚未满足确认条件，
默认隐藏，可用界面的 `Include tentative candidates` 或
`--xnav_show_tentative` 显示。没有确认物体时，最终跟踪 CSV 只有表头是合法结果。

旧目录下的 schema_version 1 结果按虚拟针孔解释。只有显式指定
`--xnav_camera_mode virtual_pinhole` 且新模式目录不存在时，才回退读取旧目录；
原生模式不会加载旧针孔结果。旧缓存可在虚拟针孔模式复用，重建结果写入新目录。
新模式目录已存在但缺少结果时直接报错，避免读取不对应的历史文件。
