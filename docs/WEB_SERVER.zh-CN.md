# x-nav Web 服务

在 Boxer 仓库目录运行。服务使用现有 Boxer Python 环境执行推理；浏览器只需要支持 WebGL，不需要 CUDA 或桌面窗口。

Docker 部署使用独立的 x86 NVIDIA / Jetson Orin 配置，模型和词表通过挂载提供，
地图固定映射至 `/nav_map`。准备和启动方式见 [Docker 部署](DOCKER.zh-CN.md)。

## 安装与启动

```bash
uv pip install --python .venv/bin/python -r requirements-server.txt
npm --prefix web ci
npm --prefix web run build
.venv/bin/python server.py
```

默认监听 `0.0.0.0:18765`，结果保存在仓库的 `output/server/`。本机访问 <http://127.0.0.1:18765/>；局域网设备使用服务器的 IP 和同一端口。

服务启动即在常驻推理进程中加载默认 BoxerNet、OWLv2 权重及 `indoor` 词表，并完成 OWLv2 预热。页面与回放接口可在加载期间使用，开始推理需等待“模型就绪”。每次重新启动服务都使用默认参数；历史任务记录仍然保留。

```bash
.venv/bin/python server.py --host 127.0.0.1 --port 18765 --output_dir /path/to/server-output
```

Web 界面、Swagger UI 脚本和 Three.js 均由本服务提供，不依赖浏览器访问外部 CDN。构建时需要 Node.js 20.19+ 或 22.12+；正常运行服务不需要 Node.js。开发前端可运行 `npm --prefix web run dev`，其 API 代理指向本机 `18765` 端口。

## 使用

1. 页面自动递归查找服务器 `/nav_map` 下的 x-nav 地图包，在下拉列表中选择后点击“加载地图”。选项显示相对路径（例如 `园区A/一楼/地图1`），只列出地图包，不列出普通文件夹及地图内部的图像、点云目录。可点击“刷新列表”重新查找；上次选择仍存在时会恢复选择，但不会自动加载地图。加载会完整验证地图、图像和点云；地图较大时需要等待。
   也可展开“上传 ZIP 地图包”，选择包含一张地图的 ZIP 后开始上传。默认 8 MiB 分块、最多 3 块并发，支持暂停、继续和取消。页面依次显示本地文件校验、上传及服务端解压校验进度；成功后自动注册并加载新地图。上传不会停止已经运行的推理，可在历史任务中继续查看原任务。
   浏览器保存会话标识，刷新后从上传记录恢复；继续传块需重新选择原 ZIP，整包哈希相同后仅补传缺块。处理阶段的会话无需重新选择文件即可恢复进度。默认单包 20 GiB、闲置 24 小时清理，GET 状态查询不延长期限。前端哈希 Worker 随页面打包，支持普通 HTTP 局域网地址。
2. 按需展开“推理参数”，选择词表、阈值、相机及跟踪选项，点击“应用参数”。等待“模型就绪”后点击“开始推理”。默认词表为 `indoor`，默认开启跟踪，其余参数沿用命令行默认值；参数有未应用修改时不能开始推理。
3. 页面显示已持久化的结果。关闭或刷新浏览器不影响推理；同一时间只接受一个推理任务，其他浏览器可以共同查看。
4. 点击“停止推理”后进入停止状态，保留已经提交的帧。先发送中断，等待最多 60 秒，再终止 worker 及其输出子进程；5 秒后仍未退出则强制结束。正常停止后复用模型，强制结束后重新加载本次服务中最后成功应用的参数。
5. 在“已有结果与历史任务”输入 `results.json` 路径加载只读结果，或选择历史任务。回放暂停只控制播放，不暂停推理。

`/nav_map` 不存在、不可读或未找到地图包时，页面会给出提示；准备好目录后可刷新列表。递归查找不跟随目录符号链接，不可读的子目录会被跳过。目录列表读取失败不影响历史任务恢复，当前加载地图的完整路径会显示在地图信息中；历史地图可以位于 `/nav_map` 之外。原生部署也需要准备服务器 `/nav_map` 目录，或使用下方 API 直接传入完整路径。

全局点云最多显示 50 万点，当前彩色点云最多显示 5 万点；这些限制只影响可视化。点云使用 24 字节一个顶点的 little-endian `float32 xyzrgb`，颜色范围为 0–1。图像、当前点云缓存每个已加载地图最多 32 MiB。

服务正常退出会停止所属推理任务。推理包装进程也会监视服务进程，服务意外退出时请求中断，超时则结束自己的进程组。服务重启会恢复任务记录，将之前未结束的推理标记为中断，不自动继续推理。每个任务的参数、日志和输出相互独立。

推理任务共享同一套已加载模型，跟踪器按任务重新创建。推理或模型加载期间不能应用参数。更换 checkpoint 只重新加载 BoxerNet，更换设备重新加载两种模型，更换有效精度重新准备 OWLv2；修改词表只更新文本/语义嵌入并预热。阈值、相机、采样和跟踪设置不重新读取模型权重。加载失败会显示错误，可修改参数后再次点击“应用参数”重试。

## 地图包中的检测结果

任务成功收尾时，一并将实际加载的完整词表保存到地图包的 `3dbbox/labels.csv`，
表头为 `sem_id,label`，使用 UTF-8。ID 从 0 开始，与本次实际类别顺序一致，
包含未检测出的类别，并支持自定义文本提示。勾选“不写 CSV”仍保存词表。
重跑原子覆盖该文件；`results.json` 的 `vocabulary_file` 记录地图内相对路径。

每次推理默认将原图特征合并到地图包的 `3dbbox/dino_features.csv`，每张原图一行。
四列为 `相机帧序号,相机序号,cls_token,patch_mean_pooling`，两个向量各为单元格内
含 384 个数值的一维 JSON 数组，读取后可恢复 float32 `[1, 384]`。
覆盖本次选择的所有有效原图，包括无检测框图像；勾选“不写 CSV”仍会保存特征。
两种相机模式都采用 native 预处理，每张原图只保存一对向量。
`results.json` 的 `dino_feature_file` 引用该地图内相对路径，按帧号与相机号定位行。
重复推理更新同键行、保留其他历史行。每处理帧在文件锁保护下原子合并 CSV，
在对应帧提交前保存完成；地图目录不可写时任务报错。
不再生成 PT，初始化时直接清理地图特征目录中的旧 PT，不转换其内容。
加载方式与完整字段见 [x-nav 输出说明](XNAV_MAP.md#输出与回放)。

server 推理成功且任务输出全部关闭后，将任务最终结果目录中的以下文件保存到输入地图包的 `3dbbox/`：

- `boxer_3dbbs_tracked.csv`：最终确认物体。
- `boxer_3dbbs_tracks.csv`：逐帧跟踪框。
- `boxer_3dbbs.csv`：逐帧 3D 检测。
- `owl_2dbbs.csv`：逐帧 2D 检测。

目录自动创建，四个文件完整暂存后逐个原子替换同名文件。任务目录继续保留原件；地图导出后，后台将 `3dbbox/` 保存为该任务的 ZIP 快照，ZIP 原子发布后任务才显示“完成”。缺少 CSV、地图目录不可写或 ZIP 归档失败会将任务标记为失败，任务目录中的推理结果仍可回放。地图源数据签名不包含新增的 `3dbbox/`。

新建且未勾选“不写 CSV”的成功任务可通过 `GET /api/jobs/{job_id}/3dbbox.zip` 下载，
包含完成时的检测 CSV、词表及地图级累计 DINO 特征，保留顶层 `3dbbox/` 目录。
后续重跑和服务重启不改变旧任务的 ZIP。历史和导入任务不补包，“不写 CSV”的任务
不提供 ZIP；响应状态码和下载示例见 [OpenAPI 使用指南](OPENAPI.zh-CN.md#78-get-apijobsjob_id3dbboxzip)。

推理失败、中断或加载已有结果时不执行 CSV 地图导出。勾选“不写 CSV”会跳过任务 CSV 与 CSV 地图导出，保留地图已有 CSV。已保存的 DINOv3 特征会保留。无检测或关闭跟踪时，相应空结果仍生成带表头的 CSV。CSV 地图导出仅适用于 server；命令行 CSV 仍使用原有输出目录。

## 视角操作

- Web 和桌面 x-nav：左键拖动旋转，右键拖动平移，滚轮缩放；Web 另支持单指旋转、双指平移及缩放。
- “相机中心跟随”与“跟随最新结果”独立。前者使用当前选择视图的 `T_map_view` 位置，保留观察方向、缩放距离及世界坐标下的手动平移偏移。
- 连续播放时平滑跟随，手动拖动帧滑条时直接定位。没有相机位姿时保持观察位置。
- “定位相机”清空平移偏移；“全图视角”关闭相机位置跟随并恢复全局观察。默认显示全图，不开启位置跟随。
- 桌面可使用 `python view_tracker.py --input /path/to/map --live --init_follow` 初始开启位置跟随。

## 页面布局

宽屏（至少 1024px）下，2D 检测和 3D 投影是独立浮窗：拖动标题栏可移动到工作区任意位置，拖动四边或四角可自由调整宽高。图像始终保持比例完整显示。点击窗口会将其置于其他图像浮窗上方。

标题栏按钮或双击标题可收起为仅标题栏，再次操作恢复原尺寸。拖动参数栏右侧的分隔条可调整宽度，参数区独立滚动，点云会实时适配剩余空间。分隔条方向键每次调整 10px，按住 Shift 时每次 50px，Home/End 分别调整到最小/最大宽度；浮窗标题栏和右下角手柄也支持方向键移动、缩放。

窄屏自动恢复堆叠布局，图像仍可折叠；回到宽屏会恢复本次打开页面时调整的布局。切帧、切换相机、地图和任务不会重置布局。顶栏“恢复布局”只重置窗口及参数栏尺寸，刷新页面也会恢复默认布局；布局不会写入浏览器存储。

## API

网页高级参数中的“静态地图去重”默认选择“保守去重”，对应 API 字段
`options.xnav_dedup: "conservative"`。选择 `"legacy"` 可使用旧策略对比。
保守模式结合中心距离、尺寸、3D 重叠和多视角 2D 检测证据，并在结束前清理
离开视野的重复轨迹。它面向静态物体；移动目标仍需要运动跟踪。
历史任务卡片显示保存的策略，缺少策略的旧结果显示为旧版策略；查看历史不会修改服务当前推理参数。

- 交互文档：<http://127.0.0.1:18765/docs>
- OpenAPI：<http://127.0.0.1:18765/openapi.json>

```bash
curl http://127.0.0.1:18765/api/health
curl http://127.0.0.1:18765/api/maps/directories
curl http://127.0.0.1:18765/api/inference
curl -X PUT http://127.0.0.1:18765/api/inference \
  -H 'Content-Type: application/json' \
  -d '{"labels":"indoor","track":true,"xnav_dedup":"conservative"}'
# 等待 GET /api/inference 返回 status: "ready" 后再开始任务。
curl -X POST http://127.0.0.1:18765/api/maps \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/xnav/map"}'
curl -X POST http://127.0.0.1:18765/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"map_id":"返回的地图 ID"}'
```

`GET /api/maps/directories` 递归列出 `/nav_map` 下按现有规则识别的地图包，返回 `{ "root": "/nav_map", "directories": [{ "name": "园区A/一楼/地图1", "path": "/nav_map/园区A/一楼/地图1" }] }`，按相对路径排序。查找只检查地图标识文件，不读取图像、点云或执行完整验证；目录不存在返回 `404`，根目录读取失败返回 `400`，空目录返回空列表。

`POST /api/maps` 继续接受服务器本地完整路径，包括 `/nav_map` 之外的路径，原有请求格式和地图验证行为保持不变。

`GET /api/inference` 返回 `status`（`loading` / `ready` / `failed`）、最后成功应用的 `options`、实际执行 `device` 和加载 `error`。`PUT /api/inference` 接收完整或部分 `InferenceOptions`，省略项使用默认值，返回 `202` 后查询状态；加载失败不会将失败参数保存为已应用参数。推理或加载期间再次应用返回 `409`。模型未就绪时创建任务返回 `503`。

`POST /api/jobs` 省略 `options` 时使用已应用配置。仍可提供 `options`，但解析后的完整配置必须与服务已应用配置一致，否则返回 `409`，需先调用 `PUT /api/inference`。参数是服务级共享状态，页面轮询更新；有本地未应用修改时保留草稿，并禁止使用过期配置启动任务。`GET /api/health` 同时返回 `inference_status` 和 `inference_error`。

`GET /api/jobs/{id}/events` 使用 SSE，连接时先发送当前任务摘要，之后发送变化。事件格式为 `event: job`，`data` 为 `JobInfo`；断线重连重新获取当前摘要。帧数据按需读取，不在事件中反复传输完整轨迹或点云。

帧索引从 0 开始，推理 `start_n` 从 1 开始。接口分别保留 `frame_id`、`image_frame_id` 和 `lidar_keyframe_id`；所有 `*_ns` 字段均编码为字符串。回放直接读取已保存的检测、轨迹和 ID，不重复运行跟踪器。读入旧、新结果格式以及 native/virtual_pinhole 模式时会验证来源地图签名和预处理版本。

首版面向可信本机和局域网，没有账号系统或上传地图功能。相机视角和回放位置属于各浏览器，不会互相覆盖。

## 验证

```bash
uv pip install --python .venv/bin/python pytest httpx
.venv/bin/python -m pytest -q tests/test_xnav_map_directories.py tests/test_xnav_server.py tests/test_xnav_runtime.py tests/test_xnav.py tests/test_xnav_execution.py tests/test_xnav_result_store.py tests/test_xnav_profile.py tests/test_orbit_camera.py
npm --prefix web test
npm --prefix web run build
```

服务测试使用真实地图格式、结果日志与常驻子进程协议，配合微型 CPU 模型验证预加载、参数应用、连续任务、跟踪重置、CSV 导出、并发冲突、失败、停止升级、服务清理和重启恢复。模型运行时测试单独核对权重加载次数及词表更新。真实 GPU 推理和显存释放需要在 NVIDIA 驱动可用的环境另行验收。
