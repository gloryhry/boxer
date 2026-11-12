# Boxer x-nav OpenAPI 详细使用文档

本文覆盖 `server.py` 的地图注册与上传、Boxer 物体检测与回放、房间分割和任务输出文件。

核对日期为 2026-11-11，服务 API 版本为 `2.0.0`。部署版本的字段与约束可通过 `/openapi.json` 核对。

| 用途 | 提交入口 | 结果入口 |
| --- | --- | --- |
| 注册或上传 x-nav 地图 | `POST /api/maps` 或 `POST /api/maps/uploads` | 地图 `id`、场景元数据、预览点云 |
| Boxer 与体素房间分割联合流程 | `POST /api/jobs` | 逐帧回放、检测框、房间、物体绑定、`3dbbox.zip` |

## 目录

- [1. 服务入口与部署](#1-服务入口与部署)
- [2. 通用约定](#2-通用约定)
- [3. 从加载地图到读取结果](#3-从加载地图到读取结果)
- [4. 接口总览](#4-接口总览)
- [5. 健康状态与推理配置](#5-健康状态与推理配置)
- [6. 地图接口](#6-地图接口)
- [7. 任务管理接口](#7-任务管理接口)
- [8. 帧数据、检测框、图像与点云接口](#8-帧数据检测框图像与点云接口)
- [9. 数据模型与坐标约定](#9-数据模型与坐标约定)
- [10. 客户端接入示例](#10-客户端接入示例)
- [11. 文件输出与历史恢复](#11-文件输出与历史恢复)
- [12. 错误处理与排查](#12-错误处理与排查)
- [13. 房间分割配置与运行](#13-房间分割配置与运行)
- [14. 房间几何格式](#14-房间几何格式)
- [15. 联合工作流与物体房间绑定](#15-联合工作流与物体房间绑定)
- [16. 实现位置与维护](#16-实现位置与维护)

## 1. 服务入口与部署

### 1.1 地址

默认 Base URL 为 `http://127.0.0.1:18765`。局域网客户端将 `127.0.0.1` 替换为服务器 IP；端口以实际启动配置为准。

| 地址 | 用途 |
| --- | --- |
| `/` | Web 地图推理与回放界面 |
| `/docs` | Swagger UI，可展开接口并使用 Try it out 调用 |
| `/openapi.json` | 机器可读的 OpenAPI schema，可供接口工具导入或生成客户端 |
| `/api/...` | 业务 API，完整清单见第 4 节 |

服务没有启用 ReDoc。Swagger UI 的脚本和样式由本服务的 `/docs-assets/` 提供，因此需要构建前端静态资源。

### 1.2 本地启动

在 Boxer Git 仓库根目录执行。

```bash
# 在已有 Boxer 环境中补充 HTTP 服务依赖。
uv pip install --python .venv/bin/python -r requirements-server.txt

# 构建 Web 页面和本地 Swagger UI 资源。
npm --prefix web ci
npm --prefix web run build

# 启动服务；默认监听 0.0.0.0:18765。
.venv/bin/python server.py
```

已有 Boxer 环境、模型权重和默认 `indoor` 词表是前提；`requirements-server.txt` 补充 HTTP 依赖和 `requirements-potential-field.txt` 中的 CPU 房间分割、几何计算依赖。环境配置见 [本地 CUDA 环境](LOCAL_CUDA.md) 和 [Jetson 部署](JETSON_ORIN_NX.md)。

可设置的服务启动参数如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--host` | `0.0.0.0` | HTTP 监听地址 |
| `--port` | `18765` | HTTP 端口 |
| `--output_dir` | 仓库下的 `output/server` | 地图注册记录、任务、日志和结果的存储根目录 |
| `--map_root` | `/nav_map` | 地图发现、上传暂存与发布的根目录；需提前创建且可写 |

例如只允许本机访问，并指定持久化目录：

```bash
.venv/bin/python server.py \
  --host 127.0.0.1 \
  --port 18765 \
  --output_dir /path/to/server-output
```

HTTP 启动时会异步预加载默认 Boxer 模型及 `indoor` 词表。提交物体检测任务前，应查询 `/api/inference` 并等待 `status` 为 `ready`。检测、房间分割与物品归属统一通过 `POST /api/jobs` 提交，完整流程以 `workflow.status` 为准。

### 1.3 Docker 部署与路径映射

项目提供 `docker-compose.x86.yaml` 和 `docker-compose.jetson.yaml`。挂载目录、权重准备和完整启动方式见 [Docker 部署文档](DOCKER.zh-CN.md)。

API 中的 `path`、`results_path`、`ckpt` 都是**服务进程可见的文件系统路径**。例如宿主机地图挂载到容器 `/nav_map` 后，请求应使用 `/nav_map/园区A/地图1`。

| 容器路径 | 用途 |
| --- | --- |
| `/nav_map` | 地图发现根目录及地图读写挂载点 |
| `/app/ckpts` | 模型权重及文本嵌入缓存 |
| `/vocab` | 自定义词表挂载点，容器设置 `BOXER_LABELS_DIR=/vocab` |
| `/app/output/server` | 默认服务输出目录 |

`GET /api/maps/directories` 默认查找 `/nav_map`。原生部署可用 `server.py --map_root /path/to/maps` 指定发现根目录，嵌入 Python 应用时可用 `create_app(map_root=...)`。`POST /api/maps` 也接受其他服务可读的完整路径。

## 2. 通用约定

### 2.1 请求与响应

- 普通请求体使用 JSON，并设置 `Content-Type: application/json`；分块上传使用 `application/octet-stream`。
- JSON 接口成功返回 `application/json`；点云为 `application/octet-stream`，图像为 `image/jpeg`，事件流为 `text/event-stream`。
- 请求体模型禁止额外字段；字段拼写错误或传入不支持的参数通常返回 `422`。
- 地图 ID 和任务 ID 是服务生成的不透明字符串。`map_id` 应使用 `POST /api/maps` 返回的 `id`，与地图清单中的业务 `map_id` 区分。
- 当前服务面向可信本机或局域网，没有账号、API Key 或 Token 认证，也没有配置跨域中间件。浏览器客户端可通过同源路径或开发代理访问；项目 Vite 开发代理将 `/api` 转发到本机 `18765`。
- 地图可通过服务器本地路径注册或上传 ZIP；已有推理结果通过服务器本地路径导入。

### 2.2 时间、编号与坐标

| 数据 | 约定 |
| --- | --- |
| `time_ns`、`image_stamp_ns` 等 `*_ns` 字段 | JSON 字符串，保存整数纳秒，避免 JavaScript 数值精度损失 |
| `created_at` | UTC ISO 8601 字符串，通常带 `+00:00` |
| 路径参数 `index` | 任务结果中从 `0` 开始的连续回放索引 |
| 推理参数 `start_n` | 从 `1` 开始的源处理帧选择位置 |
| `frame_id` | 源处理帧编号；采样后可能不连续 |
| `image_frame_id` | 新格式地图的同步图像组编号；旧地图为 `null` |
| `lidar_keyframe_id` | 与处理帧关联的雷达关键帧编号 |
| 3D 坐标 | 地图坐标系 `map`，长度单位米，`+Z` 向上 |
| `quaternion_wxyz` | 四元数顺序为 `[w, x, y, z]` |

`time_ns` 是处理帧位姿时间，`image_stamp_ns` 是各相机原始图像时间，两者允许不同。同一雷达帧可以对应多个图像组，因此不能用 `lidar_keyframe_id` 替代回放索引或处理帧 ID。

例如设置 `start_n=3, skip_n=2, max_n=2` 后，结果 `index=0,1` 通常对应源 `frame_id=2,4`。

### 2.3 并发和异步行为

服务同时只受理一个联合计算任务。物体检测的模型与已应用参数由所有客户端共享，历史回放可以同时读取。

`POST /api/jobs` 同时提交 Boxer 与体素连通复核，成功后自动绑定最终物体。联合流程同时只受理一个，已有活动流程或内部 CPU 分割占槽时返回 `409`，不排队。导入目录和 ZIP 上传完成仍只注册地图，不启动任务。完整流程以 `workflow.status` 为准；顶层 `status` 仍仅表示 Boxer 阶段。

`PUT /api/inference`、`POST /api/jobs`、停止任务接口的正常响应码为 `202`，表示请求已受理；最终状态需要继续查询。地图注册和结果导入正常返回 `200`。

关闭浏览器、断开 HTTP 请求或关闭 SSE 连接不会停止推理。需要停止时显式调用任务的 `/stop` 接口。

## 3. 从加载地图到读取结果

推荐顺序：检查服务和模型 → 注册地图 → 应用参数并等待就绪 → 创建任务 → 查询进度或订阅 SSE → 分页读取已提交帧 → 获取图像和点云。

下例使用 Bash、`curl` 和 `jq`。先启动服务，并将 `MAP_PATH` 修改为服务端实际可读的地图目录。

```bash
API_BASE='http://127.0.0.1:18765'
MAP_PATH='/nav_map/园区A/地图1'

# 1. HTTP 状态与默认地图发现列表。
curl --fail-with-body -sS "$API_BASE/api/health" | jq
curl --fail-with-body -sS "$API_BASE/api/maps/directories" | jq

# 后续配置和任务都要求模型就绪。
wait_ready() {
  local response state
  while true; do
    response=$(curl --fail-with-body -sS "$API_BASE/api/inference") || return 1
    state=$(jq -r '.status' <<<"$response")
    case "$state" in
      ready) return 0 ;;
      failed) jq . <<<"$response" >&2; return 1 ;;
      loading) sleep 1 ;;
      *) printf 'Unexpected inference status: %s\n' "$state" >&2; return 1 ;;
    esac
  done
}

wait_ready || exit 1

# 2. 注册并完整校验地图；目录发现接口不执行这一步。
MAP_RESPONSE=$(jq -n --arg path "$MAP_PATH" '{path:$path}' |
  curl --fail-with-body -sS -X POST "$API_BASE/api/maps" \
    -H 'Content-Type: application/json' --data-binary @-) || exit 1
MAP_ID=$(jq -er '.id' <<<"$MAP_RESPONSE") || exit 1

# 3. 应用服务级配置。省略的字段采用默认值。
PUT_RESPONSE=$(curl --fail-with-body -sS -X PUT "$API_BASE/api/inference" \
  -H 'Content-Type: application/json' \
  -d '{"labels":"chair,table,door","track":true,"max_n":100}') || exit 1
jq . <<<"$PUT_RESPONSE"
wait_ready || exit 1

# 4. 创建任务；省略 options，使用已应用配置。
JOB_RESPONSE=$(jq -n --arg map_id "$MAP_ID" '{map_id:$map_id}' |
  curl --fail-with-body -sS -X POST "$API_BASE/api/jobs" \
    -H 'Content-Type: application/json' --data-binary @-) || exit 1
JOB_ID=$(jq -er '.id' <<<"$JOB_RESPONSE") || exit 1
printf 'MAP_ID=%s\nJOB_ID=%s\n' "$MAP_ID" "$JOB_ID"

# 5. 查看状态；处理完成前也可反复查询。
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID" | jq

# 6. 至少提交一帧后，读取分页摘要和首帧详情。
curl --fail-with-body -sS \
  "$API_BASE/api/jobs/$JOB_ID/frames?offset=0&limit=200" | jq
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/frames/0" | jq

# 7. 下载首帧第一个可用观测的图像。
curl --fail-with-body -sS \
  "$API_BASE/api/jobs/$JOB_ID/frames/0/images/2d" -o frame-0-2d.jpg
curl --fail-with-body -sS \
  "$API_BASE/api/jobs/$JOB_ID/frames/0/images/3d?tracked=true" -o frame-0-3d.jpg
```

如果本地没有 `/nav_map`，目录发现会返回 `404`；仍可将 `MAP_PATH` 设置为其他完整地图路径直接注册。首帧结果尚未发布时，帧接口也可能返回 `404`，应等待任务进度再读取。

## 4. 接口总览

下表列出业务 API 操作。成功状态码指正常受理或读取时的状态。

| 方法 | 路径 | 成功码 | 用途 |
| --- | --- | --- | --- |
| `GET` | `/api/health` | `200` | HTTP、硬件、活动任务与模型状态 |
| `GET` | `/api/inference` | `200` | 查询已应用推理参数及模型状态 |
| `PUT` | `/api/inference` | `202` | 应用推理参数并准备模型 |
| `GET` | `/api/maps/directories` | `200` | 递归发现地图包目录 |
| `POST` | `/api/maps` | `200` | 注册并完整校验地图 |
| `POST` | `/api/maps/uploads` | `201` | 创建 ZIP 上传会话 |
| `GET` | `/api/maps/uploads` | `200` | 查询上传限制和会话列表 |
| `GET` | `/api/maps/uploads/{upload_id}` | `200` | 查询分块、处理进度和结果 |
| `PUT` | `/api/maps/uploads/{upload_id}/chunks/{index}` | `200` | 上传一个二进制分块 |
| `POST` | `/api/maps/uploads/{upload_id}/complete` | `202` | 提交合并、解压、校验和注册 |
| `DELETE` | `/api/maps/uploads/{upload_id}` | `200` | 取消未完成上传 |
| `GET` | `/api/maps/{map_id}` | `200` | 查询已注册地图 |
| `GET` | `/api/maps/{map_id}/scene` | `200` | 查询全局场景元数据 |
| `GET` | `/api/maps/{map_id}/cloud` | `200` / `304` | 获取全局可视化点云 |
| `GET` | `/api/jobs` | `200` | 获取任务列表 |
| `POST` | `/api/jobs` | `202` | 同时创建 Boxer 与体素分割任务 |
| `POST` | `/api/jobs/import` | `200` | 导入已有结果用于只读回放 |
| `GET` | `/api/jobs/{job_id}` | `200` | 获取任务摘要与进度 |
| `POST` | `/api/jobs/{job_id}/stop` | `202` | 请求停止推理任务 |
| `GET` | `/api/jobs/{job_id}/log` | `200` | 获取任务日志尾部 |
| `GET` | `/api/jobs/{job_id}/events` | `200` | 订阅任务 SSE 事件 |
| `GET` | `/api/jobs/{job_id}/3dbbox.zip` | `200` | 下载成功任务的 `3dbbox/` ZIP 快照 |
| `GET` | `/api/jobs/{job_id}/rooms` | `200` | 查询关联房间列表、统计及原始产物地址 |
| `GET` | `/api/jobs/{job_id}/room-polygons.json` | `200` | 下载房间轮廓与高度范围 |
| `GET` | `/api/jobs/{job_id}/room-bindings` | `200` | 分页查询最终物体的房间归属 |
| `GET` | `/api/jobs/{job_id}/room-bindings.json` | `200` | 下载完整绑定结果 |
| `GET` | `/api/jobs/{job_id}/frames` | `200` | 分页查询已提交帧摘要 |
| `GET` | `/api/jobs/{job_id}/frames/{index}` | `200` | 获取单帧完整结果 |
| `GET` | `/api/jobs/{job_id}/frames/{index}/images/{kind}` | `200` / `304` | 获取 2D 或 3D 叠加 JPEG |
| `GET` | `/api/jobs/{job_id}/frames/{index}/cloud` | `200` / `304` | 获取当前帧可视化点云 |
| `GET` | `/api/jobs/{job_id}/3dbbox_tracked` | `200` | 分页获取当前确认物体或最终物体 |
| `GET` | `/api/jobs/{job_id}/3dbbox_tracks` | `200` | 分页获取各已提交帧的确认跟踪框 |
| `GET` | `/api/jobs/{job_id}/3dbbox` | `200` | 分页获取各已提交帧融合后的 3D 检测 |
| `GET` | `/api/jobs/{job_id}/owl_2dbbox` | `200` | 分页获取各已提交帧、相机和视图的 OWL 2D 检测 |

`/docs`、`/openapi.json`、`/` 和静态资源地址不计入业务操作数。

## 5. 健康状态与推理配置

### 5.1 `GET /api/health`

无请求体和查询参数。响应模型为 `HealthInfo`：

```json
{
  "status": "ok",
  "cuda_available": true,
  "device": "NVIDIA GPU 名称",
  "active_job_id": null,
  "active_workflow_id": null,
  "inference_status": "ready",
  "inference_error": null
}
```

`status: "ok"` 表示 HTTP 服务可响应，不能单独用来判断可否开始推理。即使模型加载失败，该接口仍可正常返回；应检查 `inference_status` 和 `inference_error`。

这里的 `device` 是硬件名称，CUDA 不可用时为 `cpu`；实际模型执行设备以 `/api/inference` 的 `device` 为准。例如设置 `force_cpu=true` 后，健康信息仍可能显示存在 NVIDIA GPU。

`active_job_id` 表示正在运行的 Boxer 任务；`active_workflow_id` 表示尚未结束的联合流程，两者使用同一个 Boxer 任务 ID。Boxer 完成但分割或绑定仍运行时，前者可为 `null`，后者仍有值。

### 5.2 `GET /api/inference`

响应模型 `InferenceInfo` 包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | `loading` / `ready` / `failed` | 模型正在准备、已就绪或准备失败 |
| `options` | `InferenceOptions` | 本次服务中最后成功应用的完整配置；初始化时为默认配置 |
| `device` | 字符串 | 执行设备，如 `cuda`、`cpu`、`mps`；初始可能为 `initializing` |
| `error` | 字符串或 `null` | 模型准备失败等错误信息 |

配置加载期间，`options` 仍显示上一次成功配置；当前待加载配置不会提前覆盖它。准备失败后 `status` 为 `failed`，需要再次应用可用配置，不能直接用旧配置继续创建任务。

### 5.3 `PUT /api/inference`

请求体直接为 `InferenceOptions`，外面不再包装 `options`：

```bash
curl --fail-with-body -sS -X PUT "$API_BASE/api/inference" \
  -H 'Content-Type: application/json' \
  -d '{"labels":"indoor","thresh2d":0.3,"track":true}'
```

正常返回 `202` 和 `InferenceInfo`，之后轮询 GET，等待 `ready`。模型加载或推理期间再次 PUT 返回 `409`；服务正在关闭时返回 `503`。

**省略字段使用模型默认值，不会与当前配置合并。** 例如当前 `thresh2d=0.4`，发送 `{"labels":"chair"}` 后，阈值会使用默认的 `0.25`。发送 `{}` 表示重新应用整套默认配置。

如果只想改一个值并保留其他参数，可先 GET，再修改完整 `options` 后 PUT：

```bash
OPTIONS_JSON=$(curl --fail-with-body -sS "$API_BASE/api/inference" |
  jq '.options | .thresh2d = 0.35')

curl --fail-with-body -sS -X PUT "$API_BASE/api/inference" \
  -H 'Content-Type: application/json' --data-binary "$OPTIONS_JSON"
```

读取配置和提交配置之间没有版本锁；多客户端协作时应由调用方协调应用时机。

### 5.4 `InferenceOptions` 完整参数

所有字段均可省略，以下列出解析后的默认值。表中的 `null` 是 JSON 空值。

| 字段 | 类型与约束 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `labels` | 字符串，长度 `1..8192`，清理后至少一个类别 | `"indoor"` | 词表名或逗号分隔的文本提示 |
| `thresh2d` | 数值，`0..1` | `0.25` | 2D 检测置信度阈值 |
| `thresh3d` | 数值，`0..1` | `0.5` | BoxerNet 3D 预测筛选阈值 |
| `xnav_cameras` | 非空字符串 | `"all"` | 全部相机，或 `"0"`、`"0,1"` 等相机 ID 列表 |
| `xnav_camera_mode` | `"native"` / `"virtual_pinhole"` | `"native"` | 原生畸变相机或虚拟针孔视图 |
| `track` | 布尔值 | `true` | 开启跨处理帧的静态物体关联与跟踪 |
| `track_min_hits` | 整数 `>=1` 或 `null` | `null` | `null` 时实际采用 3 次不同处理帧支持作为确认条件 |
| `start_n` | 整数 `>=1` | `1` | 从第几个源处理帧开始，按 1 计数 |
| `skip_n` | 整数 `>=1` | `1` | 源处理帧采样步长 |
| `max_n` | 整数 `>=1` | `99999` | 采样后最多处理的帧数；`0` 不合法 |
| `detector_hw` | 整数 `>=2` | `960` | 保留的 2D 检测尺寸参数，当前实现的实际行为见下文 |
| `force_precision` | `"float32"` / `"bfloat16"` / `null` | `null` | `null` 自动选精度；受设备和模型实现约束 |
| `xnav_execution` | `"auto"` / `"serial"` | `"auto"` | CUDA 自动使用流水线，或强制串行 |
| `xnav_prepare_workers` | 整数 `1` / `2` | `2` | 相机反投影的 CPU 准备线程数；`1` 用于低资源运行和对照 |
| `xnav_detector_batch_size` | 整数 `1` / `2` | `1` | OWLv2 请求批大小；BoxerNet 仍逐观测执行 |
| `xnav_dedup` | `"conservative"` / `"legacy"` | `"conservative"` | 静态多视图跟踪的去重策略 |
| `force_cpu` | 布尔值 | `false` | 强制使用 CPU；否则优先 CUDA，再按环境选择 MPS/CPU |
| `no_sdp` | 布尔值 | `false` | 关闭稀疏深度点输入，以图像执行推理 |
| `no_csv` | 布尔值 | `false` | 跳过检测与跟踪 CSV 导出，仍保存地图包特征、词表与 JSON 回放 |
| `profile` | 布尔值 | `false` | 输出 `performance.json` 性能统计 |
| `ckpt` | 字符串或 `null` | `null` | BoxerNet checkpoint 的服务端路径；空白字符串会转为 `null` |

补充说明：

- `labels` 会去掉每项首尾空白和空项，例如 `" chair, ,table "` 规范为 `"chair,table"`。只有空项的字符串返回 `422`。
- 服务在 `BOXER_LABELS_DIR` 指定目录，或默认仓库 `owl/` 中，查找 `<首个标签>_classes.csv`。找到时加载该词表，每行一个标签；未找到时将各项直接作为文本提示。因此拼错词表名可能成为一个普通检测提示，不一定报“词表不存在”。
- 若首项匹配词表文件，文件内容决定类别列表；不应使用 `"indoor,chair"` 表达合并词表。修改正在使用的词表文件后，需要重新 PUT 应用配置。
- `xnav_cameras` 的字符串格式和相机存在性会在创建地图推理任务时检查。配置状态为 `ready` 并不表示这些相机一定存在于目标地图。
- `native` 保留原生畸变，等比缩放并填充，每路相机一个 `native` 视图，只使用可逆的前半球有效域。`virtual_pinhole` 使用与源图有效域相交的候选针孔视图；`view_id` 从帧观测中读取。
- `detector_hw` 会被接收和记录，但当前 `OwlWrapper.prepare_image()` 使用 OWLv2 权重配置的原生输入尺寸，`forward()` 中的 `resize_to_HW` 没有用于缩放。不要依靠修改此字段来改变实际 OWLv2 输入分辨率。BoxerNet 视图尺寸则由其 checkpoint 决定。
- 自动精度在支持 BF16 的 CUDA 设备上选择 `bfloat16`，其他情况通常为 `float32`；CPU/MPS 上 OWLv2 使用 `float32`。相机几何处理维持 float32。
- CPU/MPS 或 `serial` 模式实际使用检测批大小 1；CUDA 流水线请求批大小 2 时，显存不足可回退到 1。不能把请求值视为实际吞吐承诺。
- `xnav_dedup` 用于静态物体关联。缺少去重信息的旧结果按 `legacy` 解释，回放不会重新运行去重或跟踪。
- `ckpt=null` 使用仓库 `ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt`。该字段只选择 BoxerNet 权重，不是 OWLv2 权重路径。
- HTTP 参数只接受本表字段；CLI 的 `cache2d`、`cache3d`、`live_viz`、`skip_viz`、`write_name` 等不能直接作为 API 参数传入。server 固定采用无桌面预览的任务执行方式。

### 5.5 模型复用与重新加载

模型跨任务复用，跟踪器按任务新建。改变阈值、相机、采样或跟踪设置不重新读取模型权重；改变词表更新文本/语义嵌入并预热；改变 checkpoint 重载 BoxerNet；改变设备重载两种模型，改变有效精度重新准备 OWLv2。

每次服务进程重启使用默认配置，并重新预加载。历史任务配置保存在各自任务记录中，但不会成为新服务的默认配置。worker 在同次服务中异常退出并恢复时，会尝试重新应用本次服务最后成功的配置。

## 6. 地图接口

### 6.1 `GET /api/maps/directories`

无请求参数。默认递归发现 `/nav_map` 下的地图包，按相对路径排序：

```json
{
  "root": "/nav_map",
  "directories": [
    {"name": "园区A/一楼/地图1", "path": "/nav_map/园区A/一楼/地图1"}
  ]
}
```

`name` 用于显示，`path` 可直接传给 `POST /api/maps`。当前查找规则是目录存在 `manifest.txt`，或同时存在 `images/index.csv` 与 `poses_base.txt`；查找不读取图像和点云，不进行完整有效性验证。

不跟随目录符号链接，不可读子树被跳过。根目录自身不作为列表项目返回；若根目录本身就是地图包，直接使用 POST 注册它。空目录返回空列表，根目录不存在返回 `404`，根目录不是目录或无法读取返回 `400`。

发现列表不是已注册地图列表；调用它不会创建地图 ID。当前没有 `GET /api/maps` 列出所有注册地图的接口。

### 6.2 `POST /api/maps`

请求模型为 `MapRequest`：

```json
{"path": "/nav_map/园区A/一楼/地图1"}
```

`path` 必填，至少一个字符。实现会展开 `~` 并解析实际路径；接入时应使用绝对路径。地图必须是完整的 `OMNI_KEYFRAME_MAP` V6 快照，包含合法标定、位姿、图像索引和可用点云。

注册过程中会完整检查图像和关键帧点云，地图较大时请求可能耗时较长。新七列图像索引需要 `poses_image_base.txt`；旧六列索引继续按雷达关键帧时间线读取。详细输入规则见 [x-nav 地图约定](XNAV_MAP.md)。

正常返回 `200` 和 `MapInfo`。以下为示意值：

```json
{
  "id": "0123456789abcdef0123456789abcdef",
  "path": "/nav_map/园区A/一楼/地图1",
  "validation": {
    "map_id": "campus-a",
    "snapshot_id": "snapshot-001",
    "keyframes": 100,
    "selected_keyframes": 100,
    "frame_unit": "image",
    "frames": 300,
    "selected_frames": 300,
    "image_frames": 300,
    "images": {"saved": 600},
    "camera_ids": [0, 1],
    "camera_mode": "native",
    "observations": 600
  },
  "cameras": [
    {"id": 0, "model": "Pinhole"},
    {"id": 1, "model": "Pinhole"}
  ]
}
```

同一实际路径且来源签名相同时复用地图 ID。路径相同但来源签名改变时，重新注册会创建新记录。创建任务或导入结果前会检查地图来源是否改变。

`validation` 是地图注册时按默认全部相机、原生模式进行的验证摘要，不代表后续某个任务的相机选择、采样范围或总帧数；任务规模以 `JobInfo.total_frames` 为准。

| `validation` 字段 | 含义 |
| --- | --- |
| `map_id`、`snapshot_id` | 源地图清单中的业务标识 |
| `keyframes`、`selected_keyframes` | 雷达帧总数、当前选择涉及的不同雷达帧数 |
| `frame_unit` | `image` 表示按同步图像组处理；`lidar` 表示旧雷达帧时间线 |
| `frames`、`selected_frames` | 源处理帧总数、选中的处理帧数 |
| `image_frames` | 图像组数量；旧时间线下为至少有一路已保存图像的帧数 |
| `images` | 各图像状态的计数，如 `saved`、`no_image`、`disabled`、`legacy` |
| `camera_ids`、`camera_mode` | 验证所用相机和模式 |
| `observations` | 可用的相机/视图观测数量 |

### 6.3 `GET /api/maps/{map_id}`

返回该已注册地图的 `MapInfo`，结构与 POST 响应相同。未知 ID 返回 `404`。该接口读取注册信息，不重新完整校验源地图。

### 6.4 `GET /api/maps/{map_id}/scene`

返回 `SceneInfo`，供客户端初始化视图：

```json
{
  "center": [0.0, 0.0, 0.0],
  "radius": 20.0,
  "point_count": 500000,
  "format": "float32-le xyzrgb",
  "up_axis": "+Z",
  "unit": "metre"
}
```

`center` 为展示点云各坐标分量的中位数；`radius` 为相对该中心的最大距离，下限 5 米；`point_count` 是实际返回的展示点数。

优先读取 `global_map.pcd`。文件缺失时使用第一个处理帧关联的关键帧点云。展示点云会移除非有限坐标，并在超过 50 万点时采样。

### 6.5 `GET /api/maps/{map_id}/cloud`

返回场景点云二进制，格式见第 8.5 节。全局点云最多 500,000 点。当前实现不使用 `global_map.pcd` 的颜色，采用固定展示色 `[0.5, 0.55, 0.6]`。

支持 `ETag` 和 `If-None-Match`；内容相同可返回 `304`，此时没有响应体。

### 6.6 ZIP 地图包分块上传与续传

每个 ZIP 包含一张完整 V6 地图：`manifest.txt` 可以直接位于 ZIP 根目录，或位于一个外层文件夹内。支持 ZIP64、Stored 和 Deflate；拒绝多地图、加密包、其他压缩方法、重复目标、危险路径、符号链接和特殊文件。地图中的新旧图像索引沿用现有校验规则。

默认分块为 **8 MiB**，单 ZIP 上限 **20 GiB**，解压上限 **100 GiB**，最多 **200,000 个条目**。`GET /api/maps/uploads` 返回 `{ "limits": {...}, "uploads": [...] }`，客户端应使用服务返回的参数。部署配置见 [Docker 文档](DOCKER.zh-CN.md#地图上传配置)。

**创建会话**：

```http
POST /api/maps/uploads
Content-Type: application/json

{"filename":"园区A.zip","size":12345678,"sha256":"<整个 ZIP 的 64 位十六进制 SHA-256>"}
```

`filename` 必须是 `.zip` 文件名，不能包含目录；`size` 为正整数字节数，`sha256` 必填。新建请求总是创建独立会话；续传时应保存并复用返回的 `id`。

**上传分块**：`index` 从 0 开始，块大小以会话的 `chunk_size` 为准，最后一块取剩余长度。

```bash
curl --fail-with-body -X PUT "$API_BASE/api/maps/uploads/$UPLOAD_ID/chunks/0" \
  -H 'Content-Type: application/octet-stream' \
  -H "X-Chunk-SHA256: $CHUNK_SHA256" --data-binary @chunk-0.bin
```

接口流式读取，不使用 multipart。分块落盘、同步并原子提交记录后才返回成功；支持乱序及并发上传。同一块相同内容重复上传成功返回，不同校验值返回 `409`。长度或哈希错误不会记为已接收；断连时只需重传该块。

所有写操作和单会话查询返回 `UploadInfo`：

| 字段 | 含义 |
| --- | --- |
| `id`、`filename`、`size`、`sha256` | 会话与原始 ZIP 标识 |
| `chunk_size`、`total_chunks` | 本次会话的分块参数，配置改变后旧会话仍沿用原参数 |
| `received_chunks`、`received_bytes` | 已持久化的分块编号及字节数 |
| `status` | `uploading` / `processing` / `complete` / `failed` / `cancelled` |
| `stage` | `uploading`、`queued`、`merging`、`extracting`、`validating`、`publishing` 或终态 |
| `progress` | `0..1`；上传时为已接收字节比例，处理时为当前阶段比例 |
| `created_at`、`expires_at` | UTC 创建时间和闲置清理时间；正在处理的会话暂不清理 |
| `error`、`error_code` | 失败原因与对应 HTTP 错误码；正常时为 `null` |
| `map` | 完成后的 `MapInfo`，包含可直接创建任务的 `map.id`；完成前为 `null` |

**完成处理**：调用 `POST /api/maps/uploads/{upload_id}/complete`。缺块或仍有块正在写入返回 `409`；受理后返回 `202`，通过 GET 查询后台处理进度。合并时重查每块及整包 SHA-256；损坏块会从已接收列表移除，状态回到 `uploading`，补传后再次 complete。解压或地图校验失败时，GET 本身仍返回 `200`，会话的 `status=failed`，应检查 `error_code`；保留期限内可以重试 complete 或取消。

校验通过后，地图原子发布到 `<map_root>/uploads/<安全化包名>-<upload_id>` 并注册，不覆盖已有地图。重复 complete 返回同一结果及 `map_id`，不会重复处理已经完成的上传。发布目录中的 `.boxer-upload.json` 为服务恢复标记，应随地图保留。

**恢复与清理**：服务重启后保留已提交分块，自动恢复已经受理的处理流程。客户端重新选择原 ZIP，核对大小及整包哈希，查询 `received_chunks` 后只传缺块。默认连续 **24 小时没有成功数据写入或提交处理**的未完成会话清理；GET 不续期。成功后立即清理分块、合并 ZIP 和解压暂存，完成记录保留 24 小时；记录清理不会删除地图。取消正在写入或处理的上传时先标记取消，待使用文件的操作退出后清理。已发布地图不能通过 DELETE 上传接口删除。

服务的 `.boxer_uploads/` 暂存目录不会出现在地图发现列表；发布过程异常时会优先保留已发布目录，允许恢复注册。空间检查覆盖分块、合并和解压，峰值需求约为 `2 × ZIP 大小 + 解压大小`，多个上传还需叠加占用。

**Python 客户端**：示例只依赖标准库，运行相同命令可凭本地记录续传，支持服务重启：

```bash
python scripts/upload_xnav_map.py /path/to/园区A.zip --url http://SERVER_IP:18765
# 默认在 ZIP 旁保存 .upload.json，也可指定：
python scripts/upload_xnav_map.py /path/to/园区A.zip \
  --url http://SERVER_IP:18765 --state /path/to/upload-state.json
```

成功打印 `MapInfo`；`--new` 创建独立上传。网络故障或 `5xx` 按 1、2、4 秒退避重试，超出后保留记录，重新运行即可恢复。Web 页面提供相同能力，刷新后需要重新选择原文件才能继续传块，完成后自动加载地图。

## 7. 任务管理接口

### 7.1 `POST /api/jobs`

推荐只传入已注册的地图 ID，使用服务当前已应用的配置：

```json
{"map_id": "0123456789abcdef0123456789abcdef"}
```

请求模型 `JobRequest`：

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `map_id` | 是 | 地图注册接口返回的 `id` |
| `options` | 否 | `InferenceOptions` 或 `null`；省略/null 时使用已应用配置 |

显式提供 `options` 时，解析并补全默认值后的完整配置必须与已应用配置一致，否则返回 `409`。例如服务使用自定义词表时，`options: {}` 会被补成默认 `indoor` 配置，通常不能用于启动该任务。

模型未就绪返回 `503`；已有活动任务返回 `409`。相机不在标定中、选择范围为空、地图来源改变等问题返回 `400`。创建任务不会自动应用新参数。

正常返回 `202` 和 `JobInfo`；返回时通常为 `starting`。仍需检查响应中的 `status` 和后续任务状态：worker 发送失败等异常也可能使此次返回的任务已经为 `failed`。

API 2.0.0 统一由此接口派发物体检测、体素连通分割和几何归属。请求体保持不变；分割读取 `config/voxel_connectivity.yaml`，可由 `BOXER_VC_CONFIG` 指定。提交前检查模型、地图、`global_map.pcd`、解释器、配置和两个任务槽。模型未就绪返回 `503`，地图/全局点云不存在返回 `404`，占槽返回 `409`。受理后的阶段失败通过 `workflow` 报告，另一项继续运行。

### 7.2 `GET /api/jobs`

无请求参数，返回 `JobInfo[]`，同时包含推理任务和导入的只读任务。当前按内存记录插入顺序逆序返回，没有分页或过滤参数。服务重启从磁盘恢复后的顺序不应作为严格的创建时间排序；需要时使用 `created_at` 自行排序。

### 7.3 `GET /api/jobs/{job_id}`

返回单个 `JobInfo`。示例中的 ID、路径、设备和计数仅作说明：

```json
{
  "id": "abcdef0123456789abcdef0123456789",
  "map_id": "0123456789abcdef0123456789abcdef",
  "status": "running",
  "readonly": false,
  "created_at": "2026-11-04T02:00:00+00:00",
  "output_dir": "/app/output/server/jobs/abcdef0123456789abcdef0123456789/output",
  "results_path": "/app/output/server/jobs/abcdef0123456789abcdef0123456789/output/campus-a/snapshot-001/native/results.json",
  "options": {"labels": "indoor", "track": true},
  "processed_frames": 12,
  "total_frames": 100,
  "elapsed_seconds": 20.5,
  "tracking": true,
  "device": "cuda",
  "error": null,
  "workflow": null
}
```

示例为没有关联工作流的历史任务，并简写了 `options`；普通推理任务实际保存完整解析配置，导入任务则只保存部分历史信息。新任务的 `workflow` 见第 15 节。

| 字段 | 说明 |
| --- | --- |
| `id`、`map_id` | 任务 ID、服务注册的地图 ID |
| `status` | 任务生命周期状态，见下表 |
| `readonly` | `true` 表示导入的回放结果 |
| `created_at` | 服务创建该任务记录的 UTC 时间；导入任务不是原推理时间 |
| `output_dir` | 任务输出根目录；导入时为已有结果所在目录 |
| `results_path` | 服务器本地 `results.json` 路径，不能直接当作 HTTP 下载地址 |
| `options` | 该任务的配置快照，查看历史不会改变当前服务配置 |
| `processed_frames` | 已持久化并可读取的帧数，不是 GPU 已计算但尚未提交的帧数 |
| `total_frames` | 当前任务按采样配置选中的处理帧总数；导入时为当前已提交帧数 |
| `elapsed_seconds` | 已提交结果中的累计耗时，随结果提交更新 |
| `tracking` | 是否包含跟踪结果；优先取保存结果中的跟踪设置 |
| `device` | 执行设备或保存结果中的设备信息 |
| `error` | 错误详情，无错误通常为 `null` |
| `workflow` | 联合流程状态与阶段；新任务 `schema_version=2`，无联合流程时为 `null` |

| 状态 | 含义 | 客户端处理 |
| --- | --- | --- |
| `starting` | 任务已记录，等待 worker 开始 | 继续等待 |
| `running` | 正在推理、提交帧或进行收尾 | 读取已提交结果并继续等待 |
| `stopping` | 已请求中断，正在等待退出 | 继续等待最终状态 |
| `complete` | Boxer 推理、输出关闭及要求的地图 CSV 导出完成；旧任务还包含其 ZIP 收尾 | 继续检查联合流程的保存与归档状态 |
| `failed` | 推理、结果读取、导出或 ZIP 归档失败 | 读取 `error` 和日志；已提交帧可能仍可回放 |
| `interrupted` | 用户停止、服务重启或其他中断 | 已提交帧保留，不会自动续推 |

上述状态描述 Boxer 阶段。新联合流程须等待 `workflow.status` 的 `complete`、`failed` 或 `interrupted`；历史任务与只读导入任务的 `workflow` 为 `null`，按顶层状态处理。`processed_frames == total_frames` 不等于成功：输出收尾、地图 CSV 导出、ZIP、分割、绑定或地图保存仍可能未结束。`no_csv=true` 跳过检测 CSV 导出和 ZIP，仍执行分割、绑定及房间地图保存。

### 7.4 `POST /api/jobs/{job_id}/stop`

不需要请求体，正常返回 `202` 和当前 `JobInfo`：

```bash
curl --fail-with-body -sS -X POST "$API_BASE/api/jobs/$JOB_ID/stop"
```

活动任务先进入 `stopping`，向正在执行的推理发送 SIGINT。默认等待最多 60 秒，然后向 worker 进程组发送 SIGTERM，5 秒后仍未退出则升级 SIGKILL。正常中断后可以复用模型；worker 被强制结束后会重新准备模型。

联合流程的 `workflow.status` 进入 `stopping`，同时停止仍运行的 Boxer、关联分割与绑定，并取消未发布的地图导出和 ZIP。即使 Boxer 已完成也能停止剩余阶段。分割和绑定先发送 SIGTERM，超时升级 SIGKILL；已经成功发布的计算结果和地图目录保留。

如果任务仍在 `starting`，服务会在收到 worker 的开始通知后再发送中断。重复停止或停止已结束的普通推理任务返回其当前摘要，不会重新执行推理。导入任务为只读，调用此接口返回 `409`。

### 7.5 `GET /api/jobs/{job_id}/log`

统一通过父任务 ID 读取各阶段日志，响应为 `{"text":"…"}`。`stage` 可取 `boxer`（默认）、`room_segmentation`、`binding`；Boxer 返回日志末尾最多约 12,000 字节，房间分割和归属返回末尾最多 64 KiB。尚无阶段日志时返回空字符串；无效阶段返回 `422`。模型加载与配置日志另见服务输出根目录的 `worker.log`。

```bash
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/log?stage=room_segmentation"
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/log?stage=binding"
```

### 7.6 `POST /api/jobs/import`

请求模型 `ImportRequest`：

```json
{
  "map_id": "0123456789abcdef0123456789abcdef",
  "results_path": "/app/output/server/jobs/已有任务/output/campus-a/snapshot-001/native/results.json"
}
```

两个字段均必填，`results_path` 至少一个字符，为服务端文件路径。正常返回 `200` 和新创建的只读 `JobInfo`。

导入会检查地图来源签名、结果格式、预处理版本及相机模式/半球策略，结果与地图不匹配时拒绝读取。支持历史完整结果和仍在增长的已提交日志结果；没有已提交帧的合法运行中结果也可导入。

导入只读取保存的检测、轨迹和 ID，不加载模型执行新推理，不重新运行跟踪器，也不向地图导出 CSV。导入不要求模型为 `ready`，可在模型准备失败时用于回放已有数据。

运行中的导入结果会继续刷新已提交帧，其 `status` 随源结果变化，`total_frames` 为当前可读帧数。`readonly` 表示无法通过该任务控制源推理；需要停止源任务时，应使用源推理任务 ID。

### 7.7 `GET /api/jobs/{job_id}/events`

这是 Server-Sent Events 流，连接建立后立即发送当前 `JobInfo`，之后仅在摘要变化时发送新的摘要：

```text
event: job
data: {"id":"...","status":"running","processed_frames":12,"total_frames":100,...}

```

以上 `data` 为结构示意；实际每条 `data:` 后是完整合法的单行 JSON。事件名固定为 `job`，不是默认的 `message`。

```bash
curl --fail-with-body -N "$API_BASE/api/jobs/$JOB_ID/events"
```

- 服务每约 0.5 秒检查一次摘要变化，无变化时约每 15 秒发送 `: heartbeat` 注释。
- 响应包含 `Cache-Control: no-cache` 和 `X-Accel-Buffering: no`；使用反向代理时也要允许流式转发。
- 断线重连重新发送最新快照，没有历史事件补发序号，也没有基于 `Last-Event-ID` 的增量恢复。
- 事件不包含完整帧、轨迹和点云。客户端通过 `processed_frames` 的变化再读取帧接口。
- 任务结束后流不会自动关闭；客户端收到结束状态后应主动关闭连接。
- 未知任务在建立流之前返回 `404`。

### 7.8 `GET /api/jobs/{job_id}/3dbbox.zip`

无请求体和查询参数。新联合任务在 `no_csv=false` 且 `workflow.status=complete` 后，提供完成时源地图
`3dbbox/` 的持久化 ZIP 快照，包含本次房间分割、绑定与来源记录。Boxer 顶层 `status=complete` 仅说明自身输出和 CSV 已完成；此时可能仍不能下载 ZIP。下载不会重新压缩当前地图目录。同一地图后续重跑或服务
重启不会改变该任务的 ZIP 内容。

```bash
# 新任务先等待 workflow.status == "complete"；旧任务仍按 status 判断。
curl --fail-with-body -sS \
  "$API_BASE/api/jobs/$JOB_ID/3dbbox.zip" -o "3dbbox-$JOB_ID.zip"
```

成功响应为 `200 application/zip`，响应头为
`Content-Disposition: attachment; filename="3dbbox-<job_id>.zip"`。
ZIP 保留顶层 `3dbbox/` 及业务子目录，包含四个检测/跟踪 CSV、`labels.csv`、
`dino_features.csv` 和目录内其他业务文件；空结果 CSV 仍保留表头。
DINO 特征保留归档时的地图级累计行，并不只包含该任务新增的行。新任务还包含
`3dbbox/room_segmentation/` 的五个文件，见第 15.4 节。
内部 `.dino_features.lock`、`.boxer-*` 暂存项和原子写入临时文件不进入 ZIP；
遇到符号链接、特殊文件或不可读文件时归档失败。

| 情况 | 状态码 | 客户端处理 |
| --- | --- | --- |
| 联合流程成功完成且归档存在；旧任务按原规则 | `200` | 保存二进制 ZIP |
| 新联合流程未成功完成，或旧任务的 Boxer 未成功完成 | `409` | 即使 Boxer 已完成，也须等待地图保存及归档 |
| `no_csv=true` | `409` | 此类任务不生成 ZIP，即使目录保留旧 CSV 也不提供下载 |
| 未知任务、导入的只读任务、历史任务无归档或归档已删除 | `404` | 不会用当前地图目录补生成历史 ZIP |

新联合流程中，Boxer 后台导出 CSV 后进入 `complete`；待绑定完成，再由工作流后台保存房间结果并统一压缩。期间 `workflow.status` 保持 `running`，查询和健康检查仍可用，新联合任务须等待收尾。ZIP 使用 DEFLATE、ZIP64 和分块读写，
临时写入后原子发布；归档及完成状态持久化成功后，整体才进入 `complete`。
磁盘空间、权限等保存或归档错误会使整体进入 `failed`，`workflow.map_export.error` 记录原因，已经成功的计算结果保留。
收尾期间可以请求停止，任务经过 `stopping` 进入 `interrupted`，未发布 ZIP 被清理；
此时地图 CSV 或房间目录可能已经成功发布。服务关闭时取消并等待收尾退出，重启不自动续做归档。历史任务已有 ZIP 不补包或重写。

## 8. 帧数据、检测框、图像与点云接口

### 8.1 `GET /api/jobs/{job_id}/frames`

查询参数：

| 参数 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| `offset` | 整数 | `0` | `>=0` |
| `limit` | 整数 | `200` | `1..1000` |

返回 `FramePage`，其中 `total` 是当前已提交帧总数：

```json
{
  "total": 12,
  "offset": 0,
  "frames": [
    {
      "index": 0,
      "frame_id": 2,
      "image_frame_id": 2,
      "lidar_keyframe_id": 1,
      "time_ns": "1719249712510151726",
      "position": [10.0, 0.0, 0.0]
    }
  ]
}
```

示例只列出一项。`position` 为 `T_map_base` 的平移部分，用于轨迹绘制。实时读取时 `total` 可以持续增长；下一页可用 `offset + len(frames)` 继续读取。

`offset >= total` 返回空数组。第一个结果描述文件尚未发布时返回 `404`，提示等待首个已提交结果；已经发布合法空结果时返回 `total: 0, frames: []`。

### 8.2 `GET /api/jobs/{job_id}/frames/{index}`

路径 `index` 是回放索引，取值应为 `0..processed_frames-1`。负值、越界或尚未提交的帧返回 `404`；非整数路径参数返回 `422`。

返回 `FrameData`：

```json
{
  "index": 0,
  "frame_id": 2,
  "image_frame_id": 2,
  "lidar_keyframe_id": 1,
  "time_ns": "1719249712510151726",
  "T_map_base": [
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "observations": [],
  "tracks": [],
  "detections": []
}
```

空观测和空检测是合法结果。若源图像不可用，帧仍可存在，`observations` 为空；此时图像接口返回 `404`。

`detections` 为该处理帧多视图融合后的检测，`tracks` 为保存的跟踪状态；单个观测内还有其自身的 `detections`。字段细节见第 9 节。

### 8.3 `GET /api/jobs/{job_id}/frames/{index}/images/{kind}`

`kind` 只能是 `2d` 或 `3d`：

| 值 | 返回内容 |
| --- | --- |
| `2d` | 当前观测图像叠加 2D 检测框和类别 |
| `3d` | 当前观测图像叠加 3D 检测或跟踪框的投影 |

查询参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `camera_id` | 整数，可省略 | 不限制 | 匹配该帧观测中的相机 ID |
| `view_id` | 字符串，可省略 | 不限制 | 匹配该帧观测中的视图 ID |
| `tracked` | 布尔值 | `true` | 3D 叠加选择 `tracks`；`false` 选择原始帧级 `detections` |
| `tentative` | 布尔值 | `false` | 跟踪显示时是否包含 `TENTATIVE` 候选轨迹 |

未指定相机和视图时，取观测列表中首个符合条件的观测。为保证读取指定视角，先读取 `FrameData.observations`，再同时传入 `camera_id` 和 `view_id`。

```bash
curl --fail-with-body -sS \
  "$API_BASE/api/jobs/$JOB_ID/frames/0/images/3d?camera_id=0&view_id=native&tracked=true&tentative=false" \
  -o projection.jpg
```

匹配不到观测返回 `404`；非法 `kind` 返回 `422`。`tracked` 和 `tentative` 不改变 2D 面板的检测内容，也不触发重新跟踪。

响应为 JPEG，当前编码质量为 88，支持 ETag。图像按保存结果的预处理方式重建：原生模式为缩放、填充并屏蔽无效域后的畸变图像，虚拟模式为对应针孔视图，并非直接下载源 JPEG。

关闭跟踪的任务默认 `tracks` 为空；查看其 3D 检测投影时应传 `tracked=false`。

### 8.4 `GET /api/jobs/{job_id}/frames/{index}/cloud`

无查询参数。返回当前处理帧关联的雷达关键帧点云，已变换到地图坐标系，最多 50,000 点。优先使用彩色 `rgb` 点云，缺失或无可用点时回退 `surf`；无颜色时使用固定展示色。

```bash
curl --fail-with-body -sS \
  "$API_BASE/api/jobs/$JOB_ID/frames/0/cloud" -o frame-0.xyzrgb.bin
```

该点云用于可视化，来自整个关联关键帧，并非模型最终筛选给某一路相机的稀疏深度点集合。同一雷达锚点对应的不同图像组可能显示相同点云。

### 8.5 二进制点云布局

全局和逐帧 cloud 接口使用相同布局：无文件头，每点 6 个连续的 little-endian `float32`，共 24 字节。

| 字节偏移 | 字段 | 类型与范围 |
| --- | --- | --- |
| `0` | `x` | float32，地图坐标，米 |
| `4` | `y` | float32，地图坐标，米 |
| `8` | `z` | float32，地图坐标，米 |
| `12` | `r` | float32，颜色 `0..1` |
| `16` | `g` | float32，颜色 `0..1` |
| `20` | `b` | float32，颜色 `0..1` |

点数为 `响应字节数 / 24`。这不是 PCD，也不是三个 uint8 颜色字节。点数上限仅影响展示，不修改源地图和模型结果。

Python 解码：

```python
import numpy as np
import requests

response = requests.get(
    f"{API_BASE}/api/jobs/{JOB_ID}/frames/0/cloud", timeout=120
)
response.raise_for_status()
if len(response.content) % 24:
    raise ValueError("Invalid cloud byte length")
vertices = np.frombuffer(response.content, dtype="<f4").reshape(-1, 6)
xyz = vertices[:, :3]
rgb = vertices[:, 3:]
```

这里的 `API_BASE`、`JOB_ID` 是 Python 字符串变量，需按实际环境设置。

### 8.6 ETag 缓存

图像和点云响应带有基于响应体 SHA-256 的 `ETag`，并设置 `Cache-Control: private, no-cache`。再次请求相同资源时，可以原样发送该 ETag：

```bash
curl --fail-with-body -sS -D cloud.headers \
  "$API_BASE/api/maps/$MAP_ID/cloud" -o map.xyzrgb.bin

# 将下面的值替换成首次响应的 ETag，保留双引号。
curl -i -H 'If-None-Match: "首次响应的完整哈希"' \
  "$API_BASE/api/maps/$MAP_ID/cloud"
```

完全匹配时返回 `304`，客户端复用已缓存的响应体。当前实现按完整 header 值比较，客户端应原样回传单个 ETag。`304` 属于运行时行为，当前 OpenAPI schema 未单独声明它。

每个加载地图对图像和逐帧点云采用最多 32 MiB 的资源缓存，淘汰后可按需重建；全局场景点云另行保留。

### 8.7 检测框 JSON 查询

以下四个接口通过服务任务 `job_id` 查询结果，均支持推理运行中读取：

| GET 路径 | 响应模型 | `items` 内容 |
| --- | --- | --- |
| `/api/jobs/{job_id}/3dbbox_tracked` | `Box3DPage` | 当前确认的全局物体，推理结束后为最终物体 |
| `/api/jobs/{job_id}/3dbbox_tracks` | `Box3DPage` | 各已提交帧的非 `TENTATIVE` 跟踪框，包括保留的 `INACTIVE` 轨迹 |
| `/api/jobs/{job_id}/3dbbox` | `Box3DPage` | 各已提交帧多视图融合后的 3D 检测 |
| `/api/jobs/{job_id}/owl_2dbbox` | `Box2DPage` | 各已提交帧、相机和视图的 OWL 2D 检测 |

首个接口名称统一为 `3dbbox_tracked`。四个接口均返回 `application/json`，`items` 使用对应 CSV 的全部列名。3D 浮点数取自已保存的 JSON；带来源信息的新 2D 结果按 CSV 的坐标两位、概率六位小数输出。

#### 分页和读取时机

查询参数为 `offset`（整数，默认 `0`，最小 `0`）和 `limit`（整数，默认 `200`，范围 `1..1000`），均按**检测框记录数**计数。逐帧接口保持帧提交顺序以及帧内保存的检测框、相机和视图顺序。

共同响应字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `status` | 任务状态字符串 | 与 `JobInfo.status` 相同，包含 `running`、`complete`、`failed`、`interrupted` 等状态 |
| `processed_frames` | 整数 | 本次快照包含的已提交帧数 |
| `total` | 整数 | 本次快照中该类检测框的总记录数 |
| `offset` | 整数 | 本次请求的记录偏移 |
| `limit` | 整数 | 本次请求的最大记录数 |
| `items` | 对象数组 | 本页检测框，长度不超过 `limit` |
| `provenance_version` | 整数或 null | 新来源关联格式为 `1`；旧结果为 null，不伪造缺失关联 |

未知任务、任务结果尚未建立时返回 `404`。已建立的合法空快照、无检测、关闭跟踪时的跟踪结果，以及 `offset >= total` 均返回 `200` 和空 `items`。非法分页参数返回 `422`。失败和中断任务仍可查询已提交前缀。

四个接口复用服务定时刷新的回放缓存和提交日志，仅返回已提交数据；单次响应内的计数和记录来自同一快照。`no_csv=true` 或只导入 JSON 历史结果时同样可用。

`3dbbox_tracked` 在运行中使用最新已提交帧的非 `TENTATIVE` 轨迹。结果状态为 `complete` 且存在 `objects` 字段时，以该字段为准，包括空数组；较旧结果缺少该字段时使用最新帧作为兼容回退。关闭跟踪时返回空列表。

实时全局物体会随确认、融合发生变化，应从 `offset=0` 重新获取当前集合；分页请求之间不保证属于同一快照。逐帧跟踪保留各帧当时的实例 ID，后续 ID 合并不会改写历史记录。`status` 是服务任务状态，地图 CSV 导出失败时可能为 `failed`，此时仍可读取已完成的推理物体。

#### `Box3DRow`

三类 3D 接口共用以下字段，对应 `boxer_3dbbs_tracked.csv`、`boxer_3dbbs_tracks.csv` 和 `boxer_3dbbs.csv`：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `time_ns` | 字符串 | 新结果为来源图像时间：融合框取全部来源最早时间，轨迹取最近匹配组最早时间，最终物体取完整历史最早时间；旧结果保留原位姿时间/全局 `"0"` |
| `tx_world_object`、`ty_world_object`、`tz_world_object` | 数值 | 保存的 `center`，地图坐标系，单位米 |
| `qw_world_object`、`qx_world_object`、`qy_world_object`、`qz_world_object` | 数值 | 保存的物体旋转四元数，顺序为 `wxyz` |
| `scale_x`、`scale_y`、`scale_z` | 数值 | 保存的 `size`，物体局部轴完整边长，单位米 |
| `name` | 字符串 | 保存的类别文本 `label` |
| `instance` | 整数 | 保存的 `track_id`；未跟踪检测通常为 `-1` |
| `sem_id` | 整数 | 保存的语义 ID |
| `prob` | 数值 | 保存的框置信度 |
| `bbox3d_id` | 字符串或 null | 融合 3D 检测的稳定 ID；轨迹/最终物体为 null |
| `object_id` | 整数或 null | 解析别名后的最终物体 ID；未归属确认物体时为 null |
| `source_detection_id`、`image_file` | 字符串或 null | 该行时间对应的 2D 检测 ID，以及相对地图根目录的原图路径 |
| `frame_id`、`pose_time_ns` | 整数/纳秒字符串或 null | 逐帧记录的处理帧与位姿时间；最终物体使用第一条来源观测 |
| `observations_2d` | 对象数组或 null | 全局物体的全部关联 2D 检测，包含确认前及被合并轨迹的观测；逐帧结果/旧结果为 null |

`observations_2d` 每个元素包含对应 `Box2DRow` 的完整字段及 `bbox3d_id`，以
`detection_id` 唯一对应 OWL 行。按图像时间、帧、相机、视图和检测序号排序，
同图的不同框和视图分别保留。其首项的时间和图像路径等于最终物体的
`time_ns` 和 `image_file`。原图从原地图包读取；ZIP 不复制 `images/`。
未匹配的新轨迹快照保留上次实际观测时间，同时更新其处理帧和 `pose_time_ns`。

例如查询当前确认物体：

```bash
curl --fail-with-body -sS \
  "$API_BASE/api/jobs/$JOB_ID/3dbbox_tracked?offset=0&limit=200" | jq
```

```json
{
  "status": "complete",
  "processed_frames": 12,
  "provenance_version": 1,
  "total": 1,
  "offset": 0,
  "limit": 200,
  "items": [{
    "time_ns": "1793948371675919357",
    "tx_world_object": 1.125, "ty_world_object": -2.5, "tz_world_object": 3.75,
    "qw_world_object": 0.8, "qx_world_object": 0.2,
    "qy_world_object": 0.4, "qz_world_object": 0.4,
    "scale_x": 0.6, "scale_y": 1.2, "scale_z": 2.4,
    "name": "chair", "instance": 17, "sem_id": 1, "prob": 0.8123456789,
    "object_id": 17, "source_detection_id": "f3:c1:vnative:d0",
    "image_file": "images/3_cam1.jpg", "frame_id": 3,
    "pose_time_ns": "1793948371675919000",
    "observations_2d": [{
      "time_ns": "1793948371675919357", "frame_id": 3,
      "sensor": "cam1/native", "device": "x-nav",
      "img_width": 960, "img_height": 960,
      "x1": 10.25, "y1": 20.5, "x2": 60.75, "y2": 90.25,
      "name": "chair", "instance": -1, "sem_id": 1, "prob": 0.91,
      "detection_id": "f3:c1:vnative:d0", "object_id": 17,
      "camera_id": 1, "view_id": "native", "image_file": "images/3_cam1.jpg",
      "pose_time_ns": "1793948371675919000", "bbox3d_id": "f3:b0"
    }]
  }]
}
```

#### `Box2DRow`

`owl_2dbbox` 的字段对应 `owl_2dbbs.csv`：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `time_ns` | 字符串 | 当前相机观测的原始图像时间 `image_stamp_ns` |
| `frame_id` | 整数 | 源处理帧 ID；独立图像采样时是图像组编号，旧结果兼容 `keyframe_id` |
| `sensor` | 字符串 | `cam<camera_id>/<view_id>`，例如 `cam1/native` |
| `device` | 字符串 | 固定为 `x-nav` |
| `img_width`、`img_height` | 整数 | 保存的预处理视图尺寸 `parameters.image_size`，缺失时为 `-1` |
| `x1`、`y1`、`x2`、`y2` | 数值 | 预处理视图上的左上角和右下角像素坐标 |
| `name` | 字符串 | 保存的 OWL 类别文本 |
| `instance` | 整数 | 固定为 `-1` |
| `sem_id` | 整数 | 按结果中保存的 `parameters.labels` 顺序映射；缺少词表或类别未收录时为 `-1` |
| `prob` | 数值 | 保存的 OWL 置信度 |
| `detection_id` | 字符串或 null | 本次结果集内唯一的 2D 检测 ID |
| `object_id` | 整数或 null | 对应最终物体 ID；未关联/未确认时为 null |
| `camera_id`、`view_id` | 整数/字符串或 null | 来源相机及预处理视图 |
| `image_file` | 字符串或 null | 相对原地图根目录的原图路径，例如 `images/12_cam0.jpg` |
| `pose_time_ns` | 字符串或 null | 原处理帧位姿时间 |

`GET /api/jobs/{job_id}/owl_2dbbox?object_id=17` 可筛选该物体的全部关联 2D 检测，
支持相同的 `offset`/`limit` 分页。旧结果不补造来源，该筛选返回空列表。

保存的 `boxes_2d` 顺序为 `[xmin,xmax,ymin,ymax]`，接口对应 `x1=xmin`、`y1=ymin`、`x2=xmax`、`y2=ymax`。这些坐标对应预处理视图，包含缩放和填充；不能直接叠加到未处理的源图像上。

```bash
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/3dbbox_tracks?offset=0&limit=200" | jq
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/3dbbox?offset=0&limit=200" | jq
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/owl_2dbbox?offset=0&limit=200" | jq
```

## 9. 数据模型与坐标约定

### 9.1 `Observation`

每项 `FrameData.observations` 对应一个实际可用的相机/视图：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `camera_id` | 整数 | 源标定中的相机 ID |
| `view_id` | 字符串 | 原生模式为 `native`，虚拟模式使用实际视图名称 |
| `image_stamp_ns` | 字符串 | 该路源图像的原始纳秒时间戳 |
| `T_map_view` | `4×4` 数值数组 | 视图坐标到地图坐标的齐次变换 |
| `boxes_2d` | 数组，每项 4 个数 | 顺序为 `[xmin, xmax, ymin, ymax]`，单位像素 |
| `scores_2d` | 数值数组 | 与 2D 框逐项对应的 OWLv2 置信度 |
| `labels_2d` | 字符串数组 | 与 2D 框逐项对应的类别 |
| `detections` | `BoxRecord[]` | 当前观测的 3D 检测；尚未做整帧跨视图融合 |

2D 框采用 Boxer 的 `[xmin, xmax, ymin, ymax]` 顺序。坐标基于预处理后的视图图像，正好对应图像接口返回的面板；不应直接叠加到未处理的源图像上。

当前新结果还可能包含 `frame_id`、`keyframe_id`、`image_frame_id`、`lidar_keyframe_id` 等来源字段，客户端应容忍新增字段。

### 9.2 `BoxRecord`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `center` | 长度 3 的数值数组 | 有向 3D 框中心，地图坐标，米 |
| `quaternion_wxyz` | 长度 4 的数值数组 | 物体局部坐标到地图坐标的旋转四元数，`wxyz` |
| `size` | 长度 3 的数值数组 | 物体局部 X/Y/Z 轴的完整边长，米 |
| `label` | 字符串 | 类别文本 |
| `sem_id` | 整数 | 此结果类别集合内的语义 ID，不应视为跨词表固定编码 |
| `prob` | 数值 | 保存的框置信度 |
| `track_id` | 整数 | 跟踪实例 ID；未跟踪检测通常为 `-1` |
| `state` | 字符串或 `null` | 当前跟踪实现使用 `TENTATIVE`、`ACTIVE`、`INACTIVE`；检测记录可为空 |

轨迹记录还可包含 `support_count`、`first_seen_frame`、`last_seen_frame`、`missed_count`、`visible` 等附加信息。`FrameData`、`Observation`、`BoxRecord` 允许保存结果中的额外字段，因此外部客户端应以核心字段解析并允许扩展。

3D 单观测检测先按 BoxerNet 预测值应用 `thresh3d`，之后将该预测值与对应 2D 分数取均值保存为 `prob`。整帧融合和跟踪还可能进一步聚合分数，因此返回 `prob` 不一定大于 `thresh3d`，也不是原始 BoxerNet 分数。

默认 3D 图像隐藏 `TENTATIVE`，显示已确认及保留的非候选轨迹。直接读取 `tracks` JSON 时，调用方根据 `state` 自行过滤；`tentative=false` 只控制图像渲染，不修改帧 JSON。

### 9.3 位姿、框和时间线

对列向量，`p_map = T_map_view × p_view`，`T_map_base` 则把该处理帧的 base 坐标映射到地图。新图像时间线中：

```text
相机位姿：T_map_view = T_map_image_base × T_base_camera × T_camera_view
点云位置：p_map = T_map_lidar_base × p_lidar_base
```

新地图使用已导出的图像组位姿；雷达点云使用独立雷达锚点位姿。不要把已经在地图系中的框或 cloud 响应再乘一次 `T_map_base`。

保存结果中的 `keyframe_id` 是处理帧的兼容别名；源 `images/index.csv` 的同名列仍表示雷达锚点。当前 API 明确提供 `frame_id`、`image_frame_id` 和 `lidar_keyframe_id` 来区分这些身份。

## 10. 客户端接入示例

### 10.1 Python 完整任务流程

示例需要 `requests`，在调用方 Python 环境执行 `python -m pip install requests` 安装。脚本执行地图注册、参数准备、任务创建和等待，完成后读取首帧；任务失败或中断时抛出错误。

```python
import json
import time

import requests

API_BASE = "http://127.0.0.1:18765"
MAP_PATH = "/nav_map/园区A/地图1"
session = requests.Session()


def api(method, path, **kwargs):
    response = session.request(
        method, f"{API_BASE}{path}", timeout=(10, 300), **kwargs
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def wait_ready(timeout_seconds=900):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = api("GET", "/api/inference")
        if state["status"] == "ready":
            return state
        if state["status"] == "failed":
            raise RuntimeError(state["error"])
        time.sleep(1)
    raise TimeoutError("Model preparation timed out")


wait_ready()
map_info = api("POST", "/api/maps", json={"path": MAP_PATH})

# 从完整已应用配置开始修改，保留其他字段。
options = api("GET", "/api/inference")["options"]
options.update(labels="chair,table,door", max_n=30, track=True)
api("PUT", "/api/inference", json=options)
wait_ready()

job = api("POST", "/api/jobs", json={"map_id": map_info["id"]})
job_id = job["id"]
deadline = time.monotonic() + 3600

while True:
    job = api("GET", f"/api/jobs/{job_id}")
    state = job.get("workflow") or job
    print(state["status"], job["processed_frames"], "/", job["total_frames"])
    if state["status"] in {"complete", "failed", "interrupted"}:
        break
    if time.monotonic() >= deadline:
        # 本例约定等待超时后也中断服务器任务。
        api("POST", f"/api/jobs/{job_id}/stop")
        raise TimeoutError(f"Job {job_id} timed out; stop requested")
    time.sleep(1)

if state["status"] != "complete":
    log = api("GET", f"/api/jobs/{job_id}/log")["text"]
    raise RuntimeError(f"{state['status']}: {state['error']}\n{log}")

page = api("GET", f"/api/jobs/{job_id}/frames", params={"limit": 1})
if page["frames"]:
    frame = api("GET", f"/api/jobs/{job_id}/frames/0")
    print(json.dumps(frame, ensure_ascii=False, indent=2))
print("Server results path:", job["results_path"])
if job.get("workflow"):
    print(api("GET", f"/api/jobs/{job_id}/rooms"))
    print(api("GET", f"/api/jobs/{job_id}/room-bindings", params={"limit": 1000}))
```

等待时限是客户端示例策略，实际地图规模和首次模型编译时间不同，应按部署环境调整。发生网络超时不代表服务任务停止，调用方应保留 `job_id` 并重新查询。

### 10.2 浏览器 SSE 与纳秒时间

在服务同源网页内运行：

```javascript
const jobId = "替换为任务 ID";
const events = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);

events.addEventListener("job", (event) => {
  const job = JSON.parse(event.data);
  const state = job.workflow || job;
  console.log(state.status, job.processed_frames, job.total_frames);
  // 根据 processed_frames 按需获取新增帧。
  if (["complete", "failed", "interrupted"].includes(state.status)) {
    events.close();
  }
});

events.onerror = () => {
  // EventSource 通常自动重连；重连会先收到当前摘要。
  console.log("Event stream connection changed");
};

async function readFrame(index) {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/frames/${index}`);
  if (!response.ok) throw new Error(await response.text());
  const frame = await response.json();
  const timeNs = BigInt(frame.time_ns);
  // 保留字符串用于 JSON 传输；不要用 Number 存储纳秒时间。
  return { frame, timeNs };
}
```

### 10.3 浏览器读取点云

下面使用 `DataView` 显式按 little-endian 解码，`positions` 和 `colors` 可交给三维渲染库：

```javascript
async function readCloud(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength % 24 !== 0) throw new Error("Invalid cloud byte length");
  const count = buffer.byteLength / 24;
  const view = new DataView(buffer);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    for (let axis = 0; axis < 3; axis++) {
      positions[i * 3 + axis] = view.getFloat32(i * 24 + axis * 4, true);
      colors[i * 3 + axis] = view.getFloat32(i * 24 + 12 + axis * 4, true);
    }
  }
  return { count, positions, colors };
}
```

若手动管理条件请求并收到 `304`，应使用本地缓存，跳过上述响应体解码。

## 11. 文件输出与历史恢复

### 11.1 输出目录

默认服务根目录为仓库 `output/server/`。原生部署可通过 `--output_dir` 修改。Docker 下该目录对应容器的 `/app/output/server`，宿主机位置由 Compose 的 `OUTPUT_DIR` 决定。

```text
output/server/
├── worker.log
├── maps/
│   └── <服务地图ID>.json
├── jobs/
│   └── <Boxer任务ID>/
│       ├── job.json
│       ├── inference.log
│       ├── workflow.json
│       ├── binding-input.json
│       ├── binding.log
│       ├── room-bindings.json
│       ├── 3dbbox.zip
│       └── output/<地图清单map_id>/<snapshot_id>/<相机模式>/
│           ├── results.json
│           ├── results.<运行标识>.metadata.json
│           ├── results.<运行标识>.frames.jsonl
│           ├── owl_2dbbs.csv
│           ├── boxer_3dbbs.csv
│           ├── boxer_3dbbs_tracks.csv
│           ├── boxer_3dbbs_tracked.csv
│           └── performance.json
└── voxel_connectivity/jobs/<内部子任务ID>/
    ├── job.json
    ├── inference.log
    └── output/room_polygons.json
```

Boxer 任务的实际结果路径以 `JobInfo.results_path` 为准。`map_id` 在路径树中指源地图清单的业务 ID。体素子任务目录由服务内部管理；调用方使用父任务 ID 查询、停止、读取日志及下载房间结果。

| Boxer 文件 | 内容与生成条件 |
| --- | --- |
| `worker.log` | 常驻物体检测进程的模型加载、配置和预热日志 |
| `maps/<地图ID>.json` | 已注册地图的服务端记录 |
| `job.json`、`inference.log` | 单个任务的状态记录、参数和运行日志。HTTP 客户端通过任务详情和日志接口读取 |
| `results.json` | 任务元数据、逐帧观测、融合检测、跟踪与最终物体。运行期间使用 schema 3，成功收尾后使用独立的 schema 2 文件 |
| `results.<运行标识>.metadata.json`、`results.<运行标识>.frames.jsonl` | schema 3 描述文件引用的静态元数据和逐帧提交日志。复制未完成的结果时一并保留 |
| 四份检测和跟踪 CSV | `no_csv=false` 时生成。字段见[第 8.7 节](#87-检测框-json-查询)，发布到源地图的条件见[第 11.3 节](#113-写入源地图-3dbbox) |
| `performance.json` | 启用 `profile` 后生成的性能统计 |
| `3dbbox.zip` | 成功收尾时保存的源地图 `3dbbox/` 快照，条件和下载方式见[第 7.8 节](#78-get-apijobsjob_id3dbboxzip) |

JPEG 叠加图和可视化点云由服务按请求生成，客户端按[第 8 节](#8-帧数据检测框图像与点云接口)读取。

成功推理还会在地图包内原子保存 `3dbbox/labels.csv`，表头为 `sem_id,label`，UTF-8 编码。
完整词表取自实际解析/加载的 `parameters.labels`，按原始顺序从 0 编号，包含未检测出的类别，
支持内置词表与自定义标签。`no_csv=true` 时仍保存词表；重跑覆盖为本次成功推理的完整词表。
`results.json` 的 `vocabulary_file` 为地图内相对路径 `3dbbox/labels.csv`，写入失败使任务失败。

DINOv3 特征默认合并到地图包的 `3dbbox/dino_features.csv`，每张有效原图一行。
文件使用 UTF-8，保留原有四列 `相机帧序号,相机序号,cls_token,patch_mean_pooling`，
末尾依次追加 `base_link_x,base_link_y,base_link_z,base_link_qx,base_link_qy,base_link_qz,base_link_qw`。
帧号使用原图 `image_frame_id`（旧地图为处理帧 `frame_id`），相机号使用 `camera_id`。
两个向量各占一个单元格，保存为含 384 个数值的一维 JSON 数组，可恢复 float32 `[1, 384]`。
七个位姿列使用 float64，保存该图像对应的 base 在全局 `map` 系下的位置（米）及归一化
四元数（`qx,qy,qz,qw`）。新地图取图像组位姿，旧地图取对应关键帧位姿，均来自
`T_map_base`，不叠加相机外参或做时间插值。同一图像组各相机的 base 位姿相同。
下次推理初始化时自动为旧四列文件的全部历史行补齐位姿，保留特征，不受本次选择范围限制；
空文件仅升级表头。旧行匹配失败、位姿缺失、数据非法或升级写入失败时保留原 CSV 并报错。
无检测框、`no_csv=true` 或关闭跟踪均不影响特征保存。两种相机模式统一使用 native
预处理，每张原图只生成一对向量；保存位置由 `results.json` 的源图记录中
`dino_feature_file` 字段引用该地图内相对路径，按帧号和相机号定位行。重复推理更新同键，
保留其他历史行；`statistics.dino_feature_rows` 统计本次写入行数。
新流程不写 PT，初始化时直接删除地图特征目录中的旧 PT，不转换其内容。
数值定义和 Python 读取示例见 [x-nav 输出说明](XNAV_MAP.md#输出与回放)。

检测框可通过第 8.7 节的四个 JSON 接口分页读取；成功新任务的词表、DINOv3 特征 CSV 和原始检测 CSV 可通过第 7.8 节的 ZIP 接口下载。整份 `results.json`、模型或源地图文件通过服务端共享目录读取。

### 11.2 提交语义与结果格式

运行中 `results.json` 使用 schema 3 描述文件，引用静态元数据和本次独立的 JSONL 帧日志，并记录已提交帧数及字节边界。失败或中断时保留可回放前缀和侧文件，迁移这种结果时需要保留关联文件。

成功完成后生成可独立读取的完整 schema 2 `results.json`；项目读取器兼容 schema 1、2、3。API 回放会统一读取这些格式，并把纳秒时间戳转换为字符串。

特征 CSV 在地图级文件锁保护下逐处理帧合并，在对应帧提交前完成同步和原子替换，
写入失败不会发布该帧为已提交。迁移后需要使用图像向量时，应保留地图包的
`3dbbox/dino_features.csv`。

运行中不要把磁盘上的 schema 3 描述文件当作完整帧数组；通过 HTTP 帧接口读取，或在服务器 Python 环境使用 `utils.xnav_pipeline.read_results()`。完整磁盘 JSON 的纳秒字段可能仍是整数，外部 JavaScript 读取时也需要处理精度问题。

### 11.3 写入源地图 `3dbbox/`

普通 server 任务成功、所有输出关闭且 `no_csv=false` 时，会将以下五个文件复制到输入地图的 `3dbbox/`，覆盖同名结果：

| 文件 | 内容 |
| --- | --- |
| `boxer_3dbbs_tracked.csv` | 最终确认的全局物体 |
| `boxer_3dbbs_tracks.csv` | 每个处理帧的已确认跟踪框 |
| `boxer_3dbbs.csv` | 各处理帧融合后的 3D 检测 |
| `owl_2dbbs.csv` | 各相机/视图的 2D 检测 |
| `boxer_3dbbs_observations.csv` | 逐条 2D 检测、融合框与轨迹的关联明细 |

3D CSV 的平移和边长为米，四元数按 `wxyz` 保存。新结果的 3D `time_ns` 使用关联来源图像时间，最终物体取全部关联的最早时间；`pose_time_ns` 保存原位姿时间。2D CSV 使用原始图像时间，框坐标对应预处理视图的像素。旧结果的时间语义保持不变。表头与检测框查询字段对应，CSV 的数值和类别文本可能经过格式化。

服务先暂存五份文件，再逐个原子替换，任务目录保留原件。这是逐文件发布，不是五个文件整体的事务；同步读取多个文件时，应在任务 `complete` 后读取。

新联合流程中，地图 CSV 导出后 Boxer 阶段即可标记 `complete`，待分割与绑定完成再保存房间目录和生成 `3dbbox.zip`，详见第 15.4 节。缺少所需 CSV 或 CSV 导出失败仍使 Boxer 阶段失败；房间保存和 ZIP 失败由 `workflow.map_export` 报告并使整体失败。此时仍可从各任务目录或结果接口读取成功阶段的产物。历史任务的归档行为保持原样。

`no_csv=true`、推理失败、中断或导入历史结果时不执行 CSV 地图导出；已逐帧保存的 DINOv3 特征会保留。无检测或关闭跟踪时，相关 CSV 仍可只有表头。新增的 `3dbbox/` 不计入地图来源签名。

同一地图反复处理时，`no_csv=true` 会保留以前的检测 CSV，而词表、DINOv3 特征和本次房间结果仍可能更新；不生成 ZIP。需要交付某次成功完整流程的地图产物时，使用该任务的 `3dbbox.zip`。

### 11.4 重启与恢复

持久化相同服务输出目录后，重启会恢复已注册地图与历史任务。之前处于 `starting`、`running`、`stopping` 的普通任务会被标记为 `interrupted`，错误说明服务重启后未恢复推理。

已成功归档任务的 `3dbbox.zip` 直接从持久化文件下载；重启不会重新生成。中断收尾的未发布 ZIP 和临时 ZIP 会被清理，历史任务与导入任务不会补归档。

恢复记录不等于断点续推；创建新任务会重新建立跟踪器。地图和结果的服务端路径仍需可用，来源签名仍需匹配。服务进程重启后当前推理配置回到默认值，应重新查询和应用所需配置。

## 12. 错误处理与排查

### 12.1 响应格式和状态码

业务 HTTP 异常和已处理的输入错误通常返回：

```json
{"detail": "Map not found"}
```

请求校验错误返回 `422`，`detail` 为列表，包含错误位置和原因：

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "start_n"],
      "msg": "Input should be greater than or equal to 1",
      "input": 0,
      "ctx": {"ge": 1}
    }
  ]
}
```

校验错误附加字段随依赖版本可能变化，客户端以状态码、`loc` 和 `msg` 处理即可。

| 状态码 | 常见原因 | 处理方式 |
| --- | --- | --- |
| `400` | 地图损坏、路径不可读、相机选择无效、采样范围为空、来源签名或结果格式不匹配 | 检查路径、输入文件和业务参数；地图变化后重新注册 |
| `404` | 未知地图/任务、发现根目录不存在、结果尚未发布、帧未提交或无匹配图像观测 | 区分 ID 错误与等待结果；检查任务进度和观测列表 |
| `409` | 已有活动推理/模型加载、任务 options 与当前配置不同、停止只读结果 | 等待空闲、先应用配置，或使用源推理任务 ID |
| `413` | ZIP、分块、解压容量或条目数超限 | 核对上传限制及 ZIP 内容 |
| `415` | 分块请求没有使用 `application/octet-stream` | 修正 Content-Type |
| `422` | 请求字段错误，或分块哈希、ZIP/地图校验失败 | 按 schema 修正请求或检查地图内容 |
| `500` | 未处理的服务内部异常 | 查看服务日志和任务日志 |
| `503` | 模型尚未就绪或服务正在关闭；根页面也可能因缺少构建资源返回此码 | 查询 `/api/inference`，等待/重试配置，或构建前端 |
| `507` | 上传、解压或持久化时磁盘空间不足 | 释放空间后恢复上传或重试 complete |

OpenAPI 为各业务路由统一声明了上述错误模型，不表示每个路由都会产生所有错误码。未处理的 `500` 当前没有统一 JSON 异常包装，客户端应能回退读取纯文本，不能假设所有失败响应都可用 `.json()` 解码。

模型准备失败通常体现为 `PUT` 已返回 `202`，随后 GET 返回 `status: "failed"` 和 `error`；推理失败也主要通过 `JobInfo.status/error` 报告，而不是让最初的 POST 长时间保持连接。

### 12.2 常见现象

| 现象 | 核对方式 |
| --- | --- |
| `/api/health` 返回 `ok`，创建任务却报 `503` | 检查 `inference_status`，等待 `/api/inference.status == "ready"` |
| PUT 修改一个字段后其他配置恢复 | PUT 采用默认值补全；先 GET 完整 `options`，修改后整体提交 |
| 创建任务报参数不一致 | 先 PUT 应用；POST 省略 `options`，或提供完全一致的解析后配置 |
| `/api/maps/directories` 为 `404` | 准备服务可见的 `/nav_map`，或直接 POST 地图完整路径 |
| 发现列表有地图，但注册失败 | 列表只检查标识文件；检查 V6 manifest、标定、图像、索引和点云 |
| 配置显示就绪，但任务因相机参数失败 | 核对注册地图的 `cameras`；模型准备不做针对某地图的相机验证 |
| `/frames` 在刚创建任务后报 `404` | 等待首个结果文件发布；区分首帧前等待和任务失败 |
| 3D 图像没有框，JSON 有检测 | 无跟踪时用 `tracked=false`；仅候选轨迹时用 `tentative=true` |
| 图像请求找不到相机或视图 | 先查询该帧 `observations`，使用其中实际的 ID 组合 |
| 纳秒时间戳末尾数字变化 | 保留 API 字符串；JavaScript 运算使用 `BigInt` |
| 点云颜色或坐标异常 | 按 little-endian float32 六列解码，每点 24 字节，颜色范围 `0..1` |
| 所有帧处理完但任务失败 | 检查输出关闭、CSV 完整性和源地图 `3dbbox/` 写入权限 |
| 导入结果报地图不匹配 | 保持源地图和关联侧文件一致，核对来源签名与预处理版本 |
| 任务完成后 SSE 仍保持连接 | 收到结束状态后调用 `EventSource.close()` 或关闭流式请求 |
| 重启后任务未继续或参数变回默认 | 这是当前恢复行为；重新应用配置并创建新任务 |

## 13. 房间分割配置与运行

`POST /api/jobs` 是唯一计算任务入口。服务并行启动 Boxer 和内置 `voxel-connectivity-v10`，两者成功后按房间几何计算最终物品归属，再保存地图结果和 ZIP。请求体沿用第 7.1 节，房间分割配置由服务读取，不新增独立提交步骤。

分割输入为当前注册地图的完整 `global_map.pcd`，坐标单位为米、`+Z` 向上。每次任务重新读取 `config/voxel_connectivity.yaml`，可由 `BOXER_VC_CONFIG` 指定其他配置文件：

| 配置字段 | 单位与范围 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `resolution` | 米，`0.03–0.50` | `0.10` | 二维分割栅格分辨率 |
| `min_room_points` | 整数，`0–100000000` | `0` | 过滤支持点少于阈值的候选房间；`0` 关闭过滤 |

分割内部仍使用地图点云，发布结果统一为 `room_polygons.json`。归属、房间展示和房间导出直接使用轮廓与高度范围，不读取房间点云文件。

CPU 解释器选择顺序为 `BOXER_VC_PYTHON`、`BOXER_PF_PYTHON`、服务自身 Python；覆盖解释器也须安装 `requirements-potential-field.txt`。环境检查、配置检查在任务受理前进行。日志通过 `/api/jobs/{job_id}/log?stage=room_segmentation` 读取。

需要脱离 HTTP 服务处理文件时，仍可运行 CLI：

```bash
.venv/bin/python -m voxel_connectivity \
  --input /path/to/global_map.pcd --output-dir /path/to/rooms
# 势场 CLI 继续保留；HTTP 联合任务固定使用体素连通算法。
.venv/bin/python -m potential_field --help
```

## 14. 房间几何格式

`GET /api/jobs/{job_id}/room-polygons.json` 下载关联分割任务的原始文件，结构如下。原始算法文件的 `schema_version` 保持为 `1`；父任务结果和绑定采用版本 `2`。

```json
{
  "schema_version": 1,
  "coordinate_frame": "scene_xy",
  "units": "m",
  "resolution_m": 0.1,
  "volume_type": "vertical_prism",
  "z_bounds_source": "labeled_point_extents",
  "rooms": [{
    "id": 0,
    "area_m2": 16.0,
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[0,0],[4,0],[4,4],[0,4],[0,0]]]
    },
    "z_bounds_m": [0.0, 2.8]
  }]
}
```

- `scene_xy` 是当前注册地图 `map` 坐标系的 XY 平面，结合 `z_bounds_m` 构成三维竖直区域，不是地理经纬度。
- `geometry` 支持 GeoJSON 形状的 `Polygon` 和 `MultiPolygon`，坐标仅有 XY。每个 Polygon 首环为外环，后续环为孔洞，所有环闭合；孔洞不属于房间。
- `z_bounds_m` 是该房间已标注点的最小和最大 Z。上下界相等的区域没有体积，不会产生正体积归属。
- 房间 ID 在关联分割任务内唯一且非负，过滤后可以不连续；只可与同一父任务的物品结果组合。
- 房间可以为空数组。损坏的几何、重复 ID、非有限坐标、未闭合环或倒置高度范围会使绑定阶段失败，不自动修复拓扑。

## 15. 联合工作流与物品房间归属

### 15.1 单一任务与生命周期

所有计算通过 `POST /api/jobs` 提交，以返回的父任务 `id` 查询、停止和下载。`JobInfo.status` 表示 Boxer 阶段，`workflow.status` 表示完整流程；`workflow.room_segmentation`、`workflow.binding` 和 `workflow.map_export` 分别表示分割、归属和地图保存阶段。阶段 `job_id` 仅用于来源审计，调用方无需访问内部子任务。

新任务的 `workflow.schema_version=2`。整体沿用 `starting/running/stopping/complete/failed/interrupted`，阶段另有 `pending/skipped`。两项并行计算成功后才启动归属；一项失败时另一项继续，成功的分割结果仍可读取和保存。分割、归属、地图保存与要求的 ZIP 均发布后，整体才为 `complete`。

`POST /api/jobs/{job_id}/stop` 停止仍活动的阶段并取消未发布导出；成功结果保留。服务重启将未完成流程标为 `interrupted`，恢复未完成目录替换并清理临时文件，不自动续跑。地图源签名与 `global_map.pcd` 文件指纹在绑定和发布前复核。

原先两组独立分割任务路由以及 `/api/jobs/{job_id}/rooms/cloud` 已撤下，返回 `404`。已知历史任务缺少版本 `2` 时，新的房间、几何下载和归属接口返回 `409`，提示重新运行；历史任务文件及已有 ZIP 不改写。地图注册、上传和只读结果导入不启动计算。

### 15.2 房间结果与下载

`GET /api/jobs/{job_id}/rooms` 在分割成功后即可读取，不要求 Boxer 或归属成功。响应包括：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | `2` |
| `job_id/map_id/segmentation_job_id` | 父任务、注册地图及内部来源子任务 |
| `algorithm` | `voxel_connectivity` |
| `coordinate_frame/units/volume_type/resolution_m` | 第 14 节的几何坐标和分辨率约定 |
| `rooms` | 房间数组，每项含 `id/name/geometry/z_bounds_m/area_m2`；名称缺省时按 ID 生成为“房间 01”等 |
| `summary` | `room_count`、总平面面积 `area_m2` 和总区域体积 `volume_m3` |
| `artifacts` | `room_polygons.json` 对应的父任务下载 URL |

文件下载使用 `GET /api/jobs/{job_id}/room-polygons.json`，媒体类型为 `application/json`。不再发布 `result_colored.ply`、`rooms.json`、`summary.json` 或房间预览点云。

### 15.3 精确体积归属与分页

输入为最终 `results.json.objects`，与最终 tracked 物品一致，排除 `TENTATIVE`；保留现有物品 ID、位姿、尺寸及检测来源。按完整四元数使用朝向框，不将倾斜框近似为轴对齐框。

对物品框 B 和房间竖直区域 R，覆盖比例为 `Volume(B ∩ R) / Volume(B)`。分母固定为物品完整框体积，不按房间体积或已覆盖部分归一化。候选房间先经 XY 空间索引和高度筛选，再使用裁剪面片的有向面积积分精确计算三维体积，支持凹轮廓、孔洞和 MultiPolygon。

唯一最大覆盖比例达到 **60%** 才归属。并列、比例不足、仅边界接触或没有相交时保持未归属，不按中心点或距离补配。计算采用 float64、物品中心局部坐标和 `1e-9` 相对容差，容差只处理浮点误差，不扩大房间或物品框。

`GET /api/jobs/{job_id}/room-bindings` 支持 `offset=0`、`limit=200`（1–1000）、`room_id>=0` 筛选房间、`room_id=-1` 筛选未归属物品；省略返回全部。响应为版本 `2`，`binding_method=obb_room_volume_v1`，包含任务关联、`summary/total/offset/limit/items`。物品顺序与最终 tracked 结果一致，`total` 为筛选后的数量。

| 每项字段 | 含义 |
| --- | --- |
| 现有物品字段 | ID、类别、位姿、尺寸、概率及检测关联 |
| `room_id/room_name` | 确认归属；未归属均为 `null` |
| `object_volume_m3` | 完整 OBB 体积；无效框为 `null` |
| `intersection_volume_m3/overlap_ratio` | 最佳候选的相交体积与完整框覆盖比例；无候选为 `0`，未归属时仍保留最佳证据 |
| `overlaps` | 正相交候选 `{room_id,intersection_volume_m3,overlap_ratio}`，按体积降序、并列按房间 ID 排序 |
| `reason` | 成功为 `null`；否则为 `invalid_bbox/no_rooms/no_overlap/insufficient_overlap/ambiguous_overlap` |

不再返回 `point_count/labeled_point_count/support_points/vote_ratio/votes`。关闭跟踪时成功生成空归属结果，`summary.reason=tracking_disabled`；没有最终物品时为 `no_tracked_objects`。零房间时仍成功，物品均以 `no_rooms` 保留为未归属。

`GET /api/jobs/{job_id}/room-bindings.json` 下载完整 JSON，包含 `schema_version=2`、`binding_method`、来源签名、地图指纹、坐标约定、`parameters`、`summary` 和 `items`。参数记录 `min_overlap_ratio=0.6`、`relative_tolerance=1e-9`。归属尚未成功或版本不支持返回 `409`，未知父任务返回 `404`，分页参数无效返回 `422`。

```bash
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/rooms" | jq
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/room-polygons.json" -o room_polygons.json
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/room-bindings?room_id=-1&limit=1000" | jq
curl --fail-with-body -sS "$API_BASE/api/jobs/$JOB_ID/room-bindings.json" -o room-bindings.json
```

### 15.4 地图保存与完整 ZIP

新任务收尾时整体替换输入地图的专用房间目录：

```text
3dbbox/room_segmentation/
├── room_polygons.json
├── room-bindings.json
└── source.json
```

房间几何逐字节复制关联分割产物；归属成功时保存完整绑定 JSON。`source.json` 的版本为 `2`，记录任务、来源地图、导出时间及实际文件列表。分割成功但 Boxer 或归属失败时，仅保存几何与来源文件，清除目录中的旧归属，整体失败且不生成 ZIP。分割失败时不替换地图目录；显式停止不启动新的导出。

继续使用完整暂存、目录原子替换和替换日志，失败还原旧目录。成功任务的 ZIP 包含检测 CSV、累计地图特征以及上述房间目录；后续重跑不修改历史 ZIP。`no_csv=true` 仍保存几何和归属，跳过检测 CSV 与 ZIP。Web 提供房间几何、物品归属 JSON 和整体成功后的完整 ZIP 下载。

## 16. 实现位置与维护

文档主要依据以下项目文件；扩展接口或调整参数后，应同步更新本文并核对部署服务的 schema：

| 文件 | 职责 |
| --- | --- |
| [server.py](../server.py) | HTTP 路由、状态码、SSE、JPEG 与二进制响应、Swagger UI |
| [utils/xnav_service.py](../utils/xnav_service.py) | 请求/响应模型、配置与任务状态、目录发现、持久化和恢复 |
| [utils/xnav_workflow.py](../utils/xnav_workflow.py) | 联合调度、父子关联、生命周期与房间结果读取 |
| [utils/xnav_room_binding.py](../utils/xnav_room_binding.py) | 独立 CPU 归属进程、精确 OBB 与房间体积覆盖 |
| [utils/xnav_room_geometry.py](../utils/xnav_room_geometry.py) | 房间几何校验、完整朝向框裁剪与体积积分 |
| [utils/xnav_room_export.py](../utils/xnav_room_export.py) | 房间地图目录发布、恢复及工作流 ZIP 归档 |
| [utils/xnav_upload.py](../utils/xnav_upload.py) | 上传会话、分块提交、ZIP 安全解压、后台校验和发布恢复 |
| [utils/xnav_runtime.py](../utils/xnav_runtime.py) | 模型加载复用、设备和精度选择 |
| [utils/xnav_web_scene.py](../utils/xnav_web_scene.py) | 地图场景、点云编码、图像渲染与缓存 |
| [loaders/xnav_loader.py](../loaders/xnav_loader.py) | 地图校验、帧身份、采样、相机和点云读取 |
| [utils/xnav_pipeline.py](../utils/xnav_pipeline.py) | 检测、融合、跟踪、结果读取和记录格式 |
| [utils/xnav_result_store.py](../utils/xnav_result_store.py) | 结果提交日志及完整结果文件 |
| [utils/xnav_export.py](../utils/xnav_export.py) | 完成后向源地图发布 CSV |
| [utils/potential_field_service.py](../utils/potential_field_service.py) | CPU 分割任务、状态恢复、日志与产物发布 |
| [potential_field.py](../potential_field.py) | 势场工作进程、命令行和产物写入 |
| [voxel_connectivity.py](../voxel_connectivity.py) | 体素连通复核工作进程、配置与命令行 |
| [utils/taxonomy.py](../utils/taxonomy.py) | 词表文件和文本提示解析 |
| [owl/owl_wrapper.py](../owl/owl_wrapper.py) | 当前 OWLv2 输入预处理、精度和框输出约定 |

服务启动后可保存当前 schema，供接口比对或客户端工具导入：

```bash
curl --fail-with-body -sS http://127.0.0.1:18765/openapi.json \
  -o boxer-openapi.json
```

CPU 分割任务与产物的回归测试覆盖势场法和体素连通复核。 [势场法测试](../tests/test_potential_field_service.py) · [体素连通复核测试](../tests/test_voxel_connectivity_service.py)
