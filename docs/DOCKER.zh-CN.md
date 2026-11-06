# Docker 部署 x-nav Web 服务

首次部署可先阅读[完整部署指南](DEPLOYMENT.zh-CN.md)，包含宿主机安装、权重下载与
SHA256 校验、词表部署、首次推理验收、离线迁移及常见问题处理。

在 Boxer 仓库根目录执行以下命令。x86 NVIDIA 与 Jetson Orin 使用各自独立的
Dockerfile 和 Compose 文件；两者默认启动 `server.py`，提供 Web 页面、API 和常驻
GPU 推理进程。镜像只包含源码、运行依赖和编译后的前端，模型、地图及词表由宿主机提供。

## 平台要求

| 平台 | 构建文件 | Compose 文件 | 运行环境 |
| --- | --- | --- | --- |
| Linux x86 NVIDIA | `Dockerfile.x86` | `docker-compose.x86.yaml` | Python 3.12、PyTorch 2.10.0+cu128、CUDA 12.8 |
| Jetson Orin | `Dockerfile.jetson` | `docker-compose.jetson.yaml` | Python 3.10、Jetson PyTorch 2.10.0、CUDA 12.6 |

宿主机安装 Docker Engine、Compose v2 或更新版本，以及已配置 Docker 集成的
NVIDIA Container Toolkit。x86 的显卡和驱动须支持 PyTorch 2.10.0 / CUDA 12.8；
容器不安装宿主机驱动。

Jetson 基线为 **JetPack 6.2.1 / L4T 36.4.4 / CUDA 12.6**，面向 Orin 系列。
使用 NVIDIA 发布的 `nvcr.io/nvidia/l4t-jetpack:r36.4.0` 基础镜像和项目现有的
Jetson wheel、独立 cuDSS 归档；下载均检查已有 SHA256。不会安装会替换 JetPack
cuBLAS 的 PyPI cuDSS wheel。默认设置 `TORCHDYNAMO_DISABLE=1`。
其他 JetPack 版本需要单独适配；在目标平台原生构建和运行对应配置。

x86 通过 GPU device reservation 请求 NVIDIA 显卡；Jetson 显式使用
`runtime: nvidia`，不依赖仅配置 `gpus: all` 的设备注入方式。不需要桌面、X11
或 privileged 容器。

## 准备挂载目录

```bash
cp .env.docker.example .env
```

按本机路径编辑 `.env`。相对路径以 Compose 文件目录为基准，绝对路径指向
**Docker daemon 所在宿主机**的文件。已有 `ckpts` 符号链接可以继续使用。

| 变量 | 默认宿主机路径 | 容器路径 | 权限/用途 |
| --- | --- | --- | --- |
| `NAV_MAP_DIR` | `./nav_map` | `/nav_map` | 读写；读取地图并导出 `3dbbox/` |
| `CKPTS_DIR` | `./ckpts` | `/app/ckpts` | 读写；读取模型及保存 OWLv2 文本嵌入缓存 |
| `VOCAB_DIR` | `./owl` | `/vocab` | 只读；不覆盖容器内的 `owl` Python 代码 |
| `OUTPUT_DIR` | `./output/server` | `/app/output/server` | 读写；任务、日志、结果和历史 |
| `SERVER_PORT` | `18765` | `18765` | 宿主机发布端口 |

例如复用相邻 x-nav 仓库的地图：

```dotenv
NAV_MAP_DIR=/home/glory/Code/digitaltwins-x-nav/nav_map
CKPTS_DIR=./ckpts
VOCAB_DIR=./owl
OUTPUT_DIR=./output/server
SERVER_PORT=18765
```

地图、模型和词表目录必须提前存在，Compose 不会自动创建空目录替代这些输入。
模型目录必须包含以下三个非空、可读文件：

```text
boxernet_hw960in2x6d768-c88128f8.ckpt
dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
owlv2-base-patch16-ensemble.pt
```

镜像构建和服务启动均不下载模型。模型目录要可写，因为模型初始化时会生成缺少的
`owlv2-…_textemb_….pt` 文本嵌入缓存。输出目录可自动创建；容器默认以 root 运行，
自动创建的输出、缓存和地图 `3dbbox/` 文件可能归 root 所有。

词表目录至少包含 `indoor_classes.csv`，因为服务启动预加载默认 `indoor`。
可从项目 `owl/` 目录复制已有 CSV 到独立词表目录，再设置 `VOCAB_DIR`。
新增 `factory_classes.csv` 后，在 Web 页面的词表参数填写 `factory` 并应用即可。
文件保持每行一个标签；修改正在使用的词表后须重新应用参数。

容器使用 `BOXER_LABELS_DIR=/vocab`。非 Docker 环境未设置该变量时继续读取
项目内的 `owl/`；也可以为命令行推理和本地 server 设置此变量。

## 构建和启动

### 地图上传配置

Web 和 OpenAPI 支持 ZIP 地图包分块上传、暂停和重启续传。地图根目录默认 `/nav_map`，
暂存位于 `.boxer_uploads/`，成功后保存到 `uploads/<包名>-<upload_id>`，并自动注册。
需同时持久化地图挂载和服务输出目录；现有 Compose 已提供这两个挂载。

可在 `.env` 中调整以下变量，x86 和 Jetson 配置均会传入容器；原生部署设置同名环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `BOXER_UPLOAD_CHUNK_SIZE` | `8388608` | 8 MiB 分块；最大允许 64 MiB |
| `BOXER_UPLOAD_MAX_ZIP_BYTES` | `21474836480` | ZIP 最大 20 GiB |
| `BOXER_UPLOAD_MAX_EXPANDED_BYTES` | `107374182400` | 解压最大 100 GiB |
| `BOXER_UPLOAD_MAX_FILES` | `200000` | ZIP 最大条目数，包含目录 |
| `BOXER_UPLOAD_TTL_SECONDS` | `86400` | 未完成会话闲置 24 小时清理；状态查询不续期 |

合并时保留原分块，磁盘峰值约为 `2 × ZIP 大小 + 解压大小`；成功后立即回收暂存，
完成记录保留上述期限，已发布地图继续保留。处理失败可在期限内重试或取消。
已有会话沿用创建时的分块大小，其他限制采用当前部署配置。
原生服务可用 `--map_root /path/to/maps` 更换地图根目录，该目录需提前创建且可写。

### 启动命令

选择目标平台的一条命令：

```bash
# x86 NVIDIA
docker compose -f docker-compose.x86.yaml up -d --build

# Jetson Orin
docker compose -f docker-compose.jetson.yaml up -d --build
```

对应镜像为 `boxer-server:x86` 和 `boxer-server:jetson`。两套 Compose 使用相同的
`boxer` 项目名，在各自主机上选择一套运行。首次构建需要联网下载基础镜像、Python
依赖和前端依赖；Node.js 仅用于前端构建，运行时不需要 Node.js。

### GitLab CI 自动构建与发布 x64 镜像

仓库根目录的 `.gitlab-ci.yml` 复用 `Dockerfile.x86`，通过
`scripts/ci_build_x86.sh` 构建 `linux/amd64` 镜像，并推送到**当前 GitLab 项目的
Container Registry**。自建 GitLab 的域名和项目路径由 CI 自动提供，无需写进仓库。
当前本地 Git remote 若仍指向 GitHub，需要将代码和 CI 配置推送到自建 GitLab 项目
才能触发该流水线。

启用前在 GitLab 和 Runner 主机上完成以下配置：

1. 在自建 GitLab 启用 Container Registry，并确保项目可使用该功能。Registry 使用
   HTTPS，证书应已被 Runner 主机及 Docker 信任。
2. 为项目启用现有的 **Linux x64 Shell Runner**，勾选 **Run untagged jobs**。
   此任务没有 Runner `tags`，也不使用 Docker executor 或 DinD；Runner 的保护规则
   需允许目标 `master` 分支及要发布的 Git 标签。
3. Runner 主机安装 Docker Engine 和系统级 Buildx 插件，使用默认 Docker context。
   以 Runner 实际运行用户执行 `docker info` 和 `docker buildx version`，确认 Docker
   权限及插件可用。构建和冒烟检查不需要 GPU、模型或地图挂载。
4. 任务超时为 **2 小时**；Runner 的最大超时应至少为两小时。主机需要足够空间保存
   PyTorch/CUDA 依赖、镜像和 BuildKit 缓存，并能访问基础镜像、APT、PyPI、PyTorch
   wheel 源和 npm。连续构建复用默认 builder 在 Docker daemon 中的缓存。

流水线使用 GitLab 内置的 `CI_REGISTRY`、`CI_REGISTRY_IMAGE`、`CI_REGISTRY_USER`
和 `CI_REGISTRY_PASSWORD`；无需另建长期推送密码。登录通过 `--password-stdin`，
凭据保存在检出目录外的任务临时目录，退出时清理，不改写 Runner 用户原有的登录。

| 触发事件 | 发布到 `$CI_REGISTRY_IMAGE` 的镜像标签 |
| --- | --- |
| `master` 分支 push | `sha-<完整提交 SHA>`、`latest` |
| 任意 Git 标签 push | `sha-<完整提交 SHA>`、对应版本标签、`latest` |
| 其他分支、MR、定时或手动创建的流水线 | 不触发发布 |

合法的 Docker 标签保留 Git 标签原名，例如 `v1.2.3`；含 `/` 等不合法字符或超过
128 字符时改用 `tag-${CI_COMMIT_REF_SLUG}-${CI_COMMIT_SHORT_SHA}`。
例如 `release/v1.2.3` 会发布为 `tag-release-v1-2-3-<短 SHA>`。
Git 标签为 `latest` 或与本次提交标签相同时，只推送一次该镜像标签。

每次构建先使用任务独立的本地镜像标签，确认架构、Python 依赖、服务模块和前端
静态文件后，再推送提交及版本标签，最后更新 `latest`。构建、冒烟检查或前序推送
失败时不会执行 `latest` 推送；已推送成功的提交／版本标签会保留。
发布任务通过同一个 `resource_group` 串行执行；`latest` 始终表示最后一次成功发布
的镜像，可以来自 `master`，也可以来自某个 Git 标签，不按版本号或提交时间排序。
重试旧流水线也会在成功后更新 `latest`。
退出时清理任务临时凭据、冒烟容器和临时镜像标签；发布标签及构建缓存保留在 Runner。

首次发布前，在自建 GitLab 的 **CI Lint / Pipeline editor 校验**中检查配置。
分别推送一次 `master` 提交和版本标签，检查任务日志及项目的 Container Registry；
本次发布的提交／版本标签和 `latest` 应有相同的 manifest digest，平台为 `linux/amd64`。
流水线只发布镜像，GPU 运算和真实推理仍按下文在部署主机验收。

### 拉取已发布镜像并使用现有 Compose

在 x64 GPU 部署主机准备上述挂载目录及 `.env` 后执行；替换示例中的 Registry
域名和项目路径。`docker login` 使用交互输入，建议使用具有 `read_registry`
权限的 Deploy Token 用户名和 token，避免把密码写进命令行：

```bash
docker login registry.gitlab.example.com
BOXER_IMAGE=registry.gitlab.example.com/group/boxer:latest
# 需要固定版本时，将 latest 换成 v1.2.3 或 sha-<完整提交 SHA>。
docker pull --platform linux/amd64 "$BOXER_IMAGE"
docker image tag "$BOXER_IMAGE" boxer-server:x86
docker compose -f docker-compose.x86.yaml up -d --no-build --pull never
```

现有 Compose 使用本地 `boxer-server:x86` 标签，因此拉取后需完成上述重新打标签步骤。
每次更新重复 pull、tag 和 up 即可；模型、词表和地图仍通过原挂载提供。

参考：[GitLab Shell executor 构建 Docker 镜像](https://docs.gitlab.com/ci/docker/using_docker_build/#use-the-shell-executor)、
[Registry 认证](https://docs.gitlab.com/user/packages/container_registry/authenticate_with_container_registry/)、
[流水线资源组](https://docs.gitlab.com/ci/resource_groups/)。

### 运行检查与停止

访问 <http://127.0.0.1:18765/>，API 文档位于 <http://127.0.0.1:18765/docs>。
若修改了 `SERVER_PORT`，使用对应端口。Web 自动递归列出容器 `/nav_map` 下的地图包，
选择相对路径并点击“加载地图”，也可点击“刷新列表”重新查找。例如宿主机
`nav_map/园区A/img2` 会显示为 `园区A/img2`，对应容器路径 `/nav_map/园区A/img2`。
OpenAPI 的 `POST /api/maps` 仍接受完整容器路径，包括其他已挂载目录中的地图路径。

以下以 x86 为例；Jetson 将命令中的 Compose 文件替换为对应文件：

```bash
docker compose -f docker-compose.x86.yaml logs -f server
docker compose -f docker-compose.x86.yaml ps
curl http://127.0.0.1:18765/api/health
curl http://127.0.0.1:18765/api/inference
```

启动前会验证挂载、实际写权限和 CUDA 矩阵乘法、注意力、卷积。任何检查失败都不会
启动 HTTP 服务，具体原因记录在容器日志中。模型加载期间仍可访问 Web 页面；
`/api/inference` 的 `status` 为 `ready` 后才可推理。健康检查允许 `loading`，
在模型加载失败、CUDA 不可用或 HTTP 不通时失败。Docker 不会仅因 unhealthy
自动重启容器；`unless-stopped` 在进程退出时生效。

无需加载模型即可单独检查镜像中的 GPU 运算：

```bash
docker compose -f docker-compose.x86.yaml run --rm --no-deps server \
  python /app/scripts/check_cuda.py
```

编译和 CUDA 缓存保存到 Compose 的 `cache` 命名卷。停止服务保留缓存和宿主机结果：

```bash
docker compose -f docker-compose.x86.yaml down
```

容器通过 SIGINT 通知服务，提供 90 秒停止宽限期，并使用 init 回收子进程。
重启恢复历史任务记录，将之前未完成任务标记为中断，不自动续跑。

推理完成后四个 CSV 同步写入 `/nav_map/<地图>/3dbbox/`，主结果和历史保存在
`OUTPUT_DIR`。详细接口及导出语义见 [Web 服务说明](WEB_SERVER.zh-CN.md)。

参考：[NVIDIA L4T JetPack 镜像](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack)、
[NVIDIA Container Toolkit 配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)、
[Docker Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)。
