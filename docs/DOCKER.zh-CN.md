# Docker 部署 x-nav Web 服务

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
