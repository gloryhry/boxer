# Boxer 项目部署指南

本文面向首次部署 Boxer 的维护人员，说明如何准备 Docker 和 GPU、下载权重、
部署词汇表和 x-nav 地图、启动 Web/API 服务，以及验收、迁移和排查问题。
内容按 **2026-11-04 仓库中的 Dockerfile、Compose 和服务实现**整理。

本文中的“项目根目录”指包含 `server.py`、`Dockerfile.x86` 和 `owl/` 的目录。
例如当前工作区中的 `/home/glory/Code/boxer/boxer`，而不是它的上一级目录。
新机器应获取包含本项目 x-nav/Web/Docker 扩展的源码版本，并记录 `git rev-parse HEAD`；
仅下载上游原始 Boxer 源码不一定包含这里使用的部署文件。

部署顺序：选择平台 → 安装 Docker/GPU 运行时 → 准备目录 → 下载并校验权重 →
部署词汇表和地图 → 配置 `.env` → 构建启动 → 小规模推理验收。
已有 Docker 环境时，可从第 3 节开始。

## 1. 选择部署平台

| 项目 | Linux x86 NVIDIA 服务器/工作站 | Jetson Orin |
| --- | --- | --- |
| 宿主机架构 | `x86_64` / `amd64` | `aarch64` / `arm64` |
| Dockerfile | `Dockerfile.x86` | `Dockerfile.jetson` |
| Compose | `docker-compose.x86.yaml` | `docker-compose.jetson.yaml` |
| 容器 Python | 3.12 | 3.10 |
| 容器 PyTorch | `2.10.0+cu128` | `2.10.0`，Jetson `jp6/cu126` wheel |
| CUDA 运行环境 | 12.8 | 12.6 |
| GPU 接入 | NVIDIA GPU device reservation | 显式 `runtime: nvidia` |
| JetPack 基线 | 不适用 | JetPack 6.2.1 / L4T 36.4.4 |

x86 宿主机需要可用且兼容容器 CUDA/GPU 的 NVIDIA 驱动；容器不会安装宿主机驱动。
PyTorch 所需的 CUDA 用户态库随镜像依赖安装，宿主机无需为此另装完整 CUDA Toolkit。
Jetson 的驱动与 JetPack 配套，应保留该平台的软件栈。

Jetson 镜像使用已发布的 `nvcr.io/nvidia/l4t-jetpack:r36.4.0` 基础镜像，
其标签和宿主机 L4T `36.4.4` 不同是当前配置的既定选择。
其他 JetPack/L4T 版本需要另行验证，不应只修改镜像标签便视为兼容。
在目标架构的机器上原生构建和运行对应镜像。

开始前检查：

```bash
uname -m
cat /etc/os-release
df -h .
```

x86 额外运行 `nvidia-smi`，确认宿主机能识别 GPU。Jetson 用
`cat /etc/nv_tegra_release` 检查 L4T，用 `tegrastats` 观察资源；
不能仅凭 Jetson 上没有 `nvidia-smi` 就判断 GPU 不可用。
磁盘需容纳镜像、构建缓存、约 1.20 GB 的三个权重，以及地图和结果。
显存/共享内存需求随相机数、词表和运行方式变化，首次先用少量帧验收。

## 2. 安装 Docker 和 NVIDIA 容器运行时

### 2.1 安装 Docker Engine 与 Compose

以下以使用 Docker 官方 APT 源的 Ubuntu 22.04/24.04 为例。
已有可用 Docker Engine、Buildx 和 Compose 插件时不必重新安装；
Jetson 已由 JetPack 配好 Docker 时，先检查现有环境。
其他发行版按 [Docker 官方安装说明](https://docs.docker.com/engine/install/)操作。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg wget git python3 jq
sudo install -d -m 0755 /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo docker compose version
```

如提示已安装的软件包冲突，先按 [Ubuntu 安装页](https://docs.docker.com/engine/install/ubuntu/)
核对现有的 `docker.io`、旧 Compose、`containerd` 等包及其用途。
后续命令使用 `docker compose`，需要 Compose v2 或更新的插件版本。

本文后续命令假设当前账号可以访问 Docker。需要免 `sudo` 操作时，管理员可执行
`sudo usermod -aG docker "$USER"`，然后退出并重新登录；否则在 Docker 命令前加 `sudo`。
Docker 权限报错应通过账号/组配置解决，不要直接放开 Docker socket 的写权限。

### 2.2 配置 NVIDIA Container Toolkit

x86 上先确保宿主机 `nvidia-smi` 成功，再安装 Toolkit：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --batch --yes \
    -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

配置与软件源见 [NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
重启 Docker 应安排在允许现有容器受影响的时段。
Jetson 通常已有 JetPack 安装的 NVIDIA 容器组件，先检查，不要照搬 x86 驱动安装流程：

```bash
docker info --format '{{json .Runtimes}}'
```

Jetson 结果中应包含 `nvidia`。已有 Toolkit 但未注册时，可使用上面的
`nvidia-ctk runtime configure` 和 Docker 重启命令完成配置。

x86 可以先验证 GPU 是否传入一个普通容器：

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

这一步只确认设备接入。第 7 节还会在 Boxer 镜像中验证真实 CUDA 运算。
项目 x86 配置通过 Compose 请求 GPU，机制见
[Docker Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)。

## 3. 准备持久化目录与挂载配置

进入项目根目录，准备部署目录：

```bash
cd /path/to/boxer
mkdir -p ckpts deploy/vocab nav_map output/server
# 仅首次创建；已有 .env 时保留原文件并按需编辑。
cp -n .env.docker.example .env
```

建议的目录结构如下。地图可以稍后放入，也可以使用 Web 上传：

```text
boxer/
├── .env
├── Dockerfile.x86
├── Dockerfile.jetson
├── docker-compose.x86.yaml
├── docker-compose.jetson.yaml
├── ckpts/                 # 三个权重，以及自动生成的文本嵌入缓存
├── deploy/vocab/          # 部署用词表，只挂载 CSV 数据
├── nav_map/
│   └── demo_map/          # 一张完整的 x-nav 地图包
└── output/server/         # 服务状态、历史任务、日志、推理结果
```

编辑 `.env`，本指南后续采用以下配置：

```dotenv
NAV_MAP_DIR=./nav_map
CKPTS_DIR=./ckpts
VOCAB_DIR=./deploy/vocab
OUTPUT_DIR=./output/server
SERVER_PORT=18765
```

| `.env` 变量 | 容器内路径 | 必要权限与用途 |
| --- | --- | --- |
| `NAV_MAP_DIR` | `/nav_map` | 读写；读取地图、接收上传、向地图 `3dbbox/` 导出 CSV |
| `CKPTS_DIR` | `/app/ckpts` | 读写；加载模型，首次初始化可能写入 OWLv2 文本嵌入缓存 |
| `VOCAB_DIR` | `/vocab` | 只读；至少包含 `indoor_classes.csv` |
| `OUTPUT_DIR` | `/app/output/server` | 读写；保存任务记录、日志和结果 |
| `SERVER_PORT` | 容器端口固定 `18765` | 改变宿主机发布端口 |

相对路径以 Compose 文件所在目录为基准。绝对路径指向 **Docker daemon 所在宿主机**，
并非浏览器或远程 Docker 客户端的本地路径。复用已有地图可设置
`NAV_MAP_DIR=/home/glory/Code/digitaltwins-x-nav/nav_map`，实际部署时换成本机路径。

地图、权重和词表目录必须提前存在；Compose 对这三个输入使用
`create_host_path: false`，缺失时直接失败。输出目录虽可自动创建，仍建议提前准备。
若 `ckpts` 是符号链接，确认实际分区已挂载，可用 `readlink -f ckpts` 检查。
不要依赖指向另一台机器 `/media/...` 的旧链接。

容器默认以 root 运行，写出的缓存、结果和 `3dbbox/` 文件可能归 root 所有。
根据宿主机的维护账号、共享目录和 ACL 规划写权限；NFS、只读磁盘或已有地图子目录
仍可能限制容器写入。不要对整个地图根目录递归放开权限。

## 4. 下载、校验和部署权重

### 4.1 必需的三个文件

权重来源为 [Meta 的 facebook/boxer 模型仓库](https://huggingface.co/facebook/boxer/tree/main)。
默认启动必须同时具备以下三个文件，名称保持不变：

| 文件名 | 作用 | 本指南固定版本的大小（字节） |
| --- | --- | --- |
| `boxernet_hw960in2x6d768-c88128f8.ckpt` | BoxerNet，执行 2D 到 3D 框提升 | `343374266` |
| `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` | DINOv3 图像骨干 | `114924783` |
| `owlv2-base-patch16-ensemble.pt` | 项目打包的 OWLv2 检测器，含文本编码器与 tokenizer | `742984963` |

合计 `1201284012` 字节，约 1.20 GB / 1.12 GiB。权重不随镜像打包，
`docker compose build` 和默认服务启动也不会下载它们。
宿主机和容器共用同一套模型文件，x86 和 Jetson 不需要分别下载权重。

`ckpt` API 参数或 CLI 的 `--ckpt` 只选择 **BoxerNet** checkpoint，
不能用它替代 DINOv3 或 OWLv2 的部署。不要拿标准 Hugging Face OWLv2
`safetensors` 文件改名替换本项目的 `.pt`，两者文件结构不同。

### 4.2 使用项目下载脚本

在项目根目录执行：

```bash
bash scripts/download_ckpts.sh
ls -lh ckpts/
```

脚本将文件下载到当前目录下的 `ckpts/`，优先使用 `wget`，否则使用 `curl`。
它不会读取 `.env` 中的 `CKPTS_DIR`。若使用独立权重目录，下载后将这三个文件
复制到实际挂载的目录，或让 `CKPTS_DIR` 指向这里的 `ckpts`。

**脚本会跳过任何已存在的同名文件，且不自动校验哈希。** 中断下载留下的文件、
零字节文件、Git LFS 指针或错误页面可能被再次跳过。下载完成后务必执行 4.4 节校验。
损坏文件先移到备份位置，再重新下载；仅重跑脚本不能修复已存在的损坏文件。

### 4.3 固定版本、支持续传的下载方式

首次部署或下载不稳定时，可用下面的方式替代项目脚本。
这里固定模型仓库 revision 为 `bfba91291fb3f137c18eb6e45362eeda827eac7b`，
对应 4.4 节的校验值；下载到 `.part`，成功后才使用正式文件名：

```bash
(
  set -euo pipefail
  boxer_model_revision=bfba91291fb3f137c18eb6e45362eeda827eac7b
  boxer_model_base="https://huggingface.co/facebook/boxer/resolve/$boxer_model_revision"
  mkdir -p ckpts
  for boxer_model_file in \
    boxernet_hw960in2x6d768-c88128f8.ckpt \
    dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth \
    owlv2-base-patch16-ensemble.pt
  do
    if [ -f "ckpts/$boxer_model_file" ]; then
      printf '已有文件，请校验：%s\n' "$boxer_model_file"
      continue
    fi
    curl --fail --location --retry 5 --retry-delay 3 --continue-at - \
      --output "ckpts/$boxer_model_file.part" "$boxer_model_base/$boxer_model_file"
    mv "ckpts/$boxer_model_file.part" "ckpts/$boxer_model_file"
  done
)
```

续传文件必须来自相同 URL/版本。若服务器不支持续传或 `.part` 本身已损坏，
先将 `.part` 移到备份位置，再重新完整下载。最终仍以哈希校验为准。

### 4.4 SHA256 校验

以下大小和 SHA256 已与
[固定 revision 的官方文件元数据](https://huggingface.co/api/models/facebook/boxer/revision/bfba91291fb3f137c18eb6e45362eeda827eac7b?blobs=true)
核对。默认下载脚本使用 `main`；如果上游以后替换文件，应核对新的官方元数据，
或使用上节固定 revision 下载，不要仅凭相同文件名忽略校验失败。

```bash
cat > ckpts/SHA256SUMS <<'EOF'
d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f  boxernet_hw960in2x6d768-c88128f8.ckpt
4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea  dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf  owlv2-base-patch16-ensemble.pt
EOF
(cd ckpts && sha256sum --check SHA256SUMS)
```

三个文件均应显示 `OK`。如 `CKPTS_DIR` 指向别处，也在那个实际目录执行校验。
文件名中的 `c88128f8` 不等于当前文件 SHA256 前缀，不能代替上述校验。
启动脚本只检查文件非空、可读，不验证模型内容完整性。

## 5. 部署词汇表

### 5.1 复制默认词表

词表是项目数据文件，位于源码 `owl/`；不需要从模型仓库另行下载。
复制到前面配置的独立部署目录：

```bash
cp owl/indoor_classes.csv owl/lvisplus_classes.csv deploy/vocab/
ls -l deploy/vocab/
```

Web 服务每次启动预加载 `indoor`，所以必须提供非空的 `indoor_classes.csv`，
即使之后计划使用其他词表也要保留它。CLI `run_boxer.py` 默认使用 `lvisplus`，
它与 Web 的默认值不同，部署时建议保留两个文件。

如果不需要独立目录，可将 `.env` 中的 `VOCAB_DIR` 改为 `./owl`，直接使用源码词表。
容器始终把该目录挂载到 `/vocab`，并通过 `BOXER_LABELS_DIR=/vocab` 读取。
不要把仅含 CSV 的宿主机目录挂到 `/app/owl`，那会遮蔽 OWLv2 Python 模块。

### 5.2 文件格式与自定义词表

文件名规则为 `<词表名>_classes.csv`。虽然扩展名为 CSV，当前解析器只是逐行读取文本：

- 使用 UTF-8，建议无 BOM；每行一个英文物体类别或简短英文提示，空行会被忽略。
- 不写表头、编号、注释、多列数据或 CSV 引号；这些内容会原样成为类别文本。
- 标签首尾空白会被去除，重复标签不会自动去重，建议事先整理。
- 文件不能只包含空白；保持类别顺序稳定，便于复现与核对结果。

例如新增工厂场景词表：

```bash
cat > deploy/vocab/factory_classes.csv <<'EOF'
chair
table
cabinet
fire extinguisher
forklift
EOF
```

在 Web 的词表输入中填写 **`factory`** 并点击“应用参数”，等待模型重新就绪。
不要填写 `factory_classes.csv`、宿主机文件路径或容器路径。
也可以直接填写 `chair,table,cabinet`，此时按逗号分隔的文本提示检测。

当前查找规则是根据第一项到词表目录查找 `<第一项>_classes.csv`：
找不到时退回普通文本提示。因此 `factroy` 之类的拼写错误可能不报缺文件，
而是只检测这个单词；`indoor,chair` 也不会把 `chair` 合并进已有 `indoor` 词表。

### 5.3 验证词表并应用修改

第 7 节构建镜像后，可以先用同一套挂载检查实际读取内容：

```bash
docker compose run --rm --no-deps server python -c \
  'from utils.taxonomy import load_text_labels; labels=load_text_labels("indoor"); print("类别数:", len(labels)); print(labels[:10])'
```

检查自定义词表时将 `indoor` 改成 `factory`；类别数和前几项应与文件相符。
该命令只读词表，不加载权重、不执行 GPU 推理。

修改词表文件后，待当前任务结束，再点击“应用参数”或重新 `PUT /api/inference`。
服务不会因为宿主机文件变化就自动更新已加载的文本嵌入；新推理前会核对词表。
改变词表内容或顺序会改变相应提示哈希，不需要手动删掉全部权重或缓存。
如果在同一个挂载目录中使用原子替换更新 CSV，下次应用参数即可读取；
如果更改 `VOCAB_DIR`，需用 `docker compose up -d` 重建容器挂载，单纯 `restart` 不生效。

## 6. 准备 x-nav 地图

### 6.1 挂载已有地图

将每张完整的 V6 x-nav 地图放在 `NAV_MAP_DIR` 下，例如 `nav_map/demo_map/`。
不要只复制图像或一个全局点云；通常需要：

```text
demo_map/
├── manifest.txt                 # 完整 V6 快照标识
├── poses_base.txt
├── poses_image_base.txt          # 新七列图像索引必需，旧六列格式不使用
├── camera_calibration.yaml
├── images/index.csv
├── images/...
├── rgb/...
├── surf/...                      # RGB 点云缺失/为空时的回退几何
├── global_map.pcd                # 回放背景，可缺省
└── config/map_config.yaml        # V1 标定转换使用地图自身配置
```

新索引 `image_frame_id,keyframe_id,camera_id,image_stamp_ns,file,source,status`
与旧六列索引的配套文件要求不同，应整体复制同一次导出的完整快照。
具体格式、相机模型和坐标约定见 [地图输入说明](XNAV_MAP.md)。

宿主机 `nav_map/demo_map` 在容器中是 `/nav_map/demo_map`。
Web 下拉列表会递归发现地图包，点击“加载地图”才执行完整校验。
扫描不会跟随目录符号链接；需要复用其他目录时，将 `.env` 的 `NAV_MAP_DIR`
指向真实地图根目录，而不是在根目录里放一批指向挂载外部的链接。

### 6.2 使用 ZIP 上传

可先用空的、可写的 `nav_map/` 启动服务，再通过 Web“上传 ZIP 地图包”导入一张地图。
上传暂存位于 `/nav_map/.boxer_uploads/`，成功后发布到 `/nav_map/uploads/` 并自动注册。
应同时持久化地图目录和服务输出目录，以便重启后继续上传和恢复记录。

默认分块 8 MiB，ZIP 上限 20 GiB，解压上限 100 GiB，最多 200000 个条目，
未完成会话闲置 24 小时后清理。合并阶段保留原分块，峰值空间约为
`2 × ZIP 大小 + 解压大小`。`.env.docker.example` 中的 `BOXER_UPLOAD_*`
可调整限制；修改后用 `docker compose up -d` 应用。
上传协议、续传客户端和限制详见 [OpenAPI 上传说明](OPENAPI.zh-CN.md#66-zip-地图包分块上传与续传)。

## 7. 构建与启动服务

### 7.1 选择 Compose 文件

在项目根目录执行，并在后续操作的同一终端中保留该环境变量。
**x86 NVIDIA 主机选择：**

```bash
export COMPOSE_FILE=docker-compose.x86.yaml
```

**Jetson Orin 主机选择：**

```bash
export COMPOSE_FILE=docker-compose.jetson.yaml
```

只选择与当前宿主机匹配的一项。两套配置均使用项目名 `boxer`，不能在同一主机上
直接当作两套独立服务同时启动。新开终端后需要重新选择，或给每条命令显式加
`-f docker-compose.x86.yaml` / `-f docker-compose.jetson.yaml`。

### 7.2 检查配置、构建和 GPU

```bash
docker compose config --quiet
docker compose config
docker compose --progress plain build

# 使用与正式服务相同的镜像、GPU 配置和挂载，仅执行 CUDA 检查。
docker compose run --rm --no-deps server python /app/scripts/check_cuda.py
```

检查展开后的挂载路径和端口是否符合预期。首次构建需要联网获取基础镜像、
APT/Python/NPM 依赖，Jetson 还需下载指定的 ARM64 PyTorch wheel 和 NVIDIA cuDSS 归档。
Node.js 仅用于前端构建阶段，宿主机和最终服务容器都不需要另装 Node.js。
镜像是无桌面的 x-nav Web 服务环境，不包含桌面查看器及完整 Aria 运行依赖。

CUDA 检查应输出 GPU、Torch、CUDA 版本和
`"cuda_matmul_attention_conv2d": "PASS"`，确认矩阵乘法、注意力、卷积均能执行。
仅看到 `torch.cuda.is_available() == True` 不足以替代此检查。
这时也可以执行第 5.3 节词表检查。

### 7.3 启动与就绪状态

```bash
docker compose up -d
docker compose ps
docker compose logs --tail 200 server
# 持续查看日志；Ctrl+C 仅退出日志跟踪。
docker compose logs -f server
```

访问 [Web 页面](http://127.0.0.1:18765/) 和
[交互式 API 文档](http://127.0.0.1:18765/docs)。局域网访问将 `127.0.0.1`
替换为服务器 IP；修改过 `SERVER_PORT` 时同步替换 URL 端口。
Compose 默认将端口发布到宿主机各接口，服务目前没有账号认证，适用于可信本机/局域网。

启动分为两个阶段：

1. 容器入口检查三个权重、默认词表、目录的实际写权限、缓存路径和 CUDA 运算。
   检查失败时不会启动 HTTP 服务，容器可能因重启策略反复重启，原因在容器日志中。
2. HTTP 启动后，后台常驻 worker 加载 BoxerNet、DINOv3、OWLv2 和默认 `indoor`
   词表并预热。页面此时可访问，但要等 `/api/inference` 的 `status` 为 `ready` 才能推理。

```bash
export BOXER_API=http://127.0.0.1:18765
curl -fsS "$BOXER_API/api/health" | jq .
curl -fsS "$BOXER_API/api/inference" | jq '{status,device,error}'
```

预期模型状态为 `ready`，执行设备为 `cuda`，`error` 为 `null`。
x86 首次启动可能因 OWLv2 Torch/Triton 编译耗时较长，查看 `worker.log` 判断进展。
Jetson 默认设置 `TORCHDYNAMO_DISABLE=1`，避免依赖 Triton 编译。
`/api/health` 的 `status: "ok"` 只说明 HTTP 可响应；Docker 健康检查也允许模型
`loading`，所以 `healthy` 不等于模型已就绪。`unhealthy` 本身不会触发 Docker 自动重启。

## 8. 首次推理验收

### 8.1 Web 验收流程

1. 从下拉列表选择地图并“加载地图”，确认相机图像、点云和位姿可读。
2. 在推理参数中选 `indoor` 或自定义词表，首次将 `max_n` 设为 `5`，开启跟踪，
   应用参数并等待模型就绪。
3. 开始推理，等待任务状态为 `complete`，检查图像、3D 回放和 CSV 文件。
4. 小规模任务通过后，重新应用全量参数，再启动全图推理。

`max_n` 限制模型处理帧数，不会省略地图加载时的完整数据校验。
新图像索引按图像组计数，旧格式按 LiDAR 关键帧计数；API `start_n` 从 1 开始。
任务完成后可能仍没有确认物体：默认跟踪需要同一静态物体得到三个不同处理帧支持，
少量帧、无检测或阈值偏高时，CSV 只有表头也可能是正常结果。

### 8.2 API 验收示例

以下使用第 7 节的 `BOXER_API` 和示例地图路径；先确认当前模型为 `ready`。
应用五帧参数：

```bash
curl -fsS -X PUT "$BOXER_API/api/inference" \
  -H 'Content-Type: application/json' \
  -d '{"labels":"indoor","max_n":5,"track":true,"profile":true}' | jq .
curl -fsS "$BOXER_API/api/inference" | jq '{status,device,error}'
```

如仍为 `loading`，稍后重复 GET，等 `ready` 后再执行下一段。
`PUT` 是完整配置解析，省略字段使用默认值，不会保留当前自定义配置。
需要只修改一个字段时，先 GET 完整 `options`，修改后再 PUT。

```bash
boxer_map_id=$(curl -fsS -X POST "$BOXER_API/api/maps" \
  -H 'Content-Type: application/json' \
  -d '{"path":"/nav_map/demo_map"}' | jq -er '.id')
# 确认得到非空地图 ID 后创建任务，省略 options 以使用已应用配置。
printf '%s\n' "$boxer_map_id"
boxer_job_id=$(curl -fsS -X POST "$BOXER_API/api/jobs" \
  -H 'Content-Type: application/json' \
  -d "{\"map_id\":\"$boxer_map_id\"}" | jq -er '.id')
curl -fsS "$BOXER_API/api/jobs/$boxer_job_id" | jq .
```

重复查询任务，直到 `complete`、`failed` 或 `interrupted`。
HTTP `202` 表示受理，`processed_frames == total_frames` 也不保证收尾和导出成功，
最终以任务 `status` 和 `error` 为准。小规模验收后记得将 `max_n` 恢复为所需值，
例如默认的 `99999`。

### 8.3 检查结果和持久化

服务输出默认在宿主机 `output/server/`：

```text
output/server/
├── worker.log
├── maps/<服务地图ID>.json
└── jobs/<任务ID>/
    ├── job.json
    ├── inference.log
    └── output/<map_id>/<snapshot_id>/<camera_mode>/
        ├── results.json
        ├── owl_2dbbs.csv
        ├── boxer_3dbbs.csv
        ├── boxer_3dbbs_tracks.csv
        ├── boxer_3dbbs_tracked.csv
        └── performance.json       # profile=true 时生成
```

以 API 返回的 `results_path` 为准，服务地图 ID 与地图清单的 `map_id` 不是同一个概念。
`no_csv=false` 且任务成功时，四份 CSV 还会保存到宿主机地图的
`nav_map/demo_map/3dbbox/`，覆盖已有同名结果；任务目录保留原件。
需要保留之前的地图导出时，先备份该目录。

关闭浏览器不影响后台推理。同一服务同时只接受一个推理任务。
服务重启会恢复历史记录，将未结束的任务标记为 `interrupted`，不会自动断点续推；
当前推理参数也会恢复默认值，重启后重新应用自定义词表和参数。

## 9. 离线迁移、更新和日常操作

### 9.1 离线部署

在相同 CPU 架构、兼容 GPU/JetPack 环境的联网机器上完成镜像构建。
导出对应镜像，例如 x86：

```bash
docker image save boxer-server:x86 | gzip > boxer-server-x86.tar.gz
```

Jetson 改为导出 `boxer-server:jetson`。拷贝到目标主机后：

```bash
docker image load -i boxer-server-x86.tar.gz
docker compose up -d --no-build --pull never
```

目标主机需提前安装可用的 Docker 和 NVIDIA 运行时，并选择对应 `COMPOSE_FILE`。
镜像之外还要迁移 Compose 文件、用于路径配置的 `.env`、三个校验通过的权重、
词汇表和地图；需要恢复历史时再迁移整个 `OUTPUT_DIR`。
权重若是符号链接，要复制其实际文件，不能只复制链接本身。

构建后的前端、Three.js 和 Swagger UI 资源均随服务提供，正常使用不需要浏览器访问 CDN。
OWLv2 的 tokenizer 与文本编码器已包含在项目 `.pt` 中，默认模型初始化无需再向
Hugging Face 下载；首次仍可能本地生成文本嵌入和编译缓存。
缓存卷不属于必需迁移项，缺失时会重建。
历史记录保存容器路径，迁移后尽量保持 `/nav_map` 和 `/app/output/server` 的布局。

### 9.2 更新与停止

```bash
# 查看状态和最近日志
docker compose ps
docker compose logs --tail 200 server

# 更新源码后构建并应用新镜像；.env/挂载配置变化也用 up 应用。
docker compose up -d --build

# 只重启现有容器，不重新读取 .env 或替换镜像。
docker compose restart server

# 停止并删除服务容器/网络，保留宿主机数据与命名缓存卷。
docker compose down
```

更新前先结束正在执行的任务，并备份 `.env`、部署词表、权重校验清单和服务输出。
缓存使用 Compose 的 `cache` 命名卷，保存编译/CUDA 缓存。
正常停止发送 SIGINT，最多提供 90 秒宽限期；`down -v` 会额外删除命名卷缓存，
日常停服务不需要该选项。`docker compose restart` 也不会修复错误挂载或应用新端口。

## 10. 常见问题与处理

### 10.1 安装、构建和 GPU

| 现象或日志 | 常见原因 | 检查与处理 |
| --- | --- | --- |
| `docker: permission denied` / 无法访问 socket | 当前账号没有 Docker 权限 | 用 `sudo docker info` 区分服务问题与账号问题；配置组后重新登录 |
| `docker: 'compose' is not a docker command` | Compose 插件缺失或使用旧版本 | 安装 `docker-compose-plugin`，确认 `docker compose version` 成功 |
| `could not select device driver ... [[gpu]]` | NVIDIA Toolkit 未安装/未注册 | 检查 `docker info`、`nvidia-ctk` 配置；x86 先通过宿主机和容器的 `nvidia-smi` |
| Jetson `unknown or invalid runtime name: nvidia` | Docker 未注册 NVIDIA runtime | 检查 `.Runtimes`，配置 runtime 后重启 Docker |
| `exec format error` / wheel 架构或 Python 不匹配 | x86/ARM 镜像或依赖混用 | 核对 `uname -m`、Compose 的 `platform`、Python 版本；在目标平台重建 |
| 拉取镜像、APT、pip 或 NPM 超时/证书失败 | 构建主机网络、代理、DNS、系统时间问题 | 用 `--progress plain` 定位失败地址；分别检查 Docker daemon 拉镜像和构建阶段联网；内网可走离线镜像迁移 |
| `no space left on device` | Docker 数据目录、构建缓存或结果磁盘满 | 检查 `df -h` 与 `docker system df`；区分 Docker 分区和挂载数据分区后处理 |
| CUDA 不可用或驱动/内核不兼容 | 宿主机驱动、GPU 注入或平台基线不匹配 | 运行镜像中的 `scripts/check_cuda.py`；不要在容器里安装宿主机驱动来修复 |
| x86 首次长时间 `loading` | 模型读取、预热或 Torch/Triton 首次编译 | 查看 `OUTPUT_DIR/worker.log` 与 GPU/内存占用，确认仍在进展；保留缓存卷 |
| Jetson 缺 cuDSS、cuSOLVER 符号或 cuBLAS 报错 | JetPack/wheel 不匹配，或额外依赖替换了系统库 | 按指定 Dockerfile 重建；使用其独立 cuDSS 归档，不向容器随意安装其他 CUDA/cuDSS wheel |

镜像构建后的 GPU 运算检查比单独查看版本号更有价值。若 `check_cuda.py` 失败，
先修复 GPU 环境，再尝试模型加载。
Jetson 当前规避了固定正交旋转上的通用 CUDA 求逆需求，扩展功能额外使用 CUDA
线性代数求解器时仍需单独验证；背景见 [Jetson 环境说明](JETSON_ORIN_NX.md)。

### 10.2 权重、词表与挂载

| 现象或日志 | 检查与处理 |
| --- | --- |
| `bind source path does not exist` | 在 Docker daemon 所在主机创建输入目录；检查 `.env` 的实际展开路径和符号链接目标 |
| `Required model is missing or empty` | 检查实际 `CKPTS_DIR` 中三个文件名、大小、可读权限，不要把文件放进多余的一层子目录 |
| `torch.load` 的 EOF、归档损坏或反序列化错误 | 对实际挂载权重运行 SHA256；排除残留下载、HTML 错误页和 Git LFS 指针，再重新下载 |
| `Default vocabulary is missing/empty` | 检查 `VOCAB_DIR/indoor_classes.csv` 是否存在且有非空标签；自定义词表不能替代启动必需的默认文件 |
| 自定义词表“加载成功”但类别数不对 | 用第 5.3 节打印类别；检查命名、拼写、BOM、表头，以及是否被回退为单个普通提示 |
| `Models or vocabulary changed; apply inference parameters again` | 当前加载的模型/词表与文件不一致；任务结束后重新应用参数并等待 `ready` |
| `Directory must be writable` / 文本嵌入缓存 `PermissionError` | 检查权重、地图根目录、输出和缓存挂载的实际写权限；`CKPTS_DIR` 不能只读挂载 |
| 修改 `.env` 后路径或端口仍旧 | 用 `docker compose config` 核对，再执行 `docker compose up -d`；`restart` 保留旧容器配置 |

容器未能启动时优先查 `docker compose logs --tail 200 server`。
HTTP 已启动但模型失败时，查 `/api/inference` 的 `error` 和宿主机
`OUTPUT_DIR/worker.log`。前者通常是挂载/启动检查问题，后者包含模型加载与预热信息。

### 10.3 地图、推理和结果

| 现象 | 检查与处理 |
| --- | --- |
| Web 地图列表为空 | 检查 `/api/maps/directories`；确认挂载的是地图根目录、包内有标识文件，且地图不是扫描不会跟随的目录符号链接 |
| 加载地图失败 | 检查是否为完整 V6 快照，索引/位姿格式是否配套，图像能否解码、点云和标定是否齐全；列表发现地图不代表校验通过 |
| Web 可打开但创建任务返回 `503` | 查 `/api/inference`，等待 `ready`；若为 `failed`，修复 `error` 后重新应用参数 |
| 应用参数或创建任务返回 `409` | 有活动任务/模型正在准备，或任务参数与已应用完整配置不一致；等待空闲后应用配置，创建任务时省略 `options` |
| `422` 参数错误 | 查响应 `detail`；如 API 的 `start_n`、`max_n` 都必须至少为 1，CLI 的某些参数不能直接传给 API |
| CUDA 显存不足、进程被 OOM 结束 | 释放其他任务占用；先用较小词表、单相机、batch=1 或 `xnav_execution=serial` 定位；Jetson 同时检查共享内存和系统内存 |
| 减小 `max_n` 或 `detector_hw` 后加载仍占显存 | `max_n` 只减少处理数量；当前 OWLv2 使用权重原生输入尺寸，`detector_hw` 不会改变其实际输入分辨率 |
| 帧数已处理完但任务 `failed` | 查任务 `error`，可能是最终 CSV 导出失败；检查具体地图的 `3dbbox/`，不能只检查 `/nav_map` 根目录可写 |
| CSV 只有表头或最终物体数为零 | 检查逐帧 2D/3D 检测、词表、阈值和跟踪确认条件；少量帧没有确认轨迹不等于部署失败 |
| 重启后历史存在但推理没有继续 | 当前设计只恢复记录，将未完成任务标为中断；需要创建新任务，跟踪器也会重建 |
| 浏览器点云黑屏/远端无法访问 | 先本机检查 HTTP 与 `/api/health`；远端核对发布端口、网络规则和代理，浏览器确认 WebGL 可用 |
| ZIP 上传超限、失败或磁盘不足 | 检查 `BOXER_UPLOAD_*`、ZIP 内容和地图有效性；预留分块、合并及解压空间；通过上传状态中的错误判断是否重试 |

显存排查时，支持 BF16 的 GPU 可保持自动精度；不应默认改成占用更多内存的 float32。
降低 `max_n` 适合缩短验收时间，不能解决常驻模型本身装不下的问题。

若错误明确为 `Could not export map CSVs`，而任务目录已有完整的四份 CSV 和可回放结果，
不必为了补文件立即重跑推理。先备份旧导出并修复目标地图 `3dbbox/` 的权限，
再从该任务实际结果目录补拷贝四份文件；若需使用项目的暂存/逐文件替换方式，
可在维护脚本中调用 `utils.xnav_export.export_map_csv(source, map_root)`。
该函数两个参数都使用执行环境内的路径，`source` 是四份 CSV 所在目录，
`map_root` 是地图包根目录。手动补导出不会自动把原失败任务的状态改成完成。
如果只需要 JSON/API 回放，可以在之后的任务中应用 `no_csv=true`，但这也会关闭任务 CSV。

## 11. 部署完成检查

- [ ] 宿主机架构、驱动/JetPack 与所选 Compose 一致。
- [ ] `docker compose config --quiet` 成功，四个数据挂载指向预期路径。
- [ ] 三个权重已通过 SHA256 校验，实际挂载目录可写。
- [ ] 默认 `indoor` 词表存在，自定义词表读取的类别与预期一致。
- [ ] `check_cuda.py` 的矩阵乘法、注意力和卷积检查为 `PASS`。
- [ ] `/api/inference` 为 `ready` 且设备为 `cuda`。
- [ ] 一张完整地图能通过校验，小规模任务最终为 `complete`，结果可以回放。
- [ ] 开启 CSV 时，任务目录与地图 `3dbbox/` 均有预期文件。
- [ ] 已记录源码版本、模型 revision/哈希、词表副本、`.env` 与部署平台。

更多说明：[Docker 配置参考](DOCKER.zh-CN.md)、[Web 使用说明](WEB_SERVER.zh-CN.md)、
[完整 OpenAPI](OPENAPI.zh-CN.md)、[x-nav 地图约定](XNAV_MAP.md)。
需要原生 Python 环境或桌面查看器时，分别参考
[本机 CUDA 环境](LOCAL_CUDA.md)和 [Jetson 部署记录](JETSON_ORIN_NX.md)。
