# SWE-Vision 手把手运行教程（WSL2 + G 盘存储版）

> 本教程针对你的实际环境（WSL2 发行版 `Ubuntu_Experiment`、Docker Desktop、项目在 G 盘）编写，
> 所有新产生的数据（Docker 镜像、虚拟环境、运行中间文件、Web 会话）都放在 G 盘，不占用 C 盘空间。

以下命令假定你已经进入项目根目录；先执行一次：

```bash
PROJECT_ROOT="$(pwd)"
VENV_PATH="${VIRTUAL_ENV_PATH:-$HOME/swe-vision-venv}"
```

---

## 0. 你的环境现状（已检测）

| 项目 | 状态 |
|---|---|
| WSL 发行版 | 已启用 systemd 的 WSL2 发行版 |
| Python | 3.12.3（系统自带，PEP 668 限制，**必须用虚拟环境**；最终用 venv，见第 3 步） |
| Docker | Docker Desktop 已安装；请为当前发行版开启 WSL 集成 |
| C 盘 | 只剩 51G（78% 已用）—— 本教程全程不碰 C 盘 |
| D 盘 | 只剩 103G（86% 已用）⚠️ Docker 数据与 WSL 发行版目前都在这里 |
| G 盘 | 剩余 760G ✓ 项目本身位于当前项目根目录 |

**结论：C 盘完全不用动（Docker 和 WSL 发行版本体之前已经被你放到 D 盘了）；本项目新增的数据全部放 G 盘。**

## 存储位置总览（全部落在 G 盘）

| 数据 | 当前/默认位置 | 本教程方案 |
|---|---|---|
| Docker 镜像 / 容器数据 | 已在 D 盘 Docker 虚拟磁盘（你之前挪过） | 二选一：留在 D，或迁到 `<项目根>\\docker-data`（见第 2 步） |
| Python 环境（venv） | 无（自建） | `$VENV_PATH`（建议放在 WSL Linux 文件系统，见第 3 步） |
| 内核工作目录（`VLM_HOST_WORK_DIR`） | `<项目根>/runtime/swe-vision/workdir` | 同左（程序自动使用） |
| Web UI 会话（`VLM_WEB_SESSION_DIR`） | `<项目根>/runtime/swe-vision/web_sessions` | 同左（程序自动使用） |
| 运行轨迹 | `./trajectories/`（相对项目目录） | 已经是 G 盘 ✓ 无需改 |

> 说明：`VLM_HOST_WORK_DIR` 是宿主机与 Docker 容器共享文件的目录（见
> [config.py](swe_vision/config.py)），默认已锚定到项目根目录下的 `runtime/swe-vision/workdir`。

---

## 第 1 步：启动 Docker Desktop 并开启 WSL 集成（必做）

1. 在 **Windows** 上启动 Docker Desktop（开始菜单搜索 Docker Desktop，等右下角鲸鱼图标变绿）。
2. 打开 Docker Desktop → **Settings → Resources → WSL integration**。
3. 勾选 **`Ubuntu_Experiment`** → 点 **Apply & Restart**。
   - （如果你以后在 `Ubuntu` 发行版里也跑，可一并勾选。）
4. 回到 WSL 终端验证：

```bash
docker version          # 能看到 Client 和 Server 版本即成功
docker info | grep -E "Server Version|Storage Driver"
```

> 如果 WSL 里仍提示 "The command 'docker' could not be found"，说明集成没生效，重启 Docker Desktop 或注销重进 WSL（`wsl.exe --shutdown` 后重开终端）。

## 第 2 步：Docker 数据的位置（查证结果：已在 D 盘，不在 C 盘）

查证结果：你之前已经把 Docker Desktop 数据挪到了 **D 盘**——

- 配置：Docker Desktop 的 Disk image location
- 数据文件：该位置下的 `docker_data.vhdx`

所以 **C 盘不会受影响**。注意：Docker 不支持"只把某个镜像放 G 盘"——所有镜像都存进同一个 vhdx 文件，二选一即可：

**方案 1：留在 D 盘（最省事，什么都不用做）**

D 盘还剩 103G，本项目镜像（火山引擎 `code-sandbox` + 构建层，预计 10~20G）放得下。只是 D 盘已用 86%，如果 D 盘还经常装别的东西，将来可能紧张。

**方案 2：整体迁到 G 盘（G 剩 760G，最宽裕；只迁 6.9G，几分钟）**

1. 右键托盘 Docker Desktop 图标 → **Quit Docker Desktop**（完全退出）。
2. 重新打开 Docker Desktop → **Settings → Resources → Advanced → Disk image location**。
3. 改为项目根目录下的 `docker-data` → **Apply & Restart**（Docker Desktop 会自动把 vhdx 迁移过去）。

验证（PowerShell）：

```powershell
dir <项目根>\\docker-data     # 应能看到 docker_data.vhdx
```

> 两种方案都不影响本教程其余步骤，下面的命令与数据放 D 还是 G 无关。
> 顺带建议：**Settings → Resources** 里把 Memory 调到 **≥ 8 GB**，沙箱镜像跑 Jupyter 需要内存。

## 第 3 步：安装 Python 环境（venv，放发行版磁盘）

⚠️ **Python 环境不能放 G 盘**（实测踩坑）：`/mnt/g` 是 Windows 文件系统，WSL 在上面**创建符号链接会被拒绝**
（`Operation not permitted`，需要 Windows 开发者模式才放开），conda 建环境和 venv 建环境都会失败。
因此环境放在发行版磁盘（D 盘，仅约 1GB）：

```bash
cd "$PROJECT_ROOT/SWEVISSION"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"   # 以后每次跑之前先执行这一句
pip install -r requirements.txt         # openai / jupyter_client / docker / flask / Pillow 等
python -c "import swe_vision; print('依赖 OK')"
```

> 项目代码、运行轨迹、Docker 工作目录等其它数据仍在 G 盘；只有这 ~1GB 的环境在 D 盘。

## 第 4 步：配置 LLM API（3 个环境变量）

项目通过 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` 三个环境变量连 LLM
（[agent.py](swe_vision/agent.py#L58-L66) 会自动读取，`openai` 库也自动读 `OPENAI_API_KEY`）。

**推荐做法：** 在项目根目录建一个 `local_env.sh`（放你自己的真实值），每次跑之前 `source` 一下：

```bash
cd "$PROJECT_ROOT"
cat > local_env.sh << 'EOF'
# OpenAI 兼容的 API 配置 —— 改成你自己的值，此文件不要提交到 git
# 示例：阿里云百炼 DashScope 跑 Qwen 视觉模型（已实测可用）
export ALIYUN_API_KEY="sk-你的阿里云百炼密钥"
export OPENAI_API_KEY="$ALIYUN_API_KEY"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen3-vl-plus"
# 运行期数据全部放 G 盘（见存储总览表）
# 运行目录默认自动使用 <项目根>/runtime/swe-vision/，无需设置额外路径变量。
EOF
echo "local_env.sh" >> .gitignore        # 防止密钥被提交
```

使用时：

```bash
cd "$PROJECT_ROOT/SWEVISSION"
source "$VENV_PATH/bin/activate"
source ../local_env.sh
```

> **注意**：⚠️ 不要直接跑 [run.sh](run.sh) —— 它把 `OPENAI_API_KEY="YOUR_API_KEY_HERE"` 写死在脚本里，会覆盖你的配置。要么按第 6 步用命令行方式跑，要么先编辑 run.sh 里那 3 行占位符。
>
> 模型必须**支持视觉输入**（项目会把图片以 base64 发给模型）。
> ⚠️ 两个实测结论：① 模型必须选 `qwen3-vl-plus`——`qwen-vl-max` / `qwen-vl-plus` 在该端点**不返回工具调用**（模型会把"调用工具"写成普通文本，agent 无法执行代码）；② 项目默认的 `--reasoning` 参数会被服务端拒绝（`thinking_budget` 错误），**所有命令都要加 `--no-reasoning`**；Web UI 里把 reasoning 开关关掉。

## 第 5 步：构建 Docker 沙箱镜像

[env/Dockerfile](env/Dockerfile) 基于火山引擎的公开镜像 `vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609`：

```bash
cd "$PROJECT_ROOT/SWEVISSION"
docker build -t swe-vision:latest -f ./env/Dockerfile ./env
```

- 基础镜像**体积很大**（数 GB），第一次拉取请耐心等待（数据落在第 2 步你选定的位置：D 盘或 G 盘）。
- 本步可以跳过：第一次跑 agent 时，[kernel.py](swe_vision/kernel.py#L74-L101) 检测不到镜像会**自动构建**。
- 如果拉取超时/失败（国内网络），在 Docker Desktop **Settings → Docker Engine** 里给 `"registry-mirrors"` 配一个国内加速地址后重试。

验证：

```bash
docker images | grep swe-vision     # 应看到 swe-vision:latest
```

## 第 6 步：跑起来（CLI 方式）

```bash
cd "$PROJECT_ROOT/SWEVISSION"
source "$VENV_PATH/bin/activate"
source ../local_env.sh

# ① 默认示例：项目自带测试图（G 盘 assets/test_image.png）
#    （DashScope/qwen3-vl-plus 必须加 --no-reasoning，见第 4 步说明）
python -m swe_vision.cli --image assets/test_image.png --no-reasoning \
    "What is the gap between GPT5.2 and 6-year-olds from the chart?"

# ② 换成你自己的图
python -m swe_vision.cli -i <图片路径> --no-reasoning "这张图里有什么物体？"

# ③ 多图对比
python -m swe_vision.cli -i a.png -i b.png --no-reasoning "比较这两张图的区别"

# ④ 纯计算问题（不需要图片）
python -m swe_vision.cli --no-reasoning "用 Python 计算前 20 个斐波那契数"

# ⑤ 交互模式（多轮对话）
python -m swe_vision.cli --interactive --no-reasoning
```

每次运行的内部流程：起 Docker 容器（挂载 G 盘 `runtime/workdir` 到容器 `/mnt/data`）→ 启动 Jupyter kernel → agent 循环调 `execute_code` 工具 → 最终调 `finish` 输出答案 → 容器销毁。

> `/mnt/g` 是 drvfs（Windows 文件系统），挂进容器的 IO 比原生 Linux 慢一些，但本项目只传图片和代码文件，完全够用。
> 如果想指定轨迹保存位置：加 `--save-trajectory <输出目录>`。

## 第 7 步：Web UI（可选）

ChatGPT 风格界面，实时流式展示 agent 的思考、代码执行和结果：

```bash
cd "$PROJECT_ROOT/SWEVISSION"
source "$VENV_PATH/bin/activate"
source ../local_env.sh

python apps/web_app.py --port 8080
# Windows 浏览器直接打开 http://localhost:8080
# （WSL2 会自动把 localhost 转发到 Windows，无需额外配置）
# ⚠️ 用 qwen3-vl-plus 时：把界面上的 Reasoning 开关关掉（默认开启，会被 DashScope 拒绝）
```

## 第 8 步：轨迹查看器（可选）

每次运行都会把完整轨迹（JSON + 图片）存到 `./trajectories/`（已在 G 盘），用可视化面板回放：

```bash
python apps/trajectory_viewer.py --port 5050
# 浏览器打开 http://localhost:5050
```

---

## 跑通后的验证清单

- [ ] `docker version` 在 WSL 里能同时看到 Client 和 Server
- [ ] docker 数据 vhdx 在你选定的位置（D 盘原位置或项目根目录下的 `docker-data`）
- [ ] `docker images` 里有 `swe-vision:latest`
- [ ] agent 输出了最终答案（日志出现 "finish" / 最终回答）
- [ ] `runtime/workdir/` 下出现以时间戳命名的目录（证明 G 盘挂载生效）
- [ ] `./trajectories/` 下出现新的 `run_xxx` 目录

## 常见问题排查

| 现象 | 原因 / 解决 |
|---|---|
| WSL 里提示 docker 命令不存在 | 第 1 步的 WSL 集成没生效；Docker Desktop 重启后重进 WSL。**实测注意**：`wsl --shutdown` 或 Docker Desktop 大重启后，集成可能丢失，去 Settings → Resources → WSL integration 重新勾选本发行版即可 |
| `pip install` 报 externally-managed / 找不到模块 | 没激活 venv；回到第 3 步 `source "$VENV_PATH/bin/activate"` |
| 构建镜像时卡在拉取 `vemlp-cn-beijing...` | 国内网络；配置 registry mirror 后重试，或耐心等待 |
| 报 "Docker Jupyter kernel failed health check" | 容器没起来：`docker ps -a` 看日志；检查端口 65500–65504 是否被占用；Docker Desktop 内存调大（第 2 步末尾） |
| 跑完发现 D 盘还在变小 | 检查 `VLM_HOST_WORK_DIR` 是否真的改了（默认值在发行版磁盘里，占 D 盘）；`echo $VLM_HOST_WORK_DIR` 确认 |
| `runtime/workdir/` 越积越多 | 每次运行产生一个时间戳目录，定期清理：`rm -rf runtime/workdir/*` |
| 模型报 404 / 不支持 reasoning | 你的模型 ID 写错或该型号不支持 reasoning，换模型或加 `--no-reasoning` |

## 日常运行三句话总结

```bash
cd "$PROJECT_ROOT/SWEVISSION"
source "$VENV_PATH/bin/activate" && source ../local_env.sh   # 前提：Docker Desktop 已启动
python -m swe_vision.cli -i 你的图.png --no-reasoning "你的问题"
```

---

## 附录：运行原理与组件清单

### 必须装的东西（完整清单）

| 组件 | 作用 | 位置 |
|---|---|---|
| Docker Desktop + WSL 集成 | 跑沙箱容器（**必须**，代码写死了 Docker） | 数据在 D 盘（或迁 G） |
| Python 环境（venv） | 跑 agent 主程序 | `$VENV_PATH`（建议放在 WSL Linux 文件系统） |
| Python 依赖（`openai` / `jupyter_client` / `docker` / `flask` / `Pillow` / `regex`） | agent、内核连接、Web UI | conda 环境内 |
| `swe-vision:latest` 镜像 | 沙箱（自带 numpy/pandas/matplotlib 等 + Jupyter） | Docker 数据盘 |
| LLM API（OpenAI 兼容、支持视觉） | 模型调用（云端服务，无需本地安装） | 云端 |

不需要：GPU/CUDA、数据库、本地大模型。需要网络：拉镜像 + 调 API。

### 运行流程

```
用户: 问题 + 图片
  │
  ▼
VLMToolCallAgent.run()
  │ ① 图片编码成 base64 发给 LLM，同时把图片文件复制到
  │    runtime/workdir/<时间戳>/（被挂载为容器内 /mnt/data）
  │ ② 组装消息: [system提示词, user(文本+图片)]
  ▼
┌──── agent 循环（最多 100 轮）────────────────────────────┐
│ 调用 LLM（OpenAI 兼容 API，携带 tools 定义）             │
│                                                         │
│ ├─ 模型返回 execute_code 调用                           │
│ │     └─► 首次触发：启动 Docker 沙箱                    │
│ │     └─► 代码在容器 Jupyter kernel 里执行               │
│ │     └─► 结果（文本 + base64 PNG 图）作为 tool 消息回传 │
│ │                                                       │
│ ├─ 模型继续分析 → 再次调 execute_code（状态保留）        │
│ └─ 模型调 finish(answer) ──► 返回最终答案                │
└─────────────────────────────────────────────────────────┘
  │
  ▼
轨迹保存到 ./trajectories/run_<时间戳>/（JSON + 图片）     │
  │
  ▼
cleanup() → 停止并删除容器（G 盘 workdir 目录保留）
```

**Docker 沙箱启动细节**（第一次 `execute_code` 时触发）：

1. 检查 `swe-vision:latest` 镜像，没有就 `docker build`（拉取 code-sandbox 基础镜像，数 GB）
2. 在 G 盘 workdir 写连接文件（端口 65500–65504、HMAC 密钥）
3. `docker run -d`：挂载 G 盘 workdir → 容器 `/mnt/data`，绑定 5 个端口到宿主机 127.0.0.1
4. 容器内启动 `python -m ipykernel_launcher -f /mnt/data/.kernel_connection.json --IPKernelApp.matplotlib='inline'`
5. 宿主机用 `jupyter_client` 连上 ZMQ 端口，健康检查后开始执行
6. 每次执行：发代码 → 收集 stdout/stderr/报错/matplotlib 图片 → 文本 + base64 图片回传模型

> 注意：Docker 是**懒启动**的——只有模型调用 `execute_code` 时才起容器；纯聊天问题不会碰 Docker。
