# Resume Agent

<div align="center">

**本地优先的中文简历智能体：岗位发现、JD 拆解、项目生成、简历修改与单页导出。**

[English](README.md) · [简体中文](README.zh-CN.md)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C)
![MCP](https://img.shields.io/badge/MCP-FastMCP%20stdio-6F42C1)
![Deployment](https://img.shields.io/badge/Deployment-Local--first-2EA44F)
![License](https://img.shields.io/badge/License-尚未声明-lightgrey)

</div>

Resume Agent 面向“一岗一简历”场景：通过独立 Microsoft Edge 采集 BOSS 岗位，使用云端 OpenAI-compatible 或本地 Ollama 模型拆解 JD，生成多个具体项目候选，并将用户确认的候选新增或替换到已有简历，也可基于中文模板生成完整简历，最终导出 Markdown、DOCX 或 PDF。

系统坚持人工确认：模型推导内容保持“待核实”，模板和简历结构必须可编辑确认，所有修改使用带旧值哈希的字段级补丁，并通过 checkpoint 版本阻止旧页面覆盖新状态。

> 当前状态：可本地运行的 MVP。BOSS 页面和风控策略可能变化，Edge 或网站更新后仍需进行真实页面人工回归。

## 核心能力

- Streamlit 问答向导，按选择逐步显示后续操作。
- FastAPI 后端，LangChain + LangGraph 智能体与持久化 `thread_id`。
- 基于 FastMCP、`stdio` 的标准 BOSS/GitHub MCP 工具。
- 云端聊天与 Embedding 模型独立配置；Ollama 动态扫描本机真实模型。
- 云端 API Key 仅在本机加密保存，不写入 PostgreSQL、日志或 Git。
- 上传一份 DOCX、Markdown、TXT 或可复制文本 PDF（最大 10 MB，PDF 最多 3 页）。
- 模型动态拆解简历和模板结构，用户可编辑后确认。
- 剪贴板粘贴或上传岗位截图，PaddleOCR v6 识别、校正、模型增强和失败回退。
- BOSS 岗位每批 10 条动态加载，保留采集页并在后台解析详情。
- “Python、Java、Go 任选其一”等技能保持 OR 语义，不拆成全部必需项。
- 候选数量不足时保留已有合法候选，继续补生成缺少数量。
- 每个项目候选都可在选择前直接编辑并持久化保存；保存会使旧字段确认失效，并记录基于哈希的编辑审计信息。
- 明确选择新增或替换项目；替换前检查目标 ID 和旧值哈希。
- Markdown、DOCX、PDF 预览与下载；DOCX 可转换为 PDF 进行视觉检查。
- 超过一页时先请求用户同意字体、行距和页边距压缩，改写正文需再次授权。

## 整体流程

```mermaid
flowchart TD
    A[配置 Edge 与模型] --> B{简历来源}
    B -->|上传简历| C[提取文字与版面]
    B -->|使用模板| D[查找并预览中文模板]
    B -->|仅项目文本| E[准备可复制输出]
    C --> F[模型拆解简历结构]
    D --> G[模型拆解模板结构]
    F --> H[用户编辑并确认区块]
    G --> H
    E --> I{JD 来源}
    H --> I
    I -->|岗位和城市| J[Edge 分批采集 BOSS]
    I -->|岗位截图| K[PaddleOCR 与校正]
    I -->|公司官网| L[预留适配器]
    J --> M[后台解析完整 JD]
    K --> M
    M --> N[规则解析加所选聊天模型]
    N --> O[用户编辑并确认 JD]
    O --> P[LangGraph 匹配与候选生成]
    P --> Q{候选数量满足?}
    Q -->|否| R[保留已有候选并补齐]
    R --> Q
    Q -->|是| S[编辑保存、选择并逐字段确认候选]
    S --> T{输出方式}
    T -->|纯文本| U[复制项目经历]
    T -->|新增或替换| V[哈希校验后的局部补丁]
    V --> W[预览与单页检查]
    W --> X[下载 Markdown/DOCX/PDF]
```

## 快速开始

第一版主要面向 Windows 10/11，推荐 Python 3.12。

### 1. 安装工具

```powershell
winget install --id astral-sh.uv -e
winget install --id Git.Git -e
winget install --id Microsoft.Edge -e
winget install --id Docker.DockerDesktop -e
```

### 2. 下载并安装依赖

```powershell
git clone <你的 GitHub 仓库地址> resume-agent
Set-Location resume-agent
uv python install 3.12
uv venv --python 3.12 .venv
uv sync --extra ui --extra mcp --extra documents --extra ocr --extra dev
```

如果项目已有 `.venv`，请保留并直接运行 `uv sync ...`，无需重建、降级或删除已有依赖。

uv 在 Windows 提示 hardlink 不可用、改为复制并不影响运行。可在当前终端执行：

```powershell
$env:UV_LINK_MODE="copy"
uv sync --extra ui --extra mcp --extra documents --extra ocr --extra dev
```

### 3. 启动 PostgreSQL 并迁移数据库

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
docker compose ps
uv run python -m alembic upgrade head
```

只有 PostgreSQL 使用 Docker；FastAPI、Streamlit、Edge、MCP、OCR 和模型均在本机运行。

临时体验时可不配置 `RESUME_AGENT_DATABASE_URL`，再运行 `uv run python scripts/init_db.py` 使用 `data/resume_agent.db`。长期使用推荐 PostgreSQL。

### 4. 启动后端

```powershell
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

FastAPI 会自动启动并管理两个 FastMCP `stdio` 子进程，不要单独启动 MCP 服务。

### 5. 启动网页

在第二个 PowerShell 窗口执行：

```powershell
uv run python -m streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8666
```

- 网页：<http://127.0.0.1:8666>
- 健康检查：<http://127.0.0.1:8000/api/health>
- API 文档：<http://127.0.0.1:8000/docs>

## 如何关闭、重启并保留数据

请正常关闭系统，让程序逐页关闭独立采集 Edge，避免下次启动出现“恢复页面”。

1. 如果正在采集 BOSS 岗位，先在网页点击“重新开始”。后端会先逐个关闭详情页和用户打开的岗位页，最后关闭保留的采集页。
2. 在 Streamlit 所在的 PowerShell 窗口按 `Ctrl+C`。
3. 在 FastAPI 所在的 PowerShell 窗口按 `Ctrl+C`。FastAPI 的退出钩子也会关闭 BOSS MCP 和项目拥有的全部 Edge 页面，因此即使没有先点“重新开始”，这里仍会执行最终清理。
4. 不再使用数据库时执行：

```powershell
docker compose stop postgres
```

下次运行时先执行 `docker compose up -d postgres`，再按上文命令启动 FastAPI 和 Streamlit。`docker compose stop` 不会删除数据；`docker compose down` 会移除容器和网络但保留命名卷；**除非明确要清空全部 PostgreSQL 数据，否则不要执行 `docker compose down -v`**。

如果终端曾被强制结束，可重新启动一次后端，在网页点击“重新开始”，再按上述顺序正常关闭。系统只关闭 `data/edge-profile` 对应的项目独立 Edge，不会关闭用户日常使用的 Edge。

## PostgreSQL 在系统中的作用

PostgreSQL 是持久化工作流数据库，不负责运行模型，也不直接保存上传文件。它用于保存：

- 任务、`thread_id`、LangGraph checkpoint 版本、当前节点、租约和恢复状态；
- 岗位/JD 与简历结构、模型 Profile 元数据、项目候选版本、反馈、用户确认和模块决策；
- 字段级补丁与审计记录、旧值哈希、快照引用和导出元数据。

上传简历、OCR 图片、生成文件、已登录的 Edge Profile 以及加密后的 API Key 材料保留在本机私有 `data/` 目录，不进入 Git。API Key 不写入 PostgreSQL：数据库只保存不含密钥的模型配置和凭据引用，加密密钥仍只存在当前电脑。停止 PostgreSQL 容器不会丢失数据，只有显式删除命名卷或数据库本身时才会清空。

## 环境配置文件

仓库只提交 `.env.example`，本机首次运行时执行 `Copy-Item .env.example .env`，然后修改 `.env`。常用项如下：

| 配置项 | 示例/默认值 | 用途 |
| --- | --- | --- |
| `POSTGRES_DB`、`POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_PORT` | 本地开发值 | PostgreSQL 容器配置，修改后需同步更新数据库 URL |
| `RESUME_AGENT_DATABASE_URL` | `postgresql+psycopg://resume:resume@127.0.0.1:5433/resume_agent` | PostgreSQL 连接地址，需与 Docker 配置一致 |
| `RESUME_AGENT_DATA_ROOT` | `./data` | 私有运行数据、上传文件、Edge Profile 和加密凭据目录 |
| `RESUME_AGENT_API_HOST` | `127.0.0.1` | 后端监听地址；单机使用不要改为公网地址 |
| `RESUME_AGENT_API_PORT` | `8000` | FastAPI 端口 |
| `RESUME_AGENT_INTERNAL_TOKEN` | 留空 | 留空时自动生成到被忽略的 `data/.internal-token` |
| `RESUME_AGENT_EDGE_PATH` | 留空自动检测 | 必要时填写 `msedge.exe` 的绝对路径 |
| `RESUME_AGENT_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | 本地 Ollama 地址 |
| `RESUME_AGENT_MCP_AUTOSTART` | `true` | 由 FastAPI 自动管理 BOSS/GitHub MCP 子进程 |

不要把云端 API Key 或 GitHub Token 写入 `.env`。聊天/Embedding Key 请在网页环境页填写并保存在本机加密凭据库；GitHub Token 仅在需要模板检索时临时输入。

Compose 已将 PostgreSQL 仅绑定到 `127.0.0.1`，不会直接暴露到局域网。若不再是单机可信开发环境，请先修改示例数据库密码；密码包含特殊字符时，写入 `RESUME_AGENT_DATABASE_URL` 前需要进行 URL 编码。

## 模型配置

### 云端 OpenAI-compatible

在网页第一栏分别配置聊天模型和 Embedding 模型：Base URL、真实模型 ID、认证格式和 API Key，然后点击检测。

- Base URL 建议填写到 `/v1`，不要填写 Markdown 链接语法。
- 即使粘贴 `/chat/completions`，系统也会归一化到 API 根地址。
- 默认选择 `Bearer Key`；只有平台明确要求原始 Key 时才选择“直接 Key”。
- API Key 会使用本机密钥加密保存，后续调用自动恢复，页面只显示掩码。
- 第一次向外部模型发送简历或 JD 时需要确认，之后沿用设置。
- 聊天与 Embedding 模型可以使用不同平台和不同 Key。

### Ollama

```powershell
winget install --id Ollama.Ollama -e
ollama pull qwen2.5:7b
ollama pull bge-m3
ollama list
```

必要时运行 `ollama serve`。网页会读取 Ollama 实际安装的全部模型，并分别筛选聊天与 Embedding 能力；系统不会静默下载模型。

## 截图 OCR

首次使用前下载 PP-OCRv6 medium：

```powershell
uv run python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv6_medium_det', text_recognition_model_name='PP-OCRv6_medium_rec', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device='cpu', enable_mkldnn=False); print('PP-OCRv6 medium ready')"
```

模型通常缓存在 `%USERPROFILE%\.paddlex\official_models`。支持粘贴或上传 PNG、JPG/JPEG、WEBP、BMP，单张最大 10 MB。模型增强失败时会保留 OCR 校正文字和规则解析结果，无需重新上传。

## BOSS 采集说明

- 使用 `data/edge-profile` 下的独立应用 Profile，不访问用户日常 Edge Profile。
- 登录、验证码、滑块和风控必须由用户手动完成。
- 登录后采集窗口保持最小化；详情解析创建后台 target，不抢占网页焦点。
- 岗位每批读取 10 条；下一页会恢复原采集标签、定位最后卡片、发送可信滚轮事件、跳过“对搜索是否满意”等非岗位节点，并自动重试延迟加载。
- “在采集 Edge 查看”才会主动显示岗位；之后继续分页仍会回到原采集页。
- 点击“重新开始”会关闭项目拥有的采集 Edge，并清空当前向导。
- 正常关闭 FastAPI 时也会执行同样的清理：逐页关闭项目拥有的标签页，再结束记录的 Edge 进程。
- 系统不会绕过 BOSS 的任何验证，也不能承诺网站未来改版后继续兼容。

## 简历与模板约束

- 每个任务只允许一份主简历。
- PDF 必须包含可复制文本，最大 10 MB、最多 3 页。
- 上传简历和选中的模板都必须经过模型结构化拆解，并由用户可编辑确认。
- GitHub 模板限定为许可明确的中文简历文件；系统尝试提供至少 5 个高 Star/高相关候选，失败时回退内置模板。
- 模型合理推导的技术方案和职责均标记 `[待核实]`，正式写入前逐字段确认。
- 修改使用字段级补丁，应用前校验 checkpoint 版本和旧值哈希。
- 超过一页时优先压缩字体、行距和页边距；缩短正文必须再次征得用户同意。

## 导出

| 格式 | 预览 | 下载 |
| --- | --- | --- |
| Markdown | 可编辑/可复制文字 | `.md` |
| DOCX | 文字预览；本机有 Word/LibreOffice 时显示转换后的 PDF 视觉预览 | `.docx` |
| PDF | 网页内嵌页面预览 | `.pdf` |

所有文件通过任务范围内的鉴权接口下载，不接受任意本地文件路径。

## 隐私与上传 GitHub 前检查

先将仓库内的占位配置 `.env.example` 复制为本机配置 `.env`，再按实际环境填写；`.env` 会被忽略，`.env.example` 只保留安全示例且不应填写真实密钥。云端模型 API Key 在网页中配置，不要写入 `.env`。

`.gitignore` 已排除 `.env`、本机密钥、`data/`、数据库、`.run/` 运行日志、备份、Edge Profile、Cookie、上传简历、PDF/DOCX、截图、预览、导出文件、`.venv`、编辑器/智能体本地配置和缓存。

首次推送前仍请人工检查：

```powershell
git status --short
git diff --cached --check
git diff --cached
```

不要提交真实简历、API Key、GitHub Token、浏览器资料、Cookie、数据库或生成产物。云端模型 Key 通过网页输入并仅在本机加密保存；GitHub Token 只在当前进程内使用，不持久化。

## 测试

```powershell
uv run python -m pytest -q
uv run python -m compileall -q app tests
uv run python -m ruff check --select E9,F63,F7,F82 app tests
```

## 常见问题

### DOCX 已生成但没有 PDF 视觉预览

安装 Microsoft Word 或 LibreOffice 后重启后端。转换依赖缺失时仍可下载 Markdown/DOCX；PDF 不会返回伪文件，而会明确提示依赖缺失。

### BOSS 下一页暂时没有新岗位

保持项目采集 Edge 运行。网页会自动再执行一次后台懒加载，并在没有明确结束标记时保留游标。若 BOSS 要求登录或验证，请人工完成后重新提交搜索。

### GitHub 模板下载超时

检查 DNS、防火墙、系统代理以及 `github.com`、`raw.githubusercontent.com`。临时 GitHub Token 可降低限流概率，但不会保存；网络不可用时可使用内置模板。

## 目录结构

```text
app/
  core/       数据结构、持久化、模型网关、本机加密凭据
  mcp/        FastMCP 服务、Edge 适配器、MCP 客户端
  services/   解析、匹配、生成、工作流、模板与导出
  ui/         Streamlit 页面与剪贴板组件
alembic/      PostgreSQL 迁移
docs/         可执行技术设计文档
scripts/      初始化与冒烟测试
tests/        单元、API、持久化、MCP、工作流与导出测试
```

## 后续计划

- 实现预留的公司招聘官网适配器。
- 增加更多保留原排版的模板适配与导出渲染器。
- 仓库地址确定后加入 CI 与版本发布。
- 在保持本地隐私边界的前提下再考虑多机部署。

## 贡献

项目公开后欢迎 Issue 和 Pull Request。请保持本地优先原则，不要削弱人工确认、旧值哈希或 checkpoint 并发校验；工作流修改必须补测试，测试数据不得包含真实凭据或真实简历。

## 许可证

当前仓库尚未声明开源许可证。在添加 `LICENSE` 文件前，源码可查看，但不会自动授予复制、修改和再分发权利。正式宣布为开源项目前，请先选择并添加合适的许可证。
