# Resume Agent

Resume Agent 是一个本地运行的中文简历智能体。它可以通过可见的 Microsoft Edge 搜索 BOSS 直聘岗位、解析岗位职责与 JD，并根据岗位生成项目经历、修改已有简历或使用模板生成新简历。

当前技术栈：FastAPI、Streamlit、LangChain、LangGraph、标准 MCP `stdio`、PostgreSQL、Playwright、Microsoft Edge 和 PaddleOCR v6。

## 1. 运行环境

第一版主要面向 Windows 10/11。建议安装以下软件：

| 软件 | 是否必需 | 用途 |
| --- | --- | --- |
| uv 与 Python 3.12（64 位） | 是 | 创建 `.venv` 并运行 API、网页、智能体和 MCP |
| Git | 建议 | 下载和更新项目 |
| Microsoft Edge | 是 | 搜索和解析 BOSS 岗位 |
| Docker Desktop | 推荐 | 只用于运行 PostgreSQL |
| Microsoft Word 或 LibreOffice | 可选 | 检查 DOCX/PDF 排版和导出 |
| Ollama | 可选 | 使用本地聊天模型和 embedding 模型 |
| PaddleOCR v6 medium | 截图功能必需 | 在本机识别粘贴或上传的岗位截图 |

可以在 PowerShell 中使用 `winget` 安装基础软件：

```powershell
winget install --id=astral-sh.uv -e
winget install --id Git.Git -e
winget install --id Microsoft.Edge -e
winget install --id Docker.DockerDesktop -e
```

安装完成后重新打开 PowerShell，检查命令是否可用：

```powershell
uv --version
git --version
docker --version
docker compose version
```

项目固定使用根目录下的 `.venv`，推荐 Python 3.12。`uv` 会负责安装 Python 和管理该环境，不依赖 Microsoft Store 的 Python 占位程序。

## 2. 下载项目

### 方法一：使用 Git

将下面的 `<项目仓库地址>` 替换为实际 GitHub/Git 仓库地址：

```powershell
git clone <项目仓库地址> resume-agent
Set-Location resume-agent
```

仓库地址通常类似：

```text
https://github.com/用户名/resume-agent.git
```

### 方法二：下载 ZIP

在 GitHub 项目页面点击 `Code` → `Download ZIP`，解压后进入项目文件夹。在文件夹空白处按住 Shift 并点击鼠标右键，选择“在终端中打开”。

后续所有命令都必须在包含 `pyproject.toml`、`docker-compose.yml` 和 `app` 文件夹的项目根目录执行。

## 3. 创建项目 Python 环境

在项目根目录执行：

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
```

后续命令统一使用 `uv run`，无需手动激活虚拟环境。若希望激活后直接使用 `python`，可以执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. 安装项目和环境包

### 推荐：安装完整运行环境

```powershell
uv sync --extra ui --extra mcp --extra documents --extra ocr
```

这个命令会安装：

- FastAPI、Uvicorn、Pydantic、SQLAlchemy 和 PostgreSQL 驱动；
- LangChain、LangGraph；
- Streamlit 网页界面；
- 标准 MCP SDK 和 Playwright；
- DOCX、文本型 PDF 解析与导出依赖；
- PaddleOCR、PaddlePaddle 和 PP-OCRv6 medium 岗位截图识别依赖。

如果还需要运行测试和代码检查，再安装开发依赖：

```powershell
uv sync --extra ui --extra mcp --extra documents --extra ocr --extra dev
```

检查关键依赖：

```powershell
uv run python -c "import fastapi, streamlit, langgraph, playwright, paddleocr, paddle; print('Python dependencies OK')"
```

首次使用岗位截图前，下载一次 PP-OCRv6 medium 的检测和识别模型：

```powershell
uv run python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv6_medium_det', text_recognition_model_name='PP-OCRv6_medium_rec', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device='cpu', enable_mkldnn=False); print('PP-OCRv6 medium ready')"
```

模型默认缓存在 `%USERPROFILE%\.paddlex\official_models`。应用只读取已下载的模型，不会在点击识别时静默联网下载。

本项目通过已安装的 Microsoft Edge 工作，不要求使用 Playwright 下载的 Chromium。Edge 路径无法自动识别时，可在网页的“环境与模型”中填写 `msedge.exe` 的完整路径并重新检测。常见路径为：

```text
C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
C:\Program Files\Microsoft\Edge\Application\msedge.exe
```

## 5. 配置 PostgreSQL

项目采用“PostgreSQL 在 Docker 中运行，其余程序在本机运行”的方式。

### 5.1 创建本地配置

```powershell
Copy-Item .env.example .env
```

默认数据库配置如下：

```text
数据库地址：127.0.0.1:5433
数据库名：resume_agent
用户名：resume
密码：resume
```

这些是本地开发默认值。需要修改时，请同时修改 `.env` 和 `docker-compose.yml`。

### 5.2 启动 PostgreSQL

请先启动 Docker Desktop，等待其显示 Docker Engine 已运行，然后执行：

```powershell
docker compose up -d postgres
docker compose ps
```

当 `resume-agent-postgres` 显示 `healthy` 后，初始化数据库：

```powershell
uv run python -m alembic upgrade head
```

只想临时体验且不安装 Docker 时，不要复制 `.env.example`，或删除 `.env` 中的 `RESUME_AGENT_DATABASE_URL`。应用会改用项目 `data/resume_agent.db` 中的 SQLite；然后执行：

```powershell
uv run python scripts/init_db.py
```

SQLite 适合试用和测试，长期使用建议采用 PostgreSQL。

## 6. 配置聊天模型与 Embedding 模型

岗位职责、任职要求和技能会先经过规则解析，再由所选聊天模型做结构化增强；模型输出必须能在 JD 原文中逐字验证，否则会被丢弃。模型不可用时系统会明确提示并保留规则解析结果，不会让岗位详情丢失。

### 6.1 云端 OpenAI-compatible 聊天模型

打开网页第一栏“环境与模型” → “云端聊天模型（OpenAI 兼容）”，依次填写：

| 配置项 | 示例 | 是否持久化 |
| --- | --- | --- |
| Base URL | `https://api.openai.com/v1` 或服务商给出的兼容地址 | 是 |
| 聊天模型名称 | 服务商实际支持的模型 ID | 是 |
| API Key | 服务商控制台创建的密钥 | 否，仅在后端进程内存暂存 |
| 外部模型数据发送确认 | 首次使用时勾选 | 是，可在设置中撤销 |

点击“保存并检测连接”。检测成功后，该配置会成为默认聊天模型，并可用于 JD 解析和项目经历生成。Base URL 与模型名称属于非敏感配置，会保存到 PostgreSQL/本地数据库；API Key 不写入 `.env`、数据库、日志或备份，后端重启后需要重新填写。不要把 API Key 填入 `.env.example` 或提交到 Git。

### 6.2 Ollama 本地模型

不使用本地模型时可以跳过本节，直接使用上面的云端 OpenAI 兼容接口。

安装 Ollama：

```powershell
winget install --id Ollama.Ollama -e
```

重新打开 PowerShell，下载默认聊天模型和 embedding 模型：

```powershell
ollama pull qwen2.5:7b
ollama pull bge-m3
ollama list
```

如果 Ollama 没有自动运行，可以在单独的终端执行：

```powershell
ollama serve
```

本地聊天模型与 embedding 模型分别检测。网页只会列出本机真正下载的模型，不会自动下载模型。

网页会分别检测本地聊天模型和 Embedding 模型。点击“扫描 Ollama 已下载模型”只读取本机已有模型，不会自动下载；选择模型后还需点击“检测”，状态为“正常”后才会用于智能体流程。

## 7. 启动项目

FastAPI 和 Streamlit 需要分别在两个 PowerShell 窗口中运行。两个窗口都要进入项目目录；`uv run` 会自动使用项目中的同一个 `.venv`。

### 终端一：启动 FastAPI 后端

```powershell
Set-Location <项目目录>\resume-agent
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

看到下面的信息表示后端已启动：

```text
Uvicorn running on http://127.0.0.1:8000
```

FastAPI 会自动启动和管理 BOSS MCP、GitHub MCP 子进程，不要再手动启动两个 MCP 服务。

### 终端二：启动 Streamlit 网页

```powershell
Set-Location <项目目录>\resume-agent
uv run python -m streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8666
```

浏览器访问：

```text
http://127.0.0.1:8666
```

后端健康检查和 API 文档地址：

```text
http://127.0.0.1:8000/api/health
http://127.0.0.1:8000/docs
```

项目只监听本机回环地址 `127.0.0.1`，默认不会开放给局域网或互联网。

## 8. 首次使用顺序

1. 打开 `http://127.0.0.1:8666`。
2. 在“环境与模型”中点击 Edge 检测和 Ollama 扫描。
3. 选择并检测聊天模型；需要时再检测 embedding 模型。
4. 进入“简历项目向导”，先选择是否上传已有简历；不上传时再选择模板完整简历或纯文本项目。
5. 选择岗位截图、BOSS 岗位 URL 或“岗位名称 + 城市”获取 JD。截图可以在粘贴框聚焦后按 `Ctrl+V` 直接粘贴，也可选择本地图片上传。
6. 使用自动搜索时，系统会打开独立的可见 Edge；首次使用需在该窗口手动登录 BOSS。
7. 登录完成后回到网页，再次点击搜索。搜索和翻页期间不要关闭岗位采集 Edge。
8. 使用“上一页/下一页”浏览岗位。若 BOSS 暂未返回新的懒加载卡片，“下一页”会保持可点，可再次触发下拉加载；只有页面明确提示到底时才禁用。
9. 列表展示 HR 的“在线/活跃时间待解析”状态；在采集 Edge 打开岗位或解析详情后，会读取详情页异步展示的“刚刚活跃、本周活跃”等状态并回写列表。
10. 截图 OCR 完成后先校正识别文字并确认；确认前不会创建岗位或启动项目生成。
11. 确认 JD、生成项目候选并逐项核实模型推导内容。
12. 已上传简历时选择“仅新增/替换项目”或“生成完整简历”；模板模式生成完整简历，纯文本模式直接输出可复制项目。

系统不会绕过验证码、滑块或登录限制。如果 BOSS 出现人工验证，请在保留的 Edge 窗口完成后重新点击相应操作。

## 9. 检查服务状态

在新的 PowerShell 中执行：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-WebRequest http://127.0.0.1:8666/_stcore/health -UseBasicParsing
docker compose ps
```

第二条命令返回 `ok` 表示 Streamlit 正常。

## 10. 停止服务

在 FastAPI 和 Streamlit 所在的两个终端中分别按 `Ctrl+C`。

停止 PostgreSQL 容器：

```powershell
docker compose stop postgres
```

再次使用时执行：

```powershell
docker compose up -d postgres
```

`docker compose down` 会删除容器但保留命名数据卷。不要执行 `docker compose down -v`，除非确定需要删除全部 PostgreSQL 数据。

## 11. 更新项目

使用 Git 下载的项目可以这样更新：

```powershell
git pull
uv sync --extra ui --extra mcp --extra documents --extra ocr
uv run python -m alembic upgrade head
```

更新后重新启动 FastAPI 和 Streamlit。

## 12. 测试

安装 `dev` 依赖后执行：

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run python -m compileall -q app
uv run python -m unittest discover -s tests
uv run python scripts/smoke_test.py
```

## 13. 常见问题

### PowerShell 不允许激活虚拟环境

仅为当前终端临时放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### PostgreSQL 无法连接

```powershell
docker compose ps
docker compose logs postgres
Test-NetConnection 127.0.0.1 -Port 5433
```

确认 Docker Desktop 已启动，容器状态为 `healthy`，且 `.env` 中端口为 `5433`。

### 8000 或 8666 端口被占用

```powershell
Get-NetTCPConnection -State Listen -LocalPort 8000,8666
```

如果后端改用其他端口，例如 `8010`，启动网页前需要同时指定后端地址：

```powershell
$env:RESUME_AGENT_API_URL="http://127.0.0.1:8010"
uv run python -m streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8666
```

### Edge 检测失败

先确认 Edge 已安装，再在“环境与模型”中填写 `msedge.exe` 完整路径并保存、重新检测。BOSS 操作必须使用应用创建的岗位采集 Edge，不要使用系统默认浏览器替代。

### Ollama 显示没有模型

```powershell
ollama list
ollama pull qwen2.5:7b
ollama pull bge-m3
```

下载完成后回到网页重新扫描。

### 岗位截图识别失败

确认已经使用 `--extra ocr` 安装依赖，并执行第 4 节中的模型下载命令。支持 PNG、JPG/JPEG、WEBP、BMP，单张不超过 10 MB。Windows 上使用 PaddlePaddle 3.3 时应用会关闭 oneDNN，避免 PP-OCRv6 medium 的 PIR 属性转换错误。若提示模型文件无权限，请确认当前 Windows 用户对 `%USERPROFILE%\.paddlex\official_models\PP-OCRv6_medium_det` 和 `PP-OCRv6_medium_rec` 具有“读取和执行”权限。

## 14. 数据与隐私

- API Key 和 GitHub Token 不写入配置、数据库、日志或备份。
- BOSS 使用独立 Edge profile；Windows 默认保存在 `%LOCALAPPDATA%\ResumeAgent\edge-profile`，避免 Desktop 目录权限导致 Edge 渲染/调试进程退出。登录态只保留在本机。
- 简历上传限制为 10 MB、最多 3 页；PDF 必须包含可复制文本，当前不做 OCR。
- 岗位截图只在内存中交给本地 PaddleOCR；图片不写入磁盘。OCR 文字暂存 30 分钟，只有用户校正确认后才创建岗位记录。
- 模型根据 JD 推导的经历默认标记为待核实，用户确认前不会正式写入简历。
- 修改简历使用字段级补丁和旧值哈希校验，避免覆盖用户手动修改的内容。
- 默认日志保留 30 天，应用仅供本地单机使用。

### 14.1 发布到 GitHub 前的隐私检查

仓库的 `.gitignore` 已排除以下高风险本地内容：

- `.env`、私钥和 Streamlit secrets；
- `data/` 下的数据库、内部令牌、岗位/任务记录、上传简历、偏好 Skill、备份与缓存模板；
- BOSS 专用 Edge profile、Cookie、登录状态和 CDP 文件；
- `photos/`、`output/`、导出文件、预览文件、DOCX/PDF 和本地简历 Markdown；
- 日志、Python 缓存、虚拟环境和编辑器配置。

`.env.example` 是可以提交的配置模板，里面只能保留占位值或本地开发默认值，不能填写真实 API Key、GitHub Token、个人数据库密码或个人路径。克隆项目后，每位用户应自行执行：

```powershell
Copy-Item .env.example .env
```

然后按实际环境修改 `.env` 中的数据库地址、数据目录、端口、Ollama 地址、默认本地模型和 Edge 路径（可选）。云端模型的 Base URL、模型名和 API Key 请在网页第一栏填写，其中 API Key 不应放进 `.env`。

首次公开仓库前执行以下只读检查：

```powershell
git status --short
git ls-files .env data photos output
git grep -n -I -E "(api[_-]?key|github[_-]?token|authorization|password|BEGIN .*PRIVATE KEY)"
```

如果 `.env`、`data/`、浏览器 profile、简历或截图在完善 `.gitignore` 之前已经被 Git 跟踪，仅增加忽略规则不会自动移除历史索引。确认目标无误后，可只从 Git 索引移除而保留本地文件：

```powershell
git rm --cached .env
git rm -r --cached data photos output
```

如敏感内容已经推送到远程仓库，应立即吊销并更换相关密钥；仅删除最新提交中的文件不能清除 Git 历史。数据库默认密码只适合监听本机的开发环境，若修改端口映射或对外开放 PostgreSQL，必须同步更换 `docker-compose.yml` 与 `.env` 中的凭据。
