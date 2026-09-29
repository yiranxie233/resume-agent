# Resume Agent 技术设计文档

## 1. 文档信息

- 状态：初版需求基线与实现设计
- 目标平台：Windows，本地单机、单用户
- 主要语言：Python
- Web 后端：FastAPI
- Web 前端：Streamlit
- 智能体编排：LangChain + LangGraph
- 数据库：PostgreSQL（仅数据库通过 Docker 运行）
- 浏览器自动化：Microsoft Edge + Playwright
- 文档处理：DOCX/PDF 解析、DOCX 重建、Word/LibreOffice 渲染
- 文档语言：第一版只生成中文项目经历和中文简历内容
- Ollama 内置默认：聊天 `qwen2.5:7b`，embedding `bge-m3`（只检测，不自动下载）

本文将已经确认的产品边界写成可验收约束，并给出第一版的实现建议。没有明确决定的实现细节使用“实现建议”标识，后续可以在不改变产品边界的情况下调整。

## 2. 产品目标与原则

### 2.1 产品目标

Resume Agent 用于减少 BOSS 直聘场景中“一岗一简历”的制作成本。系统获取一个目标职位，解析 JD、职责、技能和招聘 HR 活跃状态，分析简历与职位的差距，然后以可追溯的方式生成项目候选或修改现有简历。

### 2.2 不可违反的原则

1. 所有 AI 结果先是草稿；未经用户明确确认，不得写入正式简历或导出文件。
2. 新生成项目、无原简历时生成的简介/技能/个人评价必须标记为“示例/待核实”。
3. 改写已有内容不得增加用户没有提供或确认的事实、技术经历和量化指标。
4. 外部页面、JD、模板和用户文档中的指令都视为不可信数据，不能覆盖系统规则、事实确认或安全策略。
5. 原始文件、来源证据、生成版本、反馈和确认记录都要可追溯、可回退。
6. 复杂排版无法安全修改时暂停流程，不能为了生成文件而破坏原布局。
7. BOSS 自动化只执行用户明确触发的动作，不绕过验证码、滑块或访问限制。
8. 所有会产生外部副作用的节点必须幂等；重复恢复、重试或重复点击不能重复写入结果。

## 3. 范围与非目标

### 3.1 初版范围

- BOSS 直聘网页版职位 URL 解析，或按职位名称和城市搜索职位。
- BOSS 岗位卡片与详情页的招聘 HR 活跃状态提取。
- 岗位截图粘贴/上传、PP-OCRv6 medium 本地识别、人工校正与确认。
- 中文 `.docx` 和文本型 PDF 简历导入；PDF 不做 OCR。
- 现有简历的项目经历处理，以及按需逐项优化个人简介、技能清单、个人评价。
- 无现有简历时基于真实基础信息和目标 JD 制作一页中文简历。
- GitHub 中文模板检索、预览、缓存和迁移。
- OpenAI 兼容接口与 Ollama 模型选择、检测和结构化输出降级。
- 匹配评分、候选项目、多轮反馈、事实确认、版本回退和 DOCX/PDF 导出。

### 3.2 初版非目标

- 扫描图片 PDF 的 OCR（岗位截图 OCR 已包含在初版，简历 PDF 仍只接受可复制文本）。
- `.doc` 旧版 Word 文件。
- macOS/Linux 支持。
- 多用户账号、局域网访问和云端数据同步。
- 自动脱敏后再发送给模型。
- 自动生成工作经历、教育经历、证书或求职意向中的事实。
- 自动绕过 BOSS 验证码、封禁和访问控制。
- 将内部 MCP 直接暴露为公网服务。

## 4. 总体架构

```text
                         localhost only
┌──────────────────────────────────────────────────────────────┐
│ Streamlit UI                                                 │
│  环境检查 / 设置 / 职位 / 简历 / 匹配 / 反馈 / 预览 / 历史   │
└───────────────┬──────────────────────────────────────────────┘
                │ REST commands + SSE progress
┌───────────────▼──────────────────────────────────────────────┐
│ FastAPI application                                          │
│  API / task queue / auth token / OCR staging / event stream   │
│  LangGraph runtime + checkpoint persistence                  │
└──────┬───────────────┬───────────────┬───────────────────────┘
       │               │               │
       │ stdio MCP    │ local IPC     │ SQL
┌──────▼──────┐  ┌─────▼────────┐  ┌───▼──────────────────────┐
│ BOSS MCP    │  │ Edge helper  │  │ PostgreSQL in Docker      │
│ Python      │  │ Playwright   │  │ tasks, versions, events   │
└──────┬──────┘  └─────┬────────┘  └───────────────────────────┘
       │               │
       └────── visible Microsoft Edge with isolated profile

┌──────────────────────┐       ┌──────────────────────────────┐
│ GitHub Template MCP  │       │ Model Gateway                 │
│ Python / stdio       │       │ OpenAI-compatible / Ollama    │
└──────────────────────┘       └──────────────────────────────┘

Local tools: python-docx/PyMuPDF, PP-OCRv6 medium, Word first,
LibreOffice fallback, in-memory credential scope (no key/token persistence).
```

### 4.1 进程边界

除 PostgreSQL 外，组件都在宿主机运行：

- FastAPI：业务 API、任务队列、图执行、文件和数据库协调。
- Streamlit：轻量 UI，只通过 FastAPI 调用业务，不直接写数据库。
- Edge helper：独立进程，启动可见 Edge，使用独立 profile；只监听 `127.0.0.1`。
- BOSS MCP、GitHub MCP：标准 MCP `stdio` 服务，由主应用管理生命周期。
- Word/LibreOffice：宿主机安装的渲染和转换引擎。
- Ollama：宿主机可选服务，启动时探测，不由应用自动安装。
- PostgreSQL：固定版本官方镜像、Docker volume、本地配置初始化。

任务执行使用 FastAPI 的异步后台任务和本地 FIFO 队列；第一版不引入 Redis、Celery 或其他外部队列服务。

每个运行任务必须先获取 PostgreSQL worker lease（唯一 `worker_id`、`task_id`、获取时间、过期时间和最近心跳）；只有持有未过期 lease 的进程可以执行或恢复该任务的 LangGraph 节点。节点运行期间定期续租，进程退出、心跳超时或 lease 被撤销时停止后续副作用。启动扫描过期 lease，将对应任务标记为可恢复；新进程重新取得 lease 后才允许继续。

### 4.2 网络边界

- Streamlit、FastAPI、Edge helper 默认绑定 `127.0.0.1`。
- Streamlit→FastAPI、FastAPI→Edge helper 使用本次启动生成的随机内部令牌。
- 外部 OpenAI 兼容服务必须使用 HTTPS；本机 Ollama/本地代理允许 HTTP。
- `file://` API 地址拒绝；其他 URL 限制第一版暂不额外收紧。
- 启动时端口冲突自动选择可用端口，并在环境页面显示实际地址。

## 5. 运行环境与启动

### 5.1 必需与可选依赖

必需：Windows、Python、Docker Desktop、Microsoft Edge、Docker 可运行的 PostgreSQL。

按功能可选：Microsoft Word 或 LibreOffice、Ollama、GitHub Token、外部 OpenAI 兼容接口。岗位截图功能需要 PaddleOCR 3.7、PaddlePaddle 3.3、Pillow 和已经下载到本机的 `PP-OCRv6_medium_det`/`PP-OCRv6_medium_rec`。

### 5.2 启动流程

1. 启动脚本检查 Python、Docker、Edge、数据目录、PostgreSQL 配置。
2. 若 PostgreSQL 容器不存在，拉取固定版本官方镜像，创建 volume 并初始化数据库；若已存在则启动并健康检查。端口、数据库名、用户名和密码使用本地配置文件中的默认值，并允许用户修改。
3. 启动 FastAPI，生成本次运行的内部令牌和任务恢复上下文。
4. 启动 Edge helper 和两个 stdio MCP 子进程，注册工具并执行健康检查。
5. 启动 Streamlit，打开环境检查页面。
6. 先扫描并处理遗留 `pending` 节点操作；无法判定的任务冻结并显示恢复冲突。
7. 仅恢复 PostgreSQL 中状态为 `running`、`waiting_user` 或 `paused` 且没有未处理操作的任务，标记需要用户选择继续或重试。恢复外部模型任务前检查内存凭据和模型探测状态；缺少凭据时保持 `status=paused, blocked_reason=needs_credentials`，本地模型被卸载或能力探测过期时保持 `status=paused, blocked_reason=needs_model_recheck`，不能静默换模型。

Docker 未安装、数据库无法连接或 Edge 环境缺失时，只阻止依赖它们的功能，并在环境页给出修复动作。Word/LibreOffice、Ollama、GitHub Token 检查失败可以跳过。

### 5.3 环境检查页面

每项显示状态、版本/路径、最近检测时间、错误原因和独立“重新检测”按钮：

- PostgreSQL/Docker
- Edge 路径、独立 profile、登录状态
- Word/LibreOffice 与默认渲染引擎
- Ollama 服务，以及分别按聊天角色/embedding 角色列出的已下载模型、模型 tag/digest 和探测状态
- OpenAI 兼容模型配置
- GitHub Token 和模板缓存
- 数据目录读写权限

Edge 支持自动检测 `msedge.exe` 路径，检测失败时允许手动指定；页面提供“打开 Edge 并登录 BOSS”“检测登录状态”“退出并清除会话”按钮。第一版只支持 BOSS 网页版，不支持移动端 URL。独立 profile 放在可配置数据目录下，Edge 窗口保持可见。

## 6. 标准 MCP 设计

两个 MCP 必须使用 Python MCP SDK（实现时固定依赖版本）和 JSON-RPC `stdio` 传输，不能只做普通 Python 函数。FastAPI 作为 MCP host/client，负责启动、关闭、健康检查和超时。初版只供本项目本地调用，不对外开放端口。

MCP 和 Edge helper 的工具结果统一使用 Pydantic envelope：

```json
{
  "ok": true,
  "data": {},
  "error_code": null,
  "retryable": false,
  "requires_user": false
}
```

失败时 `error_code` 必须来自版本化错误枚举；`retryable=true` 才允许节点按配置自动重试，`requires_user=true` 必须进入人工中断。网络超时、登录失效、验证码、解析失败和许可证风险使用不同错误码，不能用一条通用异常掩盖恢复路径。

### 6.1 BOSS MCP 工具

建议工具名和职责如下：

| 工具 | 输入 | 输出/副作用 |
|---|---|---|
| `boss_validate_url` | URL | 是否为允许的 `zhipin.com` 官方网页版 URL |
| `boss_check_login` | 无 | Edge 登录状态和需要人工处理的原因 |
| `boss_search_jobs` | 职位、城市、可选筛选、分页游标 | 最多 20 条职位摘要、相关性排序、卡片 HR 在线状态、`load_pending/exhausted` |
| `boss_get_job_detail` | 职位链接/内部标识 | JD 字段、精确 HR 活跃状态、来源 URL、证据段落和快照引用 |
| `boss_snapshot_job` | 当前详情页 | 去 Cookie/凭据后的 HTML 快照路径和哈希 |
| `boss_reparse_job` | 任务/职位标识 | 用户明确触发的重新解析结果 |
| `boss_logout` | 无 | 退出并清除独立 Edge 会话 |

搜索只取得摘要；用户选择一个结果后才进入详情页。搜索必填职位名称和城市，可选工作经验、学历、薪资范围、公司规模、融资阶段、行业。结果按与输入的相关性排序，首批 20 条，支持继续加载。

### 6.2 BOSS 访问、分页和 HR 活跃状态边界

- 首次启用前展示可能导致账号封禁的风险，单独要求用户确认。
- 只执行用户触发的搜索/详情/重新解析，使用保守请求间隔，不绕过验证码或访问限制。
- 登录失效、验证码、滑块和访问限制出现时暂停，用户在可见 Edge 中手动处理后再继续。
- 解析失败不自动循环重试；只在用户点击“重新解析”后再试。
- 不再依赖 `boss show time` 推断 BOSS 未公开展示的岗位发布时间，也不把未知发布时间作为新鲜度依据。
- 列表卡片出现 `.boss-online-icon` 时记录“在线”；其他卡片记录“活跃时间待解析”。只有用户选择岗位并打开详情后，才读取 `.boss-active-time` 得到“刚刚活跃、今日活跃、几天前活跃、本周活跃”等精确值，避免为每条摘要打开详情造成额外风控。
- 加载下一页时渐进滚动最后一张卡片和 document 底部，最长等待 15–20 秒并监测新卡片。一次未返回新卡片仅设置 `load_pending=true`、保留原 `next_cursor`；只有页面出现可见的明确结束标识才设置 `exhausted=true` 并清除游标。

### 6.3 职位字段与筛选

职位记录至少包含：岗位名称、公司、城市、薪资、岗位职责、任职要求、技能要求、招聘 HR 活跃状态、来源 URL、详情快照路径、抓取时间和解析证据。旧数据的 `posted_at/posted_at_label` 仅为兼容字段，新采集与界面筛选不依赖它们。

每个任务必须保存 `job_snapshot_id`，匹配、候选生成、内容优化和导出只读取该不可变快照及其 `evidence_id`，不能读取职位当前记录。

HR 活跃筛选默认不限，可选择“在线或刚刚活跃、今日活跃、近 3 天活跃、本周活跃、活跃时间待解析”。列表页无法取得精确状态的记录必须保留并标记“活跃时间待解析”，不得猜测活跃时间。

用户可从历史记录选择已解析 JD，避免重复访问 BOSS；历史 JD 提供“刷新”操作。刷新会更新职位当前记录，但不会改写任何既有任务绑定的不可变 `job_snapshot`；新任务使用刷新后的新快照，旧任务继续使用创建时的快照和证据。

### 6.4 岗位截图 OCR

- UI 同时提供剪贴板图片粘贴组件和本地图片上传，接受 PNG、JPG/JPEG、WEBP、BMP，单张最大 10 MB、最大 4000 万像素。
- FastAPI `POST /api/jobs/screenshot/ocr` 只执行真实图片校验和本地 PP-OCRv6 medium 识别，返回 `ocr_id/text/lines[{text,score,box}]/model`；原图不落盘，确认前不得创建 `JobInput`。
- OCR 模型懒加载为进程单例并用线程锁保护；Windows CPU 显式 `enable_mkldnn=false`，规避 PaddlePaddle 3.3 的 PP-OCRv6 oneDNN/PIR 兼容错误。模型缺失或不可读时返回可修复错误，不静默下载或切换模型。
- UI 将 OCR 原文放入可编辑文本框。用户填写岗位名称并校正后调用 `POST /api/jobs/screenshot/confirm`；后端校验仍在 30 分钟有效期内的 `ocr_id` 后才解析 JD、创建职位和不可变快照。确认操作为单消费者，重复提交不能生成重复职位。
- 岗位截图 OCR 与简历 PDF 边界独立：扫描版简历 PDF 仍拒绝，不能借岗位截图接口绕过简历输入约束。

### 6.5 GitHub 模板 MCP 工具

建议工具名和职责如下：

| 工具 | 输入 | 输出/副作用 |
|---|---|---|
| `github_search_resume_templates` | 关键词、语言、文件类型、页码 | 中文模板候选，按 Star 降序，默认 5 条 |
| `github_get_template_license` | 仓库/文件 | 许可证、是否允许修改和本地使用 |
| `github_preview_template` | 仓库/文件 | 统一页面或图片预览 |
| `github_download_template` | 候选标识 | 本地缓存路径、版本和来源元数据 |
| `github_check_template_update` | 本地模板记录 | 远程版本差异，不自动覆盖 |

只展示中文 `.docx` 或 Markdown 模板，许可证必须明确允许修改和本地使用；许可证不明、禁止修改或要求在简历成品中附带署名/版权声明的模板直接排除。卡片展示仓库链接、许可证、Star 数、最后更新时间和来源。GitHub Token 可选，仅在当前 GitHub MCP 请求的内存中使用；无 Token 使用匿名请求。Token 不写入配置、数据库、日志或备份。

用户选中的模板复制到本地模板目录并保存版本信息。检测到远程更新时只提示用户选择下载或继续当前版本。GitHub 不可用时使用已缓存模板；没有缓存时使用内置一页中文模板。

## 7. LangGraph 智能体设计

### 7.1 主图与专用节点

使用一个主 LangGraph 图，不采用多个智能体自由对话。建议节点如下：

```text
任务初始化
  -> 环境/模型门禁
  -> 获取职位 (URL 或搜索选择)
  -> 解析 JD + 证据定位
  -> 导入/选择简历或基础信息
  -> 解析简历 + 结构化预览
  -> 模板选择/迁移（如需要）
  -> 用户确认布局/解析修正
  -> 匹配分析与评分
  -> 并行生成 N 个候选项目
  -> 候选校验/去重/展示
  -> 用户反馈、选择或重试 (循环)
  -> 项目替换/新增与排序
  -> 个人简介优化 (若触发，用户确认循环)
  -> 技能清单优化 (若触发，用户确认循环)
  -> 个人评价优化 (若触发，用户确认循环)
  -> 一页检查与压缩审批
  -> 创建不可变 resume_snapshot
  -> 页面预览/导出反馈循环
  -> 最终确认、导出、归档
```

除纯规则节点外，每个模型节点前都经过独立的 `context_builder`：按节点白名单选择来源事实、有效 `evidence_id`、用户反馈、确认内容和 Skill，按所选模型的实际上下文窗口计算硬 token 预算，并生成上下文清单哈希。裁剪优先级固定为“用户确认事实 > 当前命中证据 > 必要 JD 职责 > 当前反馈 > 个人 Skill > 历史参考”；核心事实或证据无法放入预算时阻止模型调用并进入人工处理，不得静默截断。节点不得默认接收完整原文、全部历史版本或其他任务数据。

上下文 token 估算策略已经冻结：优先使用与当前模型 profile 绑定且版本固定的原生 tokenizer；没有可识别 tokenizer 时使用 `conservative-char-v1` 保守估算器（按输入 UTF-8 字节数上界估算，再加结构化包装开销），并预留安全余量。计数前统一 Unicode NFC、换行符 LF，并对结构化对象使用稳定的规范 JSON 序列化（排序键、固定空白规则）；计数对象必须是实际将发送的渲染结果。模型的 `context_window_tokens` 必须来自 provider metadata、版本化 tokenizer registry 或用户明确配置；窗口未知时任务门禁阻止调用。预算按规范化后的 system/developer 指令、消息、工具 schema、工具结果和输出预留统一计算：`usable_input = context_window_tokens - max_output_tokens - safety_margin`。默认安全余量为 `max(512, ceil(context_window_tokens * 0.10))`，可在新任务设置中调整但必须版本化。证据原子项和结构化字段不得被截断；核心事实仍放不下时返回 `context_budget_blocked`，暂停节点并提示用户缩短内容或调整输出上限。未知 tokenizer 可以走保守 fallback，但窗口未知、输入非法或估算器异常都直接阻止调用，不得猜测或静默截断。

每个 `context_snapshot` 记录 `tokenizer_id`、`tokenizer_version`、`estimator_mode`（`exact`/`conservative_fallback`）、`budget_policy_version`、估算策略版本、模型上下文窗口、`max_output_tokens`、安全余量、可用输入预算、估算 token 数、最终纳入/舍弃的对象 ID、裁剪顺序和阻止原因。任务开始后这些参数随模型 profile version 固定；估算器或预算策略版本变化只影响新任务。`request_key` 同时包含渲染后上下文哈希、估算器版本和预算策略版本，避免旧响应跨策略复用。

### 7.2 显式中断点

以下动作必须是 LangGraph interrupt/checkpoint，而不是前端临时状态：

- 外部数据发送首次同意或重新同意。
- BOSS 登录、验证码、滑块和访问限制人工处理。
- 简历结构化结果修正、项目区域手动标记。
- 模板选择、迁移布局确认。
- 候选项目反馈、选择、事实逐字段确认。
- 是否替换项目、是否优化简介/技能/评价。
- 是否压缩一页、是否允许 AI 改写。
- 占位符填写或明确保留。
- 导出预览反馈与最终确认。

用户操作后从原节点恢复，不重新运行已经成功且无输入变化的前置节点。

每个中断点定义允许的结构化动作集合和输入 Schema，例如 `approve`、`reject`、`revise`、`select`、`skip`、`resume`、`cancel`。FastAPI 根据当前 `node_name`、图状态和动作 Schema 校验请求；不属于当前节点的动作、缺少必填字段或重复消费的动作直接拒绝，不能交给模型自行解释。自然语言反馈作为动作 payload 的内容输入，不作为流程控制命令。

同一任务的恢复请求必须携带客户端看到的 checkpoint 版本号。后端将其与数据库中当前最新版本做乐观并发校验；版本不匹配时返回可识别的冲突错误（例如 HTTP 409），不执行动作、不写入新 checkpoint，并要求前端刷新后重新提交。重复点击同一动作在首次成功后也按过期版本处理。

每个任务创建时生成一个唯一且稳定的 LangGraph `thread_id`，并将其持久化到 `tasks` 和 `langgraph_checkpoints`。旧任务重启、继续、用户反馈和“重试当前步骤”都沿用原 `thread_id`；用户开启新的简历任务时必须生成新的 `task_id` 和新的 `thread_id`，不能复用旧线程。`task_id` 用于业务目录和历史记录，`thread_id` 用于图状态恢复，两者必须单独存储并建立一对一关联。

### 7.3 状态分层

LangGraph 状态和 PostgreSQL 持久化状态必须分为四类：

1. `source_facts`：BOSS、原简历、用户基础信息和用户明确提供的事实。
2. `model_inferences`：模型推断、候选项目、语义评分、建议改写。
3. `user_confirmed`：用户逐项确认的事实、选中的版本和确认时间。
4. `exportable_content`：允许写入正式简历和导出的最终内容。

状态转换只能由明确的用户动作触发；未确认推断不能进入 `exportable_content`。

LangGraph checkpoint 只保存流程控制状态、当前节点、checkpoint 版本、等待动作和业务对象 ID，不嵌入完整简历/JD 正文、版本正文或二进制文件。PostgreSQL 业务表和任务目录文件是内容的唯一事实来源；恢复时通过 ID 加载内容，并校验引用对象仍属于同一 `task_id`/`thread_id`。产生业务写入的节点先在事务中提交业务记录和文件清单，再提交对应 checkpoint；任一部分失败都不得推进图状态。

### 7.4 节点契约

- 每个节点使用 Pydantic 输入/输出模型。
- 模型输出先做 JSON Schema/Pydantic 校验；失败按配置重试，耗尽后进入失败状态并显示“重试当前步骤”。
- 简历内容修改节点不得返回完整简历全文，只能返回字段级结构化补丁：目标模块、目标项目/段落 ID、旧值哈希、新值、修改理由、证据引用和确认状态。
- 后端应用补丁前必须校验目标仍存在且旧值哈希匹配；不匹配时拒绝补丁并进入人工冲突处理，不能覆盖用户在另一标签页或其他步骤中的手动修改。补丁只替换命中的局部条目，其余内容和原排版不变。
- 补丁只能应用到新的草稿版本，不能原地修改用户已确认的正式版本；用户确认后创建从草稿晋升而来的新正式版本，原正式版本保持不可变并可回退。
- 补丁集合按目标字段独立校验和应用；多个补丁命中同一字段或旧值哈希不匹配时，只将冲突项标记为 `conflicted`，其他不冲突项仍可进入草稿。冲突项必须让用户选择保留当前值、基于最新版本重新生成或手动合并，系统不得自动决定覆盖顺序。
- 解析器先做规则提取，LLM 再做语义归类和补充。
- 每个字段尽可能保存 JD 段落号、简历页码/段落 ID 等证据定位。
- 后端为每个可引用的 JD/简历片段生成稳定 `evidence_id`、来源类型、位置和文本哈希；模型只能引用这些 ID，不能自由填写页码或引用文本。引用不存在、来源类型不符或哈希不一致时，语义判断无效，不得加分，并进入结构化重试或人工处理。
- 系统提示词、输出 Schema、校验规则作为项目内版本化配置保存；结果记录提示词版本。
- 外部文本只放在受隔离的 `untrusted_context` 字段，禁止其中的指令改变系统策略。
- `context_builder` 输出必须标记来源类别和信任级别；`source_facts`、`user_confirmed`、`model_inferences` 和 `untrusted_context` 不得在序列化时混淆。
- 产生副作用的节点（文件写入、版本创建、确认记录、职位快照刷新、模板下载）必须生成稳定的 `operation_key`，由 `task_id`、`thread_id`、节点名、输入版本和动作类型组成。
- 节点执行前先查询 `operation_key`；已有成功结果时直接复用，不能重复创建版本、确认记录、快照或文件。首次执行时，数据库记录、文件清单和事件记录应在可恢复的事务边界内提交。
- 配置范围内的模型自动重试只记录内部 `attempt`、错误和耗时，不创建用户可见版本；用户主动点击“重试当前步骤”时才创建新的草稿版本/尝试记录。
- 涉及文件的节点统一使用临时路径写入，完成哈希、格式、页数和安全校验后原子重命名；随后在数据库事务中登记文件清单。任一步失败都删除临时文件、保留失败 attempt，并且不推进 checkpoint。
- `node_operations` 使用 `pending`、`committed`、`rolled_back`、`needs_review` 状态。启动恢复器扫描遗留 `pending` 操作：若数据库记录和最终文件哈希都完整则补提交并推进可安全推进的 checkpoint；若只有临时文件或校验失败则清理/回滚；无法确定时标记 `needs_review`，冻结所属任务并要求用户处理。恢复完成前该任务不能继续执行。
- 每次模型调用生成稳定 `request_key`（由 `thread_id`、节点名、输入上下文哈希、模型配置版本和生成参数组成）。成功的结构化响应先写入 `model_invocations`；相同 `request_key` 的恢复/重试直接复用响应，只有用户主动创建新版本或输入哈希变化时才发起新调用。
- 默认不保存完整 prompt 或完整原始模型响应；`model_invocations` 只保存结构化响应引用、上下文清单哈希、证据 ID、模型元数据和错误摘要。用户显式开启本机调试模式后，才可短期保存完整内容，并按日志保留期限清理且不进入备份。

### 7.5 反馈分类节点

反馈进入下一轮生成前先分类为：事实、风格偏好、结构约束、普通意见。先使用规则识别明确的事实/风格/结构触发词和字段模式，无法确定时才调用 LLM 兜底。分类器输出类别、置信度、命中的反馈字符区间、提取的事实候选和冲突列表；置信度低于 `0.85`，或一条反馈同时包含互相冲突的类别/要求时暂停并要求用户澄清。明确写入的新事实仍必须进入字段确认队列；“更突出”“更简洁”等风格意见不自动构成事实确认。

### 7.6 并行候选与恢复

候选项目默认 3 个，可调为 1–5 个。任务创建候选槽位时分配稳定的 `candidate_slot_id`（如 `slot-1`），并行生成只允许写入自己的槽位。某个槽位失败或用户主动重试时只重跑该槽位，不改变其他槽位。完成后统一做 Pydantic 校验、相似度去重和岗位侧重点检查；去重淘汰的槽位保持记录并标记原因，补生成只填入缺失槽位。最终排序按后端匹配分降序，再按固定 `candidate_slot_id` 做 tie-break；仍不足指定数量时将步骤标记为 `partial`，交给用户决定继续补生成或接受当前数量。

### 7.7 智能体算法执行性审查

本轮需求澄清后，以下关键算法和恢复决策已经冻结：稳定 `thread_id`、`request_key`、checkpoint 乐观锁、字段级补丁、不可变 `resume_snapshot`、worker lease、JD/job snapshot、任务级模型配置版本、候选槽位/去重、字段确认失效、上下文预算、反馈分类阈值、MCP envelope 和崩溃补偿状态机。它们可以直接拆成数据库迁移、纯函数测试和 LangGraph 节点实现。

因此，项目已经可以开始整体构建和第一条端到端垂直切片；尚未完成的内容主要是实现参数和适配器选择，不再是产品流程歧义：

- 具体的 Pydantic 字段 Schema、错误码枚举和 API 请求/响应示例；
- worker lease 的数值参数（租期、心跳间隔、接管宽限期）；
- 本地 TF-IDF 降级的阈值校准；
- 各模型 tokenizer adapter 的实现、`conservative-char-v1` 的测试 fixture 和具体上下文窗口探测；预算策略本身已冻结；
- DOCX/PDF 解析库和复杂布局的可测试支持矩阵；
- PP-OCRv6 medium 在更多真实 BOSS 截图尺寸、缩放和主题下的识别阈值校准。

这些参数必须在代码配置中有默认值、版本号和测试；变更不能静默改变旧任务的评分、版本或导出结果。

## 8. 模型网关

### 8.1 配置与选择

聊天模型和 embedding 模型是两个独立的配置类型，不能用一个模型名或一个能力探测结果代替另一个。每个 `model_profile` 必须固定 `role=chat|embedding` 和 `provider=ollama|openai_compatible`；配置一旦被任务引用，`role`、provider、模型名和端点不能原地修改，只能创建新的 profile version。

聊天模型配置至少包含：`profile_id`、`role=chat`、`provider`、`base_url`、模型名（含 tag）、只读的 `credential_required`、生成参数、结构化 JSON/JSON Schema 能力、tool-call 能力、流式能力、`context_window_tokens`、`tokenizer_id`、`tokenizer_version`、`tokenizer_source`、上下文窗口估计器版本、`status` 和最近探测结果。embedding 配置至少包含：`profile_id`、`role=embedding`、`provider`、`base_url`、模型名（含 tag）、只读的 `credential_required`、向量维度、最大输入长度（如果服务提供）、输入切分策略版本、是否归一化、批量大小、距离度量、`status` 和 embedding 算法版本。设置页必须显示两个独立的模型卡片和两个独立的“检测连接”按钮：

```yaml
chat_model:
  provider: ollama
  base_url: http://127.0.0.1:11434
  model_name: qwen2.5:7b  # 第一版内置默认，可在设置中替换

embedding_model:
  provider: ollama
  base_url: http://127.0.0.1:11434
  model_name: bge-m3  # 第一版内置默认，可在设置中替换
```

`qwen2.5:7b` 和 `bge-m3` 是第一版内置默认值，不作为强制下载项。用户可以分别指定默认聊天模型和默认 embedding 模型；聊天模型未就绪时阻止依赖模型的任务，不自动换用其他聊天模型。embedding 模型未就绪时允许使用本地 TF-IDF/词项相似度降级，但必须在任务、评分配置、候选状态和界面中明确标记 `embedding_mode=tfidf_fallback`。系统绝不把聊天模型当作 embedding 模型，也不把 embedding-only 模型当作聊天模型。

任务开始时复制并固定 `chat_model_profile_version` 和 `embedding_model_profile_version`，并记录实际解析到的模型 tag、模型 digest（若提供）、向量维度、能力探测结果、上下文估算器版本以及 `embedding_mode`（`embedding` 或 `tfidf_fallback`）。设置页之后的修改、删除或重新探测只影响新任务；反馈阶段临时切换聊天模型必须创建新的 `generation_branch_id`，并保留原分支供比较。

Ollama 检测必须区分 `executable_missing`（命令行程序不存在）、`service_unreachable`（服务不可达）、`model_not_installed`（模型未下载）、`capability_mismatch`（角色不匹配）、`probe_failed`（能力探测失败）和 `ready`（已就绪）六种状态。`executable_missing` 主要是环境提示：如果配置的 Ollama HTTP endpoint 仍可访问且角色探测通过，profile 可以标记 `ready` 并同时显示 CLI 缺失警告；只有服务或角色探测失败才阻止该 profile。

1. 优先使用 `PATH` 中的 `ollama`（必要时检查 Windows 默认安装路径）获取版本；找不到 CLI 时仍继续探测配置的 HTTP endpoint，不自动安装或下载。
2. 请求 Ollama `/api/version`、`/api/tags`，并对选中的模型请求 `/api/show` 获取 digest 和服务端声明的 capabilities；以已下载模型的精确 `name`/tag 建立本地清单。`/api/tags` 是“已下载”判定的权威来源，不扫描 Ollama blob 文件；如存在 `OLLAMA_MODELS`，只作为界面展示的存储路径。`OLLAMA_HOST` 只用于用户明确配置的本机/本地代理地址，规范化后写入探测结果。找不到用户填写的模型名时返回 `model_not_installed`，界面提示用户手动执行 `ollama pull <model_name>` 后再点“重新检测”。禁止模糊匹配或静默替换模型。
3. 对 `role=chat` 发送最小 `/api/chat`（或兼容 `/api/generate`）探测，验证返回文本和 JSON 解析路径；对 `role=embedding` 发送最小 `/api/embed`（兼容旧版 `/api/embeddings`）探测，验证向量非空、维度稳定且数值有限。`/api/show` 的 metadata 不能替代角色专属探测。探测结果保存规范化地址、时间、接口版本、模型 digest、能力矩阵和错误码。
4. 只有角色专属探测通过才标记 `ready`。服务可达但模型未下载、端点返回角色不支持或维度不一致时均不得设为默认模型。

保存 profile 不等于任务可用：任务创建/恢复时必须对默认聊天 profile 重新核验（必要时重新 probe），并在同一个数据库事务中固定两个 profile version 与 `embedding_mode`。若聊天 profile 不是 `ready`，任务进入模型门禁；若 embedding profile 不是 `ready`，只有用户明确允许且任务尚未生成向量时才能选择 `tfidf_fallback`。任务开始后不得改变模式。

外部 OpenAI 兼容接口也按 `chat` 与 `embedding` 分别探测 `/chat/completions` 和 `/embeddings`；两个 profile 可以使用同一 `base_url`，但必须有各自的模型名和能力结果。

### 8.2 连接与能力探测

“检测连接”必须实际发送最小测试请求，且探测请求不得把完整简历、JD、用户反馈或原始模型响应持久化。结果统一返回 `provider`、规范化 `base_url`、实际 endpoint、模型名/tag、模型 digest（若有）、`checked_at`、能力矩阵、向量维度（embedding）和可展示的错误码。

能力探测按角色执行，不能把不适用于 embedding 的能力当作通用门槛：

| 角色 | 必须验证 | 可选能力（按节点需要） |
|---|---|---|
| `chat` | generation、基本错误格式、最小文本返回；至少有严格 JSON 提示词 + Pydantic 兜底路径 | JSON Schema、tool call、streaming、上下文窗口 |
| `embedding` | embeddings endpoint、非空且有限的向量、稳定 dimension、重复输入结果稳定 | batch、最大输入长度、归一化/距离度量 |

探测接口返回结构至少满足以下字段，前端不需要解析 provider 的原始响应：

```json
{
  "role": "embedding",
  "provider": "ollama",
  "status": "ready",
  "base_url": "http://127.0.0.1:11434",
  "endpoint": "/api/embed",
  "model_name": "bge-m3",
  "model_digest": "sha256:...",
  "dimension": 1024,
  "capabilities": {"embedding": true, "batch": true},
  "checked_at": "2026-09-24T00:00:00Z",
  "error_code": null,
  "retryable": false
}
```

聊天 profile 的 `dimension` 为 `null`，并返回 `generation`、`structured_json`、`tool_call`、`streaming` 等能力；embedding profile 不要求这些聊天能力。示例中的 digest 和维度仅用于说明字段；默认模型名固定为 `qwen2.5:7b` 与 `bge-m3`，但用户可在设置中替换。

OpenAI 兼容接口的 `chat` 分别探测 `/chat/completions`，`embedding` 分别探测 `/embeddings`；`base_url` 是否已经包含 `/v1` 必须由 adapter 规范化，禁止拼出重复路径。Ollama 使用其原生 `/api/*` 路径，不发送 OpenAI 鉴权头。外部接口使用 `Authorization: Bearer <key>`，但 key 只能在本次请求内存中存在。

支持结构化输出时优先使用；不支持时使用严格 JSON 提示词、Pydantic 校验和有限重试。`tool call` 或 `streaming` 为可选能力时，探测失败不应阻止保存默认模型；角色必需能力失败则不能设为 `ready`。探测状态和错误码必须可被前端直接展示，不能只返回一条无法区分原因的字符串。

LangGraph 节点只依赖内部 `ChatModelPort` 和 `EmbeddingModelPort`，不直接读取 Ollama 或 OpenAI 的原始响应。provider adapter 负责地址规范化、鉴权、超时/重试、响应解析、能力检查和统一错误码；adapter 输出必须携带 profile version、request key、模型 tag/digest（若有）和向量维度，避免在图状态中混入 provider 特有字段。

### 8.3 生成参数与超时

聊天模型设置允许调整 `temperature`、`top_p`、最大输出长度、请求超时、自动重试次数和重试间隔；embedding 设置允许调整批量大小、是否归一化、距离度量、请求超时和重试次数。每个参数按 provider 能力校验，不能把聊天生成参数发送给 embedding endpoint。上下文预算使用任务固定的 tokenizer/保守估算器版本；无法识别 tokenizer 时使用保守字符/token 估算并预留安全余量，估算失败则阻止调用并提示用户。embedding 输入长度独立受其 profile 的 `max_input_tokens` 约束，超长文本只能按固定 `chunk_policy_version` 在稳定边界切分；最大输入长度未知时阻止生成向量或由用户明确选择 `tfidf_fallback`，不得套用聊天模型窗口。单个任务最长运行时间默认 5 分钟，用户可调整；超时后暂停并保留中间结果。

### 8.4 编排框架选型

**当前建议：LangGraph 作为主编排层，LangChain 作为模型/提示词/结构化输出适配层；不在核心流程中同时引入 OpenAI Agents SDK。**

OpenAI 官方 Agents 指南将 Agents SDK 定位为运行在应用内、使用可复用 agents、tools 和 handoffs 的应用层运行时，并提供 sessions、guardrails、results/state 和 tracing 等能力：<https://platform.openai.com/docs/guides/agents>。

| 需求 | LangGraph + LangChain | OpenAI Agents SDK Python | 结论 |
|---|---|---|---|
| 显式状态图、条件分支和人工中断 | `StateGraph`、checkpointer、interrupt 直接建模 | 可通过 agent loop/session 实现，但复杂分支和补偿逻辑需应用层维护 | LangGraph 更匹配 |
| PostgreSQL 断点恢复和稳定 `thread_id` | 可将 checkpoint 接到自定义 PostgreSQL 存储 | session/state 可用，但本项目的业务快照、补丁和事务仍需自行维护 | LangGraph 更匹配 |
| OpenAI、兼容接口、Ollama 多模型 | LangChain 的模型抽象更自然 | 可接入自定义 provider，但需要额外适配和能力约束 | LangChain 更匹配 |
| 确定性解析、评分、补丁和文件副作用 | 普通 Python 节点可与 LLM 节点混合 | 也能实现，但 agent loop 不是这些步骤的必要抽象 | LangGraph 更清晰 |
| 工具调用和 MCP | 可把 MCP 工具包装为节点工具 | SDK 对 tools/handoffs 有一等支持 | 两者均可，非决定因素 |
| OpenAI 专属 tracing/guardrail | 需要自行接入 tracing/遥测 | SDK 原生体验更好 | 可作为可选增强 |

本项目的流程是“状态机 + 人工审批 + 文档副作用”，不是开放式多智能体对话。因此建议：

1. 规则解析、证据索引、评分公式、补丁应用、文件转换使用普通 Python/纯函数节点；
2. 只有 JD 归类、语义匹配、候选文本和反馈分类使用 LangChain 模型节点；
3. LangGraph 只负责状态、分支、并行、checkpoint、interrupt 和恢复；
4. 若未来需要 OpenAI 专属 tracing 或某个独立对话型 agent，再把 OpenAI Agents SDK 封装成隔离适配器，不让两个框架同时拥有主状态和副作用写入权。

## 9. 简历解析与模板处理

### 9.1 输入校验

- 每个任务只允许一份主简历，可从历史记录选择。
- 单文件不超过 10 MB、最多 3 页。
- 支持 `.docx` 和包含可复制文本的 PDF；不支持 `.doc` 和简历 PDF OCR。岗位截图 OCR 是独立输入通道。
- 原始文件复制到任务目录，保留原始版本以便恢复和重新解析。

### 9.2 结构化预览

解析后展示并允许修正姓名、联系方式、教育经历、工作经历、项目经历和模块边界。用户手动修正的内容视为已确认事实。若无法识别项目区域，暂停流程，用户在预览中选择起止段落并为每个区域填写项目名。

PDF 先解析并重建为可编辑 DOCX。若文字顺序或版式信息严重混乱，暂停并提示无法安全重建，而不是继续产生错位文件。

### 9.3 模板决策

- 有原简历：询问保留原模板或选择 GitHub 模板，并分别展示预览。
- 无原简历：先从 GitHub 中文模板选择，获取失败或用户不选择时使用内置模板。
- 迁移到新模板后先生成完整预览，用户确认布局后才进入匹配和内容优化。
- 映射或排版无法安全完成时暂停，保留原简历版本，让用户回到原模板或取消迁移。

### 9.4 复杂排版保护

尽量原样保留正文/标题格式、多栏、表格、页眉页脚和浮动图形。浮动图形只识别和保留，不移动、缩放或替换。任何无法安全修改的文本框、浮动对象或嵌套布局都触发暂停；不能静默破坏原版式。

### 9.5 无原简历制作

必填：姓名、电话、邮箱、求职目标。可空：所在地、教育经历、证书。工作经历可选，仅写入用户提供的内容。生成前展示基础信息预览并要求确认。基于 JD 生成的简介、技能、个人评价和项目全部标记“示例/待核实”。

## 10. 匹配分析与内容生成

### 10.1 JD 拆解

JD 解析结果至少包含岗位目标、职责清单、任职要求、技能清单、关键词集合和来源段落证据。每个拆解后的原子职责、技能和关键词都必须绑定有效 `evidence_id`；缺证据、重复冲突或结构化校验失败时，只允许重试/重新解析，不能将无来源原子项用于评分或生成。解析出的清单在匹配页面展示，但用户不能手动修改解析结果；如解析失败，只能重新解析，不能用手动粘贴替代详情解析。

### 10.2 匹配评分

总分为 `0–100`，四个维度默认各占 25%，用户可自定义。每项权重范围为 `0–100%`，总和必须等于 `100%`，否则不能保存。

确定性指标与模型语义评分的合成比例默认各为 50%，后续允许用户调整。每个维度的两项比例必须分别处于 `0–100%` 且总和为 `100%`；这组单项分合成比例与四个维度之间的总分权重是两层独立配置，分别校验、分别保存。

第一版 `scoring_config` 的可复算规则如下：

1. JD 关键词、技能和职责先拆成去重后的原子项；除 JD 明确给出优先级外，原子项默认等权。
2. 关键词确定性分 = 命中的归一化关键词权重之和 / 全部关键词权重之和 × 100；同一原子项重复命中只计一次。
3. 技能确定性分 = 有效证据覆盖的技能原子项数 / 技能原子项总数 × 100。
4. 职责确定性分 = 有效证据覆盖的职责原子项数 / 职责原子项总数 × 100。
5. 项目确定性分 = 已确认项目覆盖的职责和技能原子项数 / JD 职责与技能原子项总数 × 100；同一原子项跨项目重复覆盖只计一次。
6. 语义分要求模型按固定五档 rubric 返回档位和有效 `evidence_id`，由后端映射为 `0/25/50/75/100`：`0=无相关`、`25=弱相关`、`50=部分覆盖`、`75=强相关`、`100=直接且充分覆盖`。模型不得自由返回任意分数；档位非法、缺证据或引用哈希不符时该语义项无效，不参与加分并触发重试/人工处理。
7. 每个维度先按该维度的确定性/语义比例合成，再按四项维度权重计算总分。内部计算保留足够精度，展示和持久化结果保留 1 位小数；80 分触发判断使用未四舍五入的总分。

`scoring_config` 必须记录原子项提取版本、归一化词典版本、五档语义 rubric 版本、两层权重、舍入规则和计算时间；旧 `match_run` 永不因配置更新而重算。

候选去重在任务初始化时先固定 `embedding_mode`：

- `embedding`：使用已探测通过的 embedding profile，对候选正文计算归一化向量余弦相似度；第一版默认阈值为 `0.85`。
- `tfidf_fallback`：仅在任务启动时 embedding 不可用且用户允许降级时使用中文字符 2–4 gram 加技术词项的 TF-IDF，L2 归一化后计算余弦相似度；第一版独立默认阈值为 `0.75`，不得复用 `0.85`。

两种模式都先对标准化项目标题和技术栈做精确去重。超过对应阈值时保留后端匹配分更高者，另一槽位标记 `duplicate` 并保存相似度、保留理由、模型/降级方式和算法版本。`embedding_mode`、阈值、中文切分/字符 n-gram 规则、向量模型和算法版本属于 `scoring_config`，旧 `match_run` 不因模型或阈值变化而重算。任务一旦使用真实 embedding，恢复时即使模型被卸载也必须保持 `status=paused, blocked_reason=reindex_required` 并等待人工处理，不能静默切换为 TF-IDF；已经选择 `tfidf_fallback` 的任务恢复时也必须继续使用相同版本的降级算法。

| 维度 | 说明 |
|---|---|
| 技能匹配 | 简历技能与岗位技能的标准化覆盖及语义相关性 |
| 职责匹配 | 已确认经历对 JD 职责的覆盖程度 |
| 项目匹配 | 项目内容、技术方案和结果与目标岗位的相关性 |
| 关键词覆盖 | 归一化关键词在简历中的可证据命中率 |

每个维度分数也由后端计算，不能直接信任模型返回值。后端分别取得可复算的确定性指标（标准化词项命中、职责覆盖、项目字段覆盖等）和模型语义评分，再按该维度配置的合成比例计算 `0–100` 分；四个维度分数最后按用户权重计算总分。确定性指标与语义评分的合成比例允许在设置中调整，必须通过范围与总和校验。

关键词由后端标准化（大小写、空格、中文同义词、技术栈别名），保存“原词—归一化词—命中简历片段”。模型语义评分必须同时返回理由和证据引用，后端校验证据确实存在于来源中；缺少有效证据的语义项不参与加分。页面展示确定性分、语义分、合成比例、维度最终分和总分公式。某维度没有可评估信息时按 `0` 分，并说明“信息不足”，不得猜测。

总分低于 80 分时触发：

- 推荐一个待替换项目（仅在原项目数至少 2 个时）；
- 提示逐项检查个人简介、技能清单、个人评价。

无论总分是否低于 80 分，都必须生成新的候选项目。80 分阈值为第一版固定规则。

### 10.3 项目候选

生成前允许补充目标岗位级别、篇幅预设/高级字数限制、重点技术栈和候选数量。候选数量 `1–5`，默认 3 个；三个候选应采用不同侧重点（例如技术深度、业务闭环、工程落地），避免近似重复。

每个候选结构为：

1. 项目名称与项目时间；
2. 项目简介；
3. 技术栈；
4. 分条解决方案；
5. 结果与指标。

无法从来源得出的时间、指标和比例使用 `[待补充]`。允许模型根据 JD 合理推导未明确写出的技术方案和职责，但所有推导内容都必须保持“示例/待核实”并逐字段确认。候选展示覆盖的 JD 职责、匹配技能、缺失占位符、四维评分、评分理由和证据。候选确认后提供同时写入富文本和纯文本的“一键复制到剪贴板”按钮；复制内容仍保留草稿/确认状态。

候选正文先规范化并计算内容哈希，再按任务固定的 `embedding_mode` 生成或复用 `embedding_artifacts`。向量/TF-IDF 索引只允许引用同一 profile version、算法版本和维度；候选反馈重生成产生新内容哈希和新 artifact，不覆盖旧 artifact。

### 10.4 项目替换、添加与排序

- 原项目数至少 2 且总分低于 80：系统推荐一个匹配度最低/差距最大的项目，用户可查看所有项目评分和理由并改选替换对象。
- 替换后项目总数保持不变。
- 原项目数少于 2：加入用户最终选择的候选项目。
- 新增或保留项目按匹配度排序，用户随后可手动调整顺序。
- 用户最终只能选择一个候选作为本次采用内容。
- 替换前的原项目保留为可恢复版本。

### 10.5 模块优化顺序

项目处理后，只有总分低于 80 分时才逐项询问是否优化：

1. 个人简介；
2. 技能清单；
3. 个人评价。

工作经历、教育经历、证书和求职意向第一版不自动优化。每个模块都显示原文/建议对比、反馈栏和多轮“生成—反馈—再生成”，直到用户确认或跳过。所有版本都保留并支持回退。

改写已有内容不得增加用户未提供或确认的事实、技术经历和指标。反馈中明确提供的新事实可进入确认队列；单纯的风格意见不构成事实确认。

## 11. 事实确认、版本与个人 Skill

### 11.1 确认规则

- 所有 AI 生成/修改结果在写入正式简历前都必须明确确认。
- “示例/待核实”内容需要逐字段勾选事实依据。
- 未完成确认的候选只能预览或复制，不能写入正式文件。
- `[待补充]` 必须填写，或由用户逐项明确同意保留；保留时预览和导出状态标记“含未完成信息”。
- 完成事实确认并处理占位符后自动移除“示例/待核实”标记。
- 确认记录包含用户、时间、版本 ID、字段和勾选内容。
- 每条字段确认必须绑定确认时的 `value_hash`、来源版本和补丁版本；后续补丁、用户编辑、模板迁移或来源快照变化导致哈希不匹配时，只将该字段标记为 `stale` 并撤销其确认，其他未变化字段继续有效，导出前必须重新确认所有 `stale` 字段。

### 11.2 版本模型

项目、个人简介、技能清单、个人评价、模板迁移和导出预览都保留版本。版本至少记录：

- 版本 ID、父版本 ID、草稿/正式状态；
- 模型配置标识（不含密钥）、提示词版本、生成时间；
- 输入来源版本、用户反馈、证据引用；
- 用户确认状态和确认记录。

版本采用不可变策略：用户点击“重试当前步骤”、提交反馈后重新生成，或切换模型继续生成时，均创建新的草稿版本，不能覆盖旧版本。旧版本（包括用户不满意的结果）继续保留并可比较、回退；没有生成内容的失败尝试也保存为带错误信息的尝试记录，并与后续版本建立父子关系。

模型切换生成的结果与原模型版本在不同 `generation_branch_id` 下并列保存，用户可比较并选择；新结果必须结合已保存上下文，不从零重写。分支选择本身也记录为用户动作，未被选择的分支仍保留为草稿历史。

每次用户完成一轮正式确认，或进入导出准备阶段，都创建不可变 `resume_snapshot`（相同内容可通过内容哈希复用）。快照固定当时的项目/个人简介/技能/个人评价正式版本 ID、项目顺序、基础信息版本、模板版本、排版参数、确认集合哈希、来源 JD 快照和 Skill 版本。模块后续回退或新修改只能创建新的快照，不能改变旧快照；导出、预览和导出反馈都必须显式绑定一个 `snapshot_id`。

### 11.3 个人偏好 Skill

个人 Skill 是项目目录下可读、可版本化的 Markdown 文件，不是动态修改 Codex 的系统 Skill。只保存用户确认过的表达、篇幅、排版和生成偏好：

- 每次新任务自动加载；单次任务可关闭。
- 当前任务要求和最新反馈优先；发生冲突时提示用户并在确认后更新 Skill。
- 不得修改匹配权重、事实内容或安全确认规则。
- 用户手动编辑时校验 Markdown 结构、无效规则和冲突规则。
- 每次确认更新或手动保存自动创建快照，支持差异查看和回退。
- “删除全部历史数据”不删除 Skill；Skill 只通过单独入口管理。

## 12. 一页排版与导出

### 12.1 导出格式

用户可选择 DOCX、PDF 或两者。DOCX 与 PDF 需要保持视觉排版一致，合规结果必须恰好 1 页且没有空白第二页。

设置允许用户指定默认字体、字号、行距和页边距；未设置时使用系统默认值。指定字体未安装时必须提示并提供替代方案。

### 12.2 压缩流程

1. 检查分页；超过一页先提示用户，不自动压缩。
2. 用户同意后，先只调整字体、字号、行距、段距和页边距。
3. 正文内部保持字体/格式一致，标题内部保持字体/格式一致；不混用局部样式。
4. 使用默认可读性下限（字号、行距、页边距），设置中允许用户调整。
5. 生成页面级预览，等待用户反馈。
6. 只有用户再次明确允许，才使用 AI 合并或缩短正文；先保留版本再改写。
7. 达到阈值仍超过一页时，让用户手动选择隐藏/删除模块或内容，不自动删减。
8. 用户拒绝压缩时仍允许导出多页版本，但界面和文件状态明确标记“未满足一页要求”。

原始简历超过一页也遵循同样流程。所有 `[待补充]` 和未确认内容在正式导出前必须处理或明确保留。

### 12.3 引擎与失败处理

- 优先 Microsoft Word，LibreOffice 作为备用。
- 启动检查未安装办公组件时提示用户手动安装后重新检测；不自动下载/安装。
- 没有办公组件时仍可完成职位解析、匹配和文本生成，但禁用 DOCX/PDF 排版预览与导出。
- 字体未安装时提示用户并提供可用替代，不静默替换。
- 转换失败提供“重试当前转换”和“切换备用引擎”。
- 预览反馈可定位页、模块或段落，提交后只重跑相关排版/生成步骤。

## 13. PostgreSQL 数据模型

以下为逻辑模型，实际字段可在迁移脚本中细化：

| 表 | 用途 |
|---|---|
| `tasks` | 任务目录、职位、绑定的 `job_snapshot_id`、固定的 `chat_model_profile_version`、固定的 `embedding_model_profile_version`（可为空但必须记录 `embedding_mode`）、状态、队列序号、超时、恢复信息和稳定 `thread_id` |
| `task_events` | 状态变化、用户动作、脱敏错误和进度事件 |
| `node_operations` | 节点 `operation_key`、输入哈希、输出引用、状态和幂等结果 |
| `node_attempts` | 节点每次内部模型尝试、错误、耗时和重试序号（不等同于内容版本） |
| `context_snapshots` | 节点上下文清单、证据 ID、模型窗口/预算、裁剪顺序与结果、阻止原因和清单哈希 |
| `model_invocations` | `request_key`、模型配置版本、请求参数摘要、结构化响应引用、上下文哈希、证据 ID、状态和错误摘要；完整 prompt/response 仅在调试模式临时留存 |
| `worker_leases` | `worker_id`、`task_id`、租约状态、过期时间和最近心跳 |
| `langgraph_checkpoints` | 按 `thread_id` 保存轻量图 checkpoint、节点状态、checkpoint 版本和业务对象引用 |
| `job_records` | 职位字段、来源 URL、`extra.hr_activity`、兼容发布时间、快照哈希和解析版本 |
| `job_snapshots` | 任务使用的不可变 JD 字段、原始快照引用、证据集合和抓取版本 |
| `job_evidence` | JD 段落、字段证据、来源定位、稳定 `evidence_id` 和文本哈希 |
| `resume_files` | 原始/重建文件、格式、页数、哈希和任务引用 |
| `resume_sections` | 结构化模块、段落 ID、页码、稳定 `evidence_id`、文本哈希和用户修正状态 |
| `template_records` | GitHub 来源、许可证、Star、版本、缓存路径和更新状态 |
| `match_runs` | 权重、四维分数、总分、证据和模型版本 |
| `scoring_configs` | 原子项提取/归一化/rubric 版本、两层权重、舍入规则和配置快照 |
| `generated_versions` | 项目/模块草稿、正式版本、`generation_branch_id`、`candidate_slot_id` 和来源 |
| `content_patches` | 目标模块/条目、旧值哈希、新值、理由、证据、应用状态、目标草稿版本和审计 diff |
| `resume_snapshots` | 整份简历组合版本、模块版本 ID、项目顺序、模板/排版版本、确认集合哈希和来源快照 |
| `feedback_records` | 用户反馈、分类、目标版本和处理状态 |
| `feedback_classifications` | 分类器类型、类别、置信度、字符区间、事实候选和冲突列表 |
| `confirmations` | 字段级事实确认、`value_hash`、来源/补丁版本、占位符确认、状态和时间 |
| `skill_versions` | Skill Markdown、快照、差异和有效性 |
| `model_profiles` | `role`（chat/embedding）、provider、地址、模型名、Ollama 安装/服务状态、能力探测和默认标识（无密钥） |
| `model_profile_versions` | 任务可引用的不可变聊天/embedding 配置与参数快照（无密钥），包含模型 tag/digest、向量维度、探测结果、tokenizer/估算器、`budget_policy_version`、chunk/embedding 算法版本和降级策略 |
| `embedding_artifacts` | 内容哈希、profile version、模型 tag/digest、向量维度、向量或本地索引引用、算法版本、状态；禁止混用不同 profile 的向量 |
| `candidate_slots` | 候选槽位 ID、生成状态、去重状态、排序分和淘汰原因 |
| `consents` | 外部数据发送和 BOSS 页面访问风险同意状态 |
| `export_runs` | 强制绑定 `snapshot_id`，并记录引擎、格式、页数、预览、反馈和合规状态 |
| `backup_manifests` | 备份清单、校验值、创建时间和恢复记录 |

API Key、GitHub Token 不保存到数据库、配置文件、日志或备份，也不写入 Windows Credential Manager/DPAPI。它们只在当前进程的内存中短暂存在：API Key 仅随检测或任务请求提交，GitHub Token 仅随 GitHub MCP 请求提交；服务重启后必须重新输入并检测。异步 worker 只能接收带 TTL 的内存凭据句柄，不能把明文凭据写入队列；进程重启、TTL 到期或凭据被清除时，任务保持 `status=paused, blocked_reason=needs_credentials`，暂停恢复并提示用户重新输入。数据库只保存非敏感的 profile 配置和 `credential_required=true` 状态。

## 14. 本地文件布局

数据根目录默认是项目目录，也可在设置中修改：

```text
<data-root>/
├─ tasks/<job>-<company>-<timestamp>/
│  ├─ source/resume-original.*
│  ├─ source/job-snapshot.html
│  ├─ parsed/
│  ├─ drafts/<version-id>/
│  ├─ previews/
│  ├─ exports/
│  └─ manifest.json
├─ templates/cache/<template-id>/
├─ skills/resume-preferences.md
├─ backups/
├─ logs/
├─ edge-profile/                 # 旧版位置；Windows 运行时改用 %LOCALAPPDATA%/ResumeAgent/edge-profile
└─ app-config.toml               # 不保存密钥明文
```

刷新历史 JD 时直接覆盖该职位的解析结果、快照和 HR 活跃状态；同一职位重新开始完整流程时创建新的任务目录。职位快照必须剥离 Cookie、登录凭证和其他会话信息，并保留来源 URL。

## 15. 任务队列、取消与恢复

- 同一时间只运行一个简历任务；其他任务按创建时间 FIFO 排队。
- 排队任务可取消或删除，不改变其他任务顺序。
- 取消采用协作式取消：当前不可中断的模型/文件操作完成后停止后续步骤，保存中间结果并标记 `cancelled`。
- 模型自动重试只重试当前节点；达到上限后标记失败，显示“重试当前步骤”。用户触发的重试必须创建新的草稿版本/尝试记录，不覆盖原版本。
- 任务超时（默认 5 分钟，可调整）暂停并保留 checkpoint。
- 总任务超时优先于节点级重试；超过总时限后不再开始新的重试。并行候选取消时停止尚未开始的槽位，已完成槽位和其版本继续保留。
- 应用重启后恢复未完成任务；用户选择继续或重试，不隐式从头执行。
- 旧任务恢复、继续或重试沿用原 `thread_id`；新任务创建新的 `task_id`、目录和 `thread_id`，不得复用旧图状态。
- SSE 事件至少包含任务状态、当前节点、百分比/阶段、可恢复动作和脱敏错误。

建议状态集合：`queued`、`running`、`waiting_user`、`paused`、`failed`、`cancelled`、`completed`、`deleted`。`needs_credentials`、`needs_model_recheck`、`reindex_required` 和 `context_budget_blocked` 是 `blocked_reason`，不是新的顶层状态；恢复接口必须先清除对应原因并重新完成门禁检查。

## 16. FastAPI 接口草案

以下是面向 Streamlit 的内部 REST 契约，所有请求需要启动级内部令牌：

中断恢复类接口必须携带 `thread_id`、当前 checkpoint 版本和结构化动作；服务端验证动作属于当前中断点且 checkpoint 版本未过期后，才调用 LangGraph resume。冲突响应应返回最新状态摘要和刷新提示，但不返回可能被误用的写入结果。

设置写入同样使用 `settings_version` 乐观锁；`PUT /api/settings` 必须携带 `If-Match`，冲突时拒绝覆盖。凭据字段是 write-only，响应和 `GET /api/settings` 只返回 `credential_required`、`configured_in_session`、最近探测状态和时间，不返回密钥或 Token。Ollama 扫描只读取服务和已下载模型清单，不自动执行 `pull`。

```text
GET  /api/health
GET  /api/environment
POST /api/environment/{component}/recheck

GET  /api/settings
PUT  /api/settings                 # If-Match: settings_version; secret fields are write-only
GET  /api/models?role=chat       # role=embedding returns the other profile type
POST /api/models/ollama/scan       # returns installed names; role suitability still needs separate probe
POST /api/models/{profile_id}/probe
POST /api/credentials/session      # write-only secret input; returns an in-memory handle/expiry only
DELETE /api/credentials/session/{handle_id}
POST /api/consents/external-model
POST /api/consents/post-time-risk

POST /api/jobs/search
POST /api/jobs/from-url
GET  /api/jobs/{job_id}
POST /api/jobs/{job_id}/reparse
POST /api/jobs/{job_id}/refresh
GET  /api/jobs/history

POST /api/resumes/upload
GET  /api/resumes/history
POST /api/resumes/{resume_id}/parse
PATCH /api/resumes/{resume_id}/sections
POST /api/resumes/{resume_id}/mark-project-region

POST /api/templates/search
POST /api/templates/{template_id}/preview
POST /api/templates/{template_id}/download
POST /api/templates/{template_id}/migrate
POST /api/templates/{template_id}/check-update

POST /api/tasks
GET  /api/tasks/{task_id}
GET  /api/tasks/{task_id}/events       # SSE
POST /api/tasks/{task_id}/resume
POST /api/tasks/{task_id}/cancel
DELETE /api/tasks/{task_id}

POST /api/tasks/{task_id}/feedback
POST /api/tasks/{task_id}/confirmations
POST /api/tasks/{task_id}/retry-step
POST /api/tasks/{task_id}/export-preview
POST /api/tasks/{task_id}/export

GET  /api/skills
PUT  /api/skills
GET  /api/skills/versions
POST /api/skills/rollback

POST /api/backups/create
POST /api/backups/validate
POST /api/backups/restore
```

`POST /api/credentials/session` 的请求体可以包含一个 profile 的 API Key 或 GitHub Token，但响应只返回不可逆的 `credential_handle_id`、作用域和过期时间；明文不进入响应、日志、SSE、checkpoint 或数据库。创建外部模型任务时，客户端在请求体中携带当前内存 handle；worker 通过 handle 读取凭据并在 TTL 到期后清除。服务重启后 handle 全部失效，相关任务按 `blocked_reason=needs_credentials` 暂停。

## 17. 安全、隐私与日志

### 17.1 凭据

- API Key、GitHub Token 均为 write-only 输入，只在当前进程内存中短暂使用；界面不回显明文，`GET /api/settings` 只返回 `credential_required`、`configured_in_session` 和最近探测状态。
- 两类凭据不写入 Windows Credential Manager/DPAPI、配置文件、PostgreSQL、日志、快照或备份。服务重启、用户清除会话或凭据 TTL 到期后必须重新输入并检测。
- 恢复备份后相关配置显示“未配置”；依赖外部模型或 GitHub 的排队任务保持 `status=paused, blocked_reason=needs_credentials`，不得以空凭据自动重试。

### 17.2 外部模型同意

首次准备把真实简历、JD 或用户反馈发送到外部模型时说明数据范围并要求确认；仅发送合成短文本的连接探测不触发同意。同意状态可在设置中撤销，撤销后下一次真实数据调用重新确认。第一版不做自动脱敏。使用 Ollama 不触发外部服务同意。第一版不展示 Token 使用量或费用估算。

### 17.3 日志与快照

日志只保留错误、状态和必要诊断，默认脱敏姓名、电话、邮箱、地址、完整简历和完整 JD。非必要日志默认保留 30 天，用户可调整或关闭自动清理；用户明确保存的文件和 Skill 不自动删除。BOSS 快照去会话信息。

模型调试模式默认关闭；开启时必须显示敏感内容留存提示，调试 prompt/response 只保存在本机临时目录，遵循配置的保留期限，删除历史或创建备份时不纳入归档。

### 17.4 本地接口

FastAPI、Streamlit、Edge helper 只监听 loopback；内部请求必须携带启动级随机令牌。第一版不提供用户登录，默认不开放局域网。

## 18. 备份、恢复与删除

### 18.1 备份

用户手动点击执行备份。备份包括 PostgreSQL 数据、任务目录、原始简历、JD 快照、导出文件、模板缓存、Skill 和版本记录；排除 API Key、GitHub Token、Edge profile 和 BOSS 登录会话。备份不额外加密。

备份创建期间阻止新任务，并等待当前任务进入可保存状态，生成包含文件清单、版本信息和 SHA-256 校验值的归档。

### 18.2 恢复

恢复前停止运行和排队任务，先校验清单并解包到临时目录，展示同名任务/文件冲突。用户逐项选择覆盖、跳过或另存为新目录，确认后一次性写入；失败自动回滚。恢复后重新加载 LangGraph 状态和数据库记录，缺失凭据需要重新配置。

### 18.3 删除

支持按任务、文件或全部历史数据删除。删除任务同步删除 PostgreSQL 记录、任务目录、版本和反馈；独立备份不变。删除全部历史数据不删除个人 Skill。

## 19. 实现阶段建议

### 阶段 0：工程骨架

- 初始化 Python 项目、依赖锁定、配置和日志脱敏。
- Docker PostgreSQL 启动脚本、迁移、健康检查和本地 loopback 令牌。
- FastAPI/Streamlit 最小页面与 SSE 事件协议。

### 阶段 1：环境与凭据

- Edge/Word/LibreOffice/Ollama 检测。
- 内存凭据作用域、TTL、清除和 `needs_credentials` 恢复门禁（不持久化 API Key/GitHub Token）。
- 模型配置、能力探测、默认模型门禁。

### 阶段 2：MCP 与职位

- Python stdio MCP host/servers。
- Playwright Edge helper、登录状态和人工处理暂停。
- BOSS URL/搜索/可靠分页/详情/快照/HR 活跃状态与访问风险确认。
- 岗位截图粘贴/上传、PP-OCRv6 medium 识别、校正与确认 API。
- GitHub 模板搜索、许可证过滤、预览和缓存。

### 阶段 3：解析与匹配图

- DOCX/PDF 解析和结构化预览。
- LangGraph checkpoint、Pydantic 节点契约、证据定位。
- 四维评分、权重设置和匹配解释。

### 阶段 4：生成与确认

- 三候选并行生成、去重、候选失败重试。
- 项目替换/新增/排序。
- 三个模块的反馈循环、版本和字段级事实确认。
- Markdown Skill 加载、校验、快照和回退。

### 阶段 5：排版、导出与运维

- 模板迁移、Word/LibreOffice 渲染、页数检测和一页审批。
- DOCX/PDF 导出、预览反馈、重试/备用引擎。
- 备份恢复、删除、日志清理和重启恢复。

## 20. 验收标准

### 功能验收

- 无有效默认模型时，任何解析/匹配/生成任务均被阻止并给出修复入口。
- 聊天模型和 embedding 模型可分别填写名称；默认分别为 `qwen2.5:7b` 和 `bge-m3`。Ollama 扫描能列出电脑中精确已下载的模型 tag，并分别完成角色探测。缺失模型只提示用户手动 `ollama pull` 后重新检测，不自动替换或下载。
- embedding 缺失时，任务在初始化阶段明确选择 `tfidf_fallback` 或阻止；恢复过程中不得在 `embedding` 与 `tfidf_fallback` 之间静默切换。
- 同一模型、输入、tokenizer/估算器和预算策略版本必须得到稳定的 token 估算与 `context_manifest_hash`；超出 `usable_input` 时在 HTTP 请求发出前阻止调用，不推进 checkpoint。
- 上下文门禁页面显示模型窗口、输入估算、输出预留、安全余量、被裁剪条目和恢复入口；核心事实放不下时状态为 `paused + blocked_reason=context_budget_blocked`，用户可减少内容、调整输出上限或切换模型后重试。
- 工具调用多轮累计的消息、schema 和工具结果必须重新计算预算；provider 返回 `context_length_exceeded` 时不得自动重试或推进图状态。
- embedding 的最大输入长度和 chunk 策略独立于聊天模型窗口；长度未知或 chunk 失败时阻止向量生成，除非用户明确选择降级。
- tokenizer/估算器测试必须覆盖中文、英文、混合文本、JSON、输出 schema、工具 schema 和边界“刚好放入/超出 1 token”场景；tokenizer 加载失败只能显式走保守 fallback，窗口未知或估算异常必须在发出模型请求前阻止。
- BOSS URL 白名单、登录暂停、访问风险确认和 HR 活跃状态提取均生效。
- 搜索首批最多 20 条；慢加载不清除分页游标，可再次点击；只有明确到底才禁用下一页；详情按选择后请求，HR 活跃筛选和历史 JD 刷新符合定义。
- 剪贴板粘贴与图片上传均可进入 OCR；非法/超限图片被拒绝；缺依赖/缺模型给出修复提示；校正确认前不创建职位，确认后使用修正文本解析 JD。
- JD 刷新不会改变既有任务的 `job_snapshot_id`、证据或匹配复现结果。
- `.docx`/文本型 PDF 校验 10 MB/3 页；扫描 PDF 和 `.doc` 被拒绝。
- 解析结果可手动修正；无法识别项目时必须手动标记后才继续。
- 总分和四维分数、权重校验、证据、80 分触发规则正确。
- 无论总分高低都生成候选；低于 80 才推荐替换和模块优化。
- 未确认内容不能写正式文件；示例字段和占位符流程可审计。
- 用户可反馈、切换模型、比较版本、回退和复制富文本/纯文本。
- DOCX/PDF 可选导出；一页压缩需用户同意，超限时允许手动删减；无办公组件禁用导出。
- 任务重启可恢复，单任务 FIFO、取消、重试、超时和中间结果有效。
- worker lease 过期后只能由新 worker 接管任务；无有效 lease 的进程不能执行节点或写入副作用。

### 安全验收

- 日志、备份、数据库和快照中不存在 API Key、GitHub Token、Cookie 或完整登录会话。
- 外部模型首次同意、撤销再确认和 HTTPS 校验生效。
- 外部文本中的 prompt injection 不能改变系统策略。
- Edge helper 和内部 API 未携带随机令牌时拒绝请求。

### 数据验收

- 任务目录命名为“职位-公司-时间戳”，同职位新任务不覆盖旧任务。
- 正式确认和导出均生成/绑定不可变 `resume_snapshot`；导出不能读取未绑定的可变草稿。
- 备份清单和校验值可验证；恢复冲突有覆盖/跳过/另存选项且失败回滚。
- Skill 版本快照、差异、回退可用，删除全部历史不影响 Skill。

## 21. 后续可微调项

以下不改变已确认产品边界，可在初版实现后调整：

- Streamlit 进度通道最终选 SSE 还是 WebSocket（第一版优先 SSE）。
- PostgreSQL 镜像具体版本、默认端口、数据库名和配置文件格式。
- 解析器具体库和复杂 DOCX 元素的支持矩阵。
- 匹配语义模型的校准方法和去重阈值。
- 内置模板的具体视觉样式。
- 岗位截图 OCR 的低置信度提示阈值和更多真实截图回归样本。
- API 路径命名、分页字段和前端组件细节。
