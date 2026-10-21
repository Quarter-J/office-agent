# Office Agent

基于 LangGraph 的多功能智能办公助手，支持消息发送、工作日报、知识库检索、网页搜索和多轮对话。

## 改动说明

- [已完成改动总览](docs/change-summary.md)：Top-k 检索、路由交接、超时、冲突与长上下文处理的改动原因、代码位置、验证结果和当前边界。
- [执行时限与超时结果](docs/execution-timeouts.md)：运行入口、超时配置、返回状态及恢复行为。
- [Agent 冲突处理](docs/agent-conflicts.md)：执行约束、任务去重、冲突记录、文件版本和未知结果核实。
- [长上下文处理](docs/context-management.md)：模型输入预算、滚动摘要、工具回执配对和超限恢复。

## 功能概览

| 功能 | 说明 |
|------|------|
| 消息发送 | QQ（OneBot 协议）、Windows 微信；企业微信、钉钉为预留接口 |
| 工作日报 | 生成、查看 Markdown 工作日报 |
| 知识库检索 | 本地文档 RAG 语义检索（FAISS 向量索引） |
| 网页搜索 | 联网搜索实时信息（天气、新闻、价格等） |
| 多轮对话 | 带短期 + 长期记忆的上下文对话 |

## 架构

```text
+-----------------------------------------------+
|                 交互层                          |
|         CLI 控制台 / Gradio Web 界面            |
+-----------------------------------------------+
|               智能调度层                        |
|   +-------------+    +--------------------+   |
|   | Router Agent |--->| Supervisor + 子Agent|   |
|   | (Tool Call)  |    |  Message / Report / |   |
|   +------+-+----+    |  Search / Content   |   |
|          |            +--------------------+   |
|          v                                      |
|     ToolNode (直接执行工具)                      |
+-----------------------------------------------+
|               能力工具层                        |
|   消息工具 / 日报工具 / 检索工具 / 搜索工具      |
+-----------------------------------------------+
|               数据存储层                        |
|   SQLite / FAISS 向量库 / Redis / 文件系统       |
+-----------------------------------------------+
```

简单任务由 Router Agent 通过 Function Calling 直接选择工具执行；复杂任务（如先生成内容再发送）由 Supervisor 编排多个子 Agent 协作完成。

## 技术栈

- **Agent 框架**: LangGraph + LangChain
- **LLM**: OpenAI API（默认 gpt-4o-mini，兼容其他 OpenAI 格式 API）
- **Web UI**: Gradio
- **CLI**: Rich
- **向量存储**: FAISS
- **关系存储**: SQLite
- **短期记忆**: Redis（可选，无 Redis 时自动回退到内存）
- **配置管理**: Pydantic Settings

## 项目结构

```text
office-agent/
+-- src/
|   +-- agent/            # Agent 图与子 Agent
|   |   +-- graph.py      # LangGraph 工作流定义
|   |   +-- state.py      # 状态模型
|   |   +-- nodes/        # 图节点（intent, router, executor 等）
|   |   +-- sub_agents/   # 子 Agent（supervisor, message, report, search, content）
|   +-- api/              # LLM 接口封装
|   +-- config/           # 配置与模板
|   |   +-- settings.py   # 环境变量加载
|   |   +-- templates/    # 日报模板
|   +-- memory/           # 记忆系统（短期 + 长期）
|   +-- scheduler/        # 定时任务（预留）
|   +-- storage/          # 数据库存储
|   +-- tools/            # 工具实现（消息、日报、检索、搜索）
|   +-- utils/            # 日志、启动检查
|   +-- webui/            # Gradio Web 界面
+-- data/
|   +-- db/               # SQLite 数据库、向量索引
|   +-- knowledge_base/   # 知识库文档目录
|   +-- logs/             # 运行日志
|   +-- reports/          # 生成的日报文件
+-- output/               # 输出目录
+-- build_index.py        # 构建知识库向量索引
+-- run_webui.py          # Web 界面启动入口
+-- requirements.txt      # Python 依赖
+-- .env.example          # 环境变量示例
+-- .gitignore
```

## 快速开始

### 1. 环境要求

- Python 3.8+
- Redis（可选，用于短期记忆持久化）

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填入 LLM 配置：

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini
```

### 4. 启动

**CLI 控制台模式：**

```bash
python -m src.main
```

**Web 界面模式（Gradio）：**

```bash
python run_webui.py
```

启动后访问 http://127.0.0.1:7860

### 5. 构建知识库索引（可选）

将文档放入 `data/knowledge_base/` 目录，然后运行：

```bash
python build_index.py            # 构建索引
python build_index.py --rebuild  # 强制重建
```

## 使用示例

```text
给QQ好友12345发消息：明天下午开会
生成今天的日报，我是张三，完成了项目开发
在知识库中搜索Spring Boot相关文档
搜索一下Python 3.12的新特性
今天天气怎么样
```

## 记忆系统

系统采用双层记忆架构：

- **短期记忆**: 维护当前会话的对话上下文，超过阈值轮次后自动用 LLM 压缩为摘要
- **长期记忆**: 从对话中异步提取用户事实和偏好，存储到 SQLite + FAISS，支持语义检索

每次请求加载相关记忆；交互保存及滚动摘要在收尾阶段执行，用户事实提取异步进行。模型请求按总预算裁剪历史和参考资料，详见 [长上下文处理](docs/context-management.md)。

## 配置说明

所有配置通过 `.env` 文件管理，主要项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥（必填） | - |
| `OPENAI_BASE_URL` | API 地址 | `https://api.openai.com/v1` |
| `LLM_MODEL_NAME` | 模型名称 | `gpt-4o-mini` |
| `LLM_TEMPERATURE` | 生成温度 | `0.7` |
| `QQ_BOT_WS_URL` | QQ 机器人 WebSocket 地址 | `ws://127.0.0.1:6700` |
| `WECOM_WEBHOOK_URL` | 企业微信 Webhook | - |
| `DINGTALK_WEBHOOK_URL` | 钉钉 Webhook | - |
| `REDIS_URL` | Redis 地址（留空使用内存） | - |
| `EMBEDDING_MODEL` | 向量化模型 | `text-embedding-3-small` |
| `CHUNK_SIZE` | 文档分块大小 | `500` |
| `DATABASE_PATH` | SQLite 数据库路径 | `data/db/office_agent.db` |

完整配置项见 [.env.example](.env.example)。
