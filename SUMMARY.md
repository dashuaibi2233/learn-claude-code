# 项目摘要：Learn Claude Code -- 真实智能体的工程Harness

## 概述
本仓库是一个 **Harness工程项目**，旨在教授如何围绕智能体模型构建工作环境（而非构建模型本身）。它包含 20 个渐进式课程，每节课都在核心智能体循环之上增加一种 Harness 机制。

## 核心理念
- **智能（Agency）源自模型**（训练），**而非外部代码编排**
- **智能体 = 模型 (LLM) + Harness (运行环境)**
- Harness 提供：工具 + 知识 + 观察 + 动作接口 + 权限控制
- 模型负责决策，Harness 负责执行。模型负责推理，Harness 提供上下文。

## 依赖项 (requirements.txt)
- `anthropic>=0.25.0` — Anthropic Claude API 客户端
- `python-dotenv>=1.0.0` — 环境变量管理
- `pyyaml>=6.0` — YAML 解析支持

## 20 个渐进式课程

| 阶段 | 课程 | 重点 |
|-------|---------|-------|
| **1. 让智能体行动** | s01-s04 | 智能体循环、工具使用、权限系统、钩子系统 |
| **2. 处理复杂工作** | s05, s06, s08 | TodoWrite、子智能体、上下文压缩 |
| **3. 记忆与恢复** | s09-s11 | 记忆系统、系统提示词、错误恢复 |
| **4. 运行长任务** | s12-s14 | 任务系统、后台任务、定时调度 |
| **5. 多智能体协调** | s15-s18 | 智能体团队、团队协议、自主智能体、Worktree 隔离 |
| **6. 扩展与组装** | s07, s19, s20 | 技能加载、MCP 插件、综合智能体 |

## 快速开始
```bash
git clone https://github.com/shareAI-lab/learn-claude-code
cd learn-claude-code
pip install -r requirements.txt
cp .env.example .env   # 配置 ANTHROPIC_API_KEY
python s01_agent_loop/code.py
```

## 项目结构
- `s01_*` 到 `s20_*` — 每节课一个文件夹（含 README、翻译、code.py、图片）
- `agents/` — 旧版 12 节课的可运行代码
- `skills/` — s07 课程使用的技能文件
- `docs/` — 旧版 12 节课的文档
- `web/` — 渲染旧版文档的 Web 平台
- `tests/` — 测试文件

## 相关项目
- **[Kode Agent CLI](https://github.com/shareAI-lab/Kode-Agent)** — 开源编程智能体 CLI 工具
- **[Kode Agent SDK](https://github.com/shareAI-lab/kode-agent-sdk)** — 在应用中嵌入智能体能力的 SDK
- **[claw0](https://github.com/shareAI-lab/claw0)** — 姊妹教程：关于始终在线的 AI 助手（心跳、定时任务、IM 渠道）

## 许可证
MIT
