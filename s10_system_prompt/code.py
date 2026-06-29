#!/usr/bin/env python3
"""
s10: System Prompt - 使用千问/OpenAI-compatible 工具格式动态组装系统提示词。

Run:  python s10_system_prompt/code.py
Need: pip install openai python-dotenv

.env example:
    DASHSCOPE_API_KEY=sk-xxxx
    QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    MODEL_ID=qwen-plus

相对 s09 的变化：
  - PROMPT_SECTIONS：按主题拆分 system prompt 片段
  - assemble_system_prompt(context)：根据真实上下文选择并拼接片段
  - get_system_prompt(context)：用 json.dumps 做确定性的缓存 key
  - agent_loop：每轮请求前把动态 system prompt 放进 system 消息

当 .memory/MEMORY.md 存在且有内容时，才注入 memory 片段。
"""

import json
import os
import subprocess
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# 工作目录和 memory 文件都来自真实运行状态，而不是用户问题里的关键词。
WORKDIR = Path.cwd()
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

# 千问 DashScope 提供 OpenAI-compatible 接口，所以直接使用 OpenAI SDK。
# 优先读取 DASHSCOPE_API_KEY；如果你复用 OpenAI 环境变量，也可以走 OPENAI_API_KEY。
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=(
        os.getenv("QWEN_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)
MODEL = os.environ.get("MODEL_ID", "qwen-plus")


# System Prompt 片段

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    """根据当前上下文选择 system prompt 片段，并按稳定顺序拼接。"""
    sections = []

    # 固定加载：身份、工具说明、工作目录。
    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["tools"])
    sections.append(PROMPT_SECTIONS["workspace"])

    # 条件加载：只有真实 memory 文件有内容时，才注入 memory。
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")

    return "\n\n".join(sections)


_last_context_key = None
_last_prompt = None


def get_system_prompt(context: dict) -> str:
    """system prompt 缓存：上下文不变时，不重复组装字符串。"""
    global _last_context_key, _last_prompt

    # 不使用 Python 内置 hash()，因为它有进程随机化，也不适合嵌套结构。
    # json.dumps(sort_keys=True) 能为同一个上下文生成稳定的字符串 key。
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt

    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)

    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt


# 工具实现

def safe_path(p: str) -> Path:
    """把相对路径限制在当前工作区内，避免读写工作区外的文件。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令，并把 stdout/stderr 合并返回给模型。"""
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容；limit 用于限制返回的行数，避免上下文过长。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件内容；父目录不存在时自动创建。"""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# 千问/OpenAI-compatible 工具声明格式：
#   - 外层 type 固定为 "function"
#   - function.parameters 使用 JSON Schema 描述参数
#   - 模型返回工具调用时，会放在 assistant 消息的 tool_calls 字段里
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

TOOL_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write}


def read_text_with_fallback(path: Path) -> str:
    """读取文本文件：优先 UTF-8，失败时兼容常见中文 Windows 编码。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    # 最后的兜底：保留可读内容，无法解码的字节用替换符显示，避免启动直接崩溃。
    return path.read_text(encoding="utf-8", errors="replace")


# 上下文派生

def update_context(context: dict, messages: list) -> dict:
    """从真实状态派生上下文：启用哪些工具、memory 文件是否存在。"""
    memories = ""
    if MEMORY_INDEX.exists():
        content = read_text_with_fallback(MEMORY_INDEX).strip()
        if content:
            memories = content

    return {
        "enabled_tools": list(TOOL_HANDLERS.keys()),
        "workspace": str(WORKDIR),
        "memories": memories,
    }


# Agent Loop - 千问 / OpenAI-compatible

def _dump_tool_call(tool_call) -> dict:
    """把 SDK 对象转成可放回 messages 历史的 OpenAI/Qwen tool_calls dict。"""
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def agent_loop(messages: list, context: dict) -> str:
    """主循环：动态组装 system prompt，并处理千问/OpenAI 工具调用。"""
    system = get_system_prompt(context)

    while True:
        # OpenAI/Qwen 的 system prompt 是 messages 里的第一条 system 消息。
        # 这里不把 system 永久 append 到 history，避免下一轮上下文变化时留下旧 system。
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        # 如果 assistant 发起工具调用，必须先把这条 assistant/tool_calls 消息加入历史。
        # 后续 role=tool 的结果会通过 tool_call_id 与它关联。
        assistant_message = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if tool_calls:
            assistant_message["tool_calls"] = [_dump_tool_call(tc) for tc in tool_calls]
        messages.append(assistant_message)

        if not tool_calls:
            return msg.content or ""

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            # 千问/OpenAI 的 function.arguments 是 JSON 字符串，需要先解析成 dict。
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: invalid JSON arguments: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                })
                continue

            print(f"\033[36m> {tool_name}\033[0m")

            handler = TOOL_HANDLERS.get(tool_name)
            if handler is None:
                output = f"Unknown tool: {tool_name}"
            else:
                try:
                    output = handler(**args)
                except TypeError as e:
                    output = f"Error: bad tool arguments: {e}"
                except Exception as e:
                    output = f"Error: tool execution failed: {e}"

            print(str(output)[:200])

            # 千问/OpenAI 工具结果格式：role=tool，并带上对应 tool_call_id。
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })

        # 每一轮工具执行后重新读取真实状态，例如 memory 文件可能刚被创建或修改。
        context = update_context(context, messages)
        system = get_system_prompt(context)


if __name__ == "__main__":
    print("s10: system prompt - Qwen runtime assembly")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    context = update_context({}, [])

    while True:
        try:
            query = input("\033[36ms10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        answer = agent_loop(history, context)
        context = update_context(context, history)

        if answer:
            print(answer)

        print()
