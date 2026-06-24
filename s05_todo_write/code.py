#!/usr/bin/env python3
"""
s05: TodoWrite - 在 s04 hooks 的基础上增加一个计划工具。

  +---------+      +-------+      +------------------+
  |  User   | ---> |  LLM  | ---> | TOOL_HANDLERS    |
  | prompt  |      |       |      |  bash            |
  +---------+      +---+---+      |  read_file       |
                        ^         |  write_file      |
                        | result  |  edit_file       |
                        +---------+  glob            |
                                      todo_write -> NEW
                                   +------------------+
                                        |
                         in-memory current_todos
                                        |
                        if rounds_since_todo >= 3:
                          inject <reminder>

Changes from s04:
  + todo_write tool + run_todo_write() implementation
  + Nag reminder (inject reminder after 3 rounds without todo update)
  + SYSTEM prompt includes "plan before execute" guidance
  + rounds_since_todo counter in agent_loop
  Loop unchanged: new tool auto-dispatches via TOOL_HANDLERS.

Run: python s05_todo_write/code.py
Needs: pip install openai python-dotenv
.env example:
  DASHSCOPE_API_KEY=sk-xxxx
  MODEL_ID=qwen-plus
  # optional:
  # OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
"""

import ast
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
except ImportError:
    pass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# 当前工作目录就是 agent 可操作的根目录。safe_path 会基于它做路径限制。
WORKDIR = Path.cwd()

# DashScope/千问提供 OpenAI-compatible 接口，所以这里直接使用 OpenAI SDK。
# 如果 .env 中没有 OPENAI_BASE_URL，就默认使用 DashScope 兼容模式地址。
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = os.getenv("MODEL_ID", "qwen-plus")

# todo_write 只维护内存中的任务列表，程序退出后不会持久化。
CURRENT_TODOS: list[dict] = []

# s05 的核心变化：系统提示词要求模型先规划，再执行，并在过程中更新任务状态。
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)
"""
你是一个工作目录位于 {WORKDIR} 的编程智能体。
在开始任何包含多个步骤的任务之前，使用 todo_write 规划任务步骤。
在执行过程中及时更新任务状态。"""


# ============================================================================
# 工具实现：这些 Python 函数是真正会在本地执行的能力
# ============================================================================

def safe_path(p: str) -> Path:
    """把相对路径限制在当前工作目录内，防止读写逃出项目目录。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令，并把 stdout/stderr 合并后返回给模型。"""
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
    """读取文件内容；limit 用来限制最多返回多少行，避免上下文过大。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入完整文件内容；父目录不存在时会自动创建。"""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """做一次精确文本替换，只替换第一次出现的 old_text。"""
    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """按 glob 模式查找文件，例如 **/*.py。"""
    import glob as g

    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# s05 新增：todo_write 计划工具，只记录计划，不直接执行任务
# ============================================================================

def _normalize_todos(todos):
    """兼容模型传入 list 或 JSON 字符串，并校验 todo 数据结构。"""
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"

    if not isinstance(todos, list):
        return None, "Error: todos must be a list"

    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"

    return todos, None


def run_todo_write(todos: list) -> str:
    """更新当前任务列表，并在终端中打印一个可读的任务面板。"""
    global CURRENT_TODOS

    todos, error = _normalize_todos(todos)
    if error:
        return error

    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {
            "pending": " ",
            "in_progress": "\033[36m*\033[0m",
            "completed": "\033[32m✓\033[0m",
        }[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")

    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"


# OpenAI/Qwen 工具声明格式：
# - 顶层 type 固定为 function
# - function.parameters 是 JSON Schema，描述工具参数
# 模型看到这些 schema 后，会自动决定是否生成 tool_calls。
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
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    # s05 新增工具：要求模型用结构化数组维护任务状态。
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Create and manage a task list for your current coding session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
]

# 工具名到本地 Python 函数的分发表。
TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
}


# ============================================================================
# Hook 系统：在用户提交、工具执行前后、结束时插入自定义逻辑
# ============================================================================

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    """给某个事件注册一个回调函数。"""
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """依次触发事件下的回调；任何回调返回非 None 都会中断后续流程。"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


@dataclass
class ToolBlock:
    """OpenAI tool_call 的轻量适配器，让 hook 继续用 block.name/block.input/block.id。"""

    id: str
    name: str
    input: dict


# 简单命令黑名单：教学用防护，不是完整安全沙箱。
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]


def permission_hook(block):
    """PreToolUse：工具执行前检查危险命令。"""
    if block.name == "bash":
        for p in DENY_LIST:
            if p in block.input.get("command", ""):
                print(f"\n\033[31mBlocked: '{p}'\033[0m")
                return "Permission denied"
    return None


def log_hook(block):
    """PreToolUse：打印即将执行的工具名，方便观察 agent 行为。"""
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None


def context_inject_hook(query: str):
    """UserPromptSubmit：用户输入提交时打印当前工作目录。"""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    """Stop：本轮停止前统计工具调用次数。"""
    tool_count = sum(1 for m in messages if m.get("role") == "tool")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("Stop", summary_hook)


# ============================================================================
# agent_loop：OpenAI/Qwen-compatible 工具调用循环 + todo 提醒计数器
# ============================================================================

# 记录连续多少轮工具调用没有更新 todo；超过阈值后插入提醒。
rounds_since_todo = 0


def _parse_tool_arguments(raw_arguments: str) -> dict:
    """OpenAI tool_call 参数是 JSON 字符串，这里解析成 Python dict。"""
    try:
        return json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return {"__json_error__": f"Invalid tool arguments JSON: {e}", "raw": raw_arguments}


def agent_loop(messages: list):
    """持续与模型交互，直到模型不再请求工具调用并返回最终文本。"""
    global rounds_since_todo

    while True:
        # 如果模型连续 3 轮工具调用都没更新 todo，就插入提醒消息。
        if rounds_since_todo >= 3 and messages:
            messages.append({
                "role": "user",
                "content": "<reminder>Update your todos.</reminder>",
            })
            rounds_since_todo = 0

        # Qwen/DashScope 的 OpenAI 兼容接口使用 chat.completions.create。
        # system 消息要放在 messages 列表最前面。
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # 没有 tool_calls 表示模型已经给出最终回答，本轮结束。
        if not tool_calls:
            content = message.content or ""
            messages.append({"role": "assistant", "content": content})

            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return content

        # 有 tool_calls 时，必须先把 assistant 的 tool_calls 原样追加到历史。
        # 后续每个工具结果都通过 role=tool + tool_call_id 关联回对应调用。
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in tool_calls],
            }
        )

        rounds_since_todo += 1
        for tc in tool_calls:
            # OpenAI/Qwen 返回的 function.arguments 是 JSON 字符串，需要先解析。
            name = tc.function.name
            args = _parse_tool_arguments(tc.function.arguments)
            block = ToolBlock(id=tc.id, name=name, input=args)

            # 如果模型生成了非法 JSON 参数，把错误作为工具结果反馈给模型。
            if "__json_error__" in args:
                output = args["__json_error__"]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                continue

            # 工具真正执行之前先跑 PreToolUse hooks，允许 hook 拦截。
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(blocked)})
                continue

            # 根据工具名找到本地 handler 并执行。
            handler = TOOL_HANDLERS.get(name)
            output = handler(**args) if handler else f"Unknown: {name}"

            # 工具执行后触发 PostToolUse hooks。
            trigger_hooks("PostToolUse", block, output)

            # 调用了 todo_write 就重置提醒计数。
            if name == "todo_write":
                rounds_since_todo = 0

            # OpenAI/Qwen 工具结果格式：role=tool，并带上 tool_call_id。
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)})


if __name__ == "__main__":
    print("s05: TodoWrite - plan before execute, nag if you forget")
    print("Type a question, press Enter. Type q to quit.\n")

    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        answer = agent_loop(history)
        if answer:
            print(answer)
        print()
