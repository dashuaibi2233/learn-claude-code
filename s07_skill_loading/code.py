#!/usr/bin/env python3
"""
s07: Skill Loading - 用到技能时再加载完整内容，不把所有知识都塞进 system prompt。

  Layer 1 (cheap, always present):
    SYSTEM prompt includes skill names + one-line descriptions (~100 tokens/skill)
    "Skills available: agent-builder, code-review, mcp-builder, pdf"

  Layer 2 (expensive, on demand):
    Agent calls load_skill("code-review") -> full SKILL.md content
    injected via tool result (~2000 tokens/skill)

Run: python s07_skill_loading/code.py
Needs: pip install openai python-dotenv pyyaml
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

import yaml
from dotenv import load_dotenv
from openai import OpenAI

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
except ImportError:
    pass

load_dotenv(override=True)

# 当前工作目录是 Agent 能操作的根目录；skills/ 也从这个目录下扫描。
WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"

# 千问 DashScope 提供 OpenAI-compatible 接口，所以这里使用 OpenAI SDK。
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = os.getenv("MODEL_ID", "qwen-plus")

# todo_write 只维护内存中的任务列表，程序退出后不会持久化。
CURRENT_TODOS: list[dict] = []


# ============================================================================
# s07 新增：技能目录扫描。启动时只注入“技能名 + 简短描述”
# ============================================================================

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 SKILL.md 顶部的 YAML frontmatter，返回 (元数据, 正文)。"""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()


# 技能注册表用于安全查找，load_skill 只按名称查表，不直接接收文件路径。
SKILL_REGISTRY: dict[str, dict] = {}


def _scan_skills():
    """扫描 skills/ 目录，把每个 SKILL.md 的名称、描述和完整内容放进注册表。"""
    if not SKILLS_DIR.exists():
        return

    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue

        manifest = d / "SKILL.md"
        if not manifest.exists():
            continue

        raw = manifest.read_text(encoding="utf-8")
        meta, _body = _parse_frontmatter(raw)
        name = meta.get("name", d.name)
        desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
        SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}


_scan_skills()


def list_skills() -> str:
    """列出所有可用技能的名称和一句话描述。"""
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())


def build_system() -> str:
    """构造 system prompt，把轻量技能目录注入进去。"""
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed. "
        "For complex sub-problems, use the task tool to spawn a subagent."
    )


SYSTEM = build_system()

# 子 Agent 使用独立 system prompt，不加载 skill，也不能继续派发 task。
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# ============================================================================
# 工具实现：这些 Python 函数是真正在本地执行的能力
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
# todo_write 计划工具：只记录计划，不直接执行任务
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
    """更新当前任务列表，并在终端打印一个可读的任务面板。"""
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


def load_skill(name: str) -> str:
    """按名称加载完整 SKILL.md。只查注册表，不接受路径，避免路径遍历。"""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]


# ============================================================================
# OpenAI/Qwen 工具声明：Qwen 兼容接口要求 type=function + function.parameters
# ============================================================================

def make_tool(name: str, description: str, parameters: dict) -> dict:
    """把简单的 name/schema 包成 OpenAI/Qwen 兼容的工具声明。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOLS = [
    make_tool(
        "bash",
        "Run a shell command.",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    ),
    make_tool(
        "read_file",
        "Read file contents.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    ),
    make_tool(
        "write_file",
        "Write content to a file.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    ),
    make_tool(
        "edit_file",
        "Replace exact text in a file once.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    ),
    make_tool(
        "glob",
        "Find files matching a glob pattern.",
        {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    ),
    make_tool(
        "todo_write",
        "Create and manage a task list for your current coding session.",
        {
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
    ),
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
}


# ============================================================================
# Subagent：独立 messages[]，只返回最终摘要
# ============================================================================

# 子 Agent 只拿基础工具，不拿 todo_write/load_skill/task，保持上下文隔离且禁止递归派发。
SUB_TOOLS = [tool for tool in TOOLS if tool["function"]["name"] not in {"todo_write"}]
SUB_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


@dataclass
class ToolBlock:
    """OpenAI tool_call 的轻量适配器，让 hook 继续使用 block.name/block.input/block.id。"""

    id: str
    name: str
    input: dict


def _parse_tool_arguments(raw_arguments: str) -> dict:
    """OpenAI/Qwen tool_call 参数是 JSON 字符串，这里解析成 Python dict。"""
    try:
        return json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return {"__json_error__": f"Invalid tool arguments JSON: {e}", "raw": raw_arguments}


def _dump_tool_call(tool_call) -> dict:
    """保留模型返回的 tool_call 结构，后续 role=tool 结果要靠 id 对齐。"""
    return tool_call.model_dump(exclude_none=True)


def extract_text(content) -> str:
    """从 OpenAI/Qwen 消息内容中取出文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def spawn_subagent(description: str) -> str:
    """启动一个拥有全新 messages[] 的子 Agent，并只把最终摘要返回给父 Agent。"""
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]
    last_answer = ""

    for _ in range(30):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUB_SYSTEM}] + messages,
            tools=SUB_TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            last_answer = extract_text(message.content)
            messages.append({"role": "assistant", "content": last_answer})
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [_dump_tool_call(tc) for tc in tool_calls],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            args = _parse_tool_arguments(tc.function.arguments)
            block = ToolBlock(id=tc.id, name=name, input=args)

            if "__json_error__" in args:
                output = args["__json_error__"]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                continue

            # 子 Agent 的工具调用同样经过 hook，权限策略不会因为上下文隔离而绕过。
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(blocked)})
                continue

            handler = SUB_HANDLERS.get(name)
            output = handler(**args) if handler else f"Unknown: {name}"
            trigger_hooks("PostToolUse", block, output)

            print(f"  \033[90m[sub] {name}: {str(output)[:100]}\033[0m")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)})

    if not last_answer:
        # 如果 30 轮安全上限被打满，尽量回退到最近一次 assistant 文本。
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_answer = extract_text(msg["content"])
                break
        if not last_answer:
            last_answer = "Subagent stopped after 30 turns without final answer."

    print(f"\033[35m[Subagent done]\033[0m")
    return last_answer


# 父 Agent 额外获得 task 和 load_skill；子 Agent 因为 SUB_TOOLS 已经创建，不会拿到这两个工具。
TOOLS.append(
    make_tool(
        "task",
        "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
        {
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"],
        },
    )
)
TOOLS.append(
    make_tool(
        "load_skill",
        "Load the full content of a skill by name.",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
)
TOOL_HANDLERS["task"] = spawn_subagent
TOOL_HANDLERS["load_skill"] = load_skill


# ============================================================================
# Hook 系统：在用户提交、工具执行前后、停止时插入自定义逻辑
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
    """PreToolUse：打印即将执行的工具名，方便观察 Agent 行为。"""
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

rounds_since_todo = 0


def agent_loop(messages: list):
    """持续与模型交互，直到模型不再请求工具调用并返回最终文本。"""
    global rounds_since_todo

    while True:
        # 连续 3 轮工具调用都没更新 todo，就插入提醒消息。
        if rounds_since_todo >= 3 and messages:
            messages.append({
                "role": "user",
                "content": "<reminder>Update your todos.</reminder>",
            })
            rounds_since_todo = 0

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            content = extract_text(message.content)
            messages.append({"role": "assistant", "content": content})

            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return content

        # Qwen/OpenAI 工具调用必须先记录 assistant 的 tool_calls，再追加 role=tool 的结果。
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [_dump_tool_call(tc) for tc in tool_calls],
            }
        )

        rounds_since_todo += 1
        for tc in tool_calls:
            name = tc.function.name
            args = _parse_tool_arguments(tc.function.arguments)
            block = ToolBlock(id=tc.id, name=name, input=args)

            if "__json_error__" in args:
                output = args["__json_error__"]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                continue

            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(blocked)})
                continue

            handler = TOOL_HANDLERS.get(name)
            output = handler(**args) if handler else f"Unknown: {name}"

            trigger_hooks("PostToolUse", block, output)

            if name == "todo_write":
                rounds_since_todo = 0

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)})


if __name__ == "__main__":
    print("s07: Skill Loading - catalog in SYSTEM, content on demand")
    print("Type a question, press Enter. Type q to quit.\n")

    history = []
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
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
