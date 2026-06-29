#!/usr/bin/env python3
"""
s08: Context Compact - 千问 / DashScope OpenAI-compatible 版本

本节演示在每次调用大模型前插入四层上下文压缩管线：

    L1: snip_compact       消息数量过多时，裁掉中间历史
    L2: micro_compact      把较早的工具结果替换成占位符
    L3: tool_result_budget 大工具结果落盘，只把预览塞回上下文
    L4: compact_history    调用一次 LLM，把历史总结成短摘要

    Emergency: reactive_compact
    当 API 仍然报 prompt too long / context length 超限时，做兜底压缩后重试。

核心原则：便宜的压缩先做，昂贵的 LLM 总结最后做。
"""

import ast
import json
import os
import subprocess
import time
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

# 当前工作目录就是 agent 能访问的根目录，所有文件读写都要被限制在这里。
WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"

# 千问 / DashScope 提供 OpenAI-compatible 接口，因此可以直接使用 OpenAI SDK。
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = os.getenv("MODEL_ID", "qwen-plus")

# todo_write 只维护当前进程内的任务列表，程序退出后不会持久化。
CURRENT_TODOS: list[dict] = []


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 SKILL.md 顶部的 YAML 风格 frontmatter。"""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, parts[2].strip()


SKILL_REGISTRY: dict[str, dict] = {}


def _scan_skills():
    """扫描 skills/ 下的 SKILL.md，把技能名、描述和完整内容登记起来。"""
    if not SKILLS_DIR.exists():
        return

    for directory in sorted(SKILLS_DIR.iterdir()):
        if not directory.is_dir():
            continue

        manifest = directory / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text(encoding="utf-8")
            meta, _ = _parse_frontmatter(raw)
            name = meta.get("name", directory.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}


_scan_skills()


def list_skills() -> str:
    """返回给 system prompt 的轻量技能目录。"""
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())


def load_skill(name: str) -> str:
    """按名称加载某个 skill 的完整内容。"""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]


def build_system() -> str:
    """运行时组装 system prompt，避免把所有 skill 全量塞进上下文。"""
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )


SYSTEM = build_system()

# 子 agent 使用独立 system prompt；它没有压缩工具，也不会继续创建子 agent。
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


def safe_path(path_text: str) -> Path:
    """把路径限制在当前工作目录内，防止读写逃出项目目录。"""
    path = (WORKDIR / path_text).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_text}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令，并把 stdout/stderr 合并后返回给模型。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容；limit 可限制返回行数，避免一次塞入太多上下文。"""
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def run_write(path: str, content: str) -> str:
    """写入完整文件内容；父目录不存在时自动创建。"""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """做一次精确文本替换，只替换第一次出现的 old_text。"""
    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_glob(pattern: str) -> str:
    """按 glob 模式查找文件，例如 **/*.py。"""
    import glob as glob_module

    try:
        results = []
        for match in glob_module.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as exc:
        return f"Error: {exc}"


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

    for index, todo in enumerate(todos):
        if not isinstance(todo, dict):
            return None, f"Error: todos[{index}] must be an object"
        if "content" not in todo or "status" not in todo:
            return None, f"Error: todos[{index}] missing 'content' or 'status'"
        if todo["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{index}] has invalid status '{todo['status']}'"

    return todos, None


def run_todo_write(todos: list) -> str:
    """更新当前任务列表，并在终端打印可读的任务面板。"""
    global CURRENT_TODOS

    todos, error = _normalize_todos(todos)
    if error:
        return error

    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for todo in CURRENT_TODOS:
        icon = {
            "pending": " ",
            "in_progress": "\033[36m*\033[0m",
            "completed": "\033[32m✓\033[0m",
        }[todo["status"]]
        lines.append(f"  [{icon}] {todo['content']}")

    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"


def qwen_tool(name: str, description: str, parameters: dict) -> dict:
    """把简洁的函数定义包装成 OpenAI/Qwen-compatible tool schema。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


SUB_TOOLS = [
    qwen_tool(
        "bash",
        "Run a shell command.",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    ),
    qwen_tool(
        "read_file",
        "Read file contents.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    ),
    qwen_tool(
        "write_file",
        "Write content to a file.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
    qwen_tool(
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
    qwen_tool(
        "glob",
        "Find files matching a glob pattern.",
        {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    ),
]


TOOLS = [
    *SUB_TOOLS,
    qwen_tool(
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
                }
            },
            "required": ["todos"],
        },
    ),
    qwen_tool(
        "task",
        "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
        {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
    ),
    qwen_tool(
        "load_skill",
        "Load the full content of a skill by name.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    ),
    qwen_tool(
        "compact",
        "Summarize earlier conversation to free context space.",
        {"type": "object", "properties": {"focus": {"type": "string"}}},
    ),
]


SUB_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

TOOL_HANDLERS = {
    **SUB_HANDLERS,
    "todo_write": run_todo_write,
    "task": lambda description: spawn_subagent(description),
    "load_skill": load_skill,
}


@dataclass
class ToolBlock:
    """OpenAI tool_call 的轻量适配器，让 hook 继续使用 block.name/input/id。"""

    id: str
    name: str
    input: dict


def _parse_tool_arguments(raw_arguments: str) -> dict:
    """OpenAI/Qwen 的 function.arguments 是 JSON 字符串，这里解析成 dict。"""
    try:
        return json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return {"__json_error__": f"Invalid tool arguments JSON: {exc}", "raw": raw_arguments}


def _dump_tool_call(tool_call) -> dict:
    """把 SDK 返回的 tool_call 对象转成可放进 messages 的普通 dict。"""
    return tool_call.model_dump(exclude_none=True)


HOOKS = {"PreToolUse": [], "PostToolUse": []}


def trigger_hooks(event: str, *args):
    """依次触发某个事件下的回调；任意回调返回非 None 就中断后续流程。"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]


def permission_hook(block: ToolBlock):
    """PreToolUse：执行 bash 前做一个教学版危险命令拦截。"""
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                return "Permission denied"
    return None


def log_hook(block: ToolBlock):
    """PreToolUse：打印即将执行的工具名，方便观察 agent 行为。"""
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None


HOOKS["PreToolUse"].append(permission_hook)
HOOKS["PreToolUse"].append(log_hook)


def spawn_subagent(task: str) -> str:
    """启动一个轻量子 agent，最后只返回总结文本给主 agent。"""
    print("\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": task}]

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
            result = message.content or ""
            print("\033[35m[Subagent done]\033[0m")
            return result or "(empty subagent response)"

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [_dump_tool_call(tc) for tc in tool_calls],
            }
        )

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = _parse_tool_arguments(tool_call.function.arguments)
            block = ToolBlock(id=tool_call.id, name=name, input=args)

            if "__json_error__" in args:
                output = args["__json_error__"]
            else:
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    output = str(blocked)
                else:
                    handler = SUB_HANDLERS.get(name)
                    output = handler(**args) if handler else f"Unknown: {name}"
                    trigger_hooks("PostToolUse", block, output)

            print(f"  \033[90m[sub] {name}: {str(output)[:100]}\033[0m")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(output)})

    print("\033[35m[Subagent done]\033[0m")
    return "Subagent stopped after 30 turns without final answer."


CONTEXT_LIMIT = 50000
KEEP_RECENT = 3
PERSIST_THRESHOLD = 30000
MAX_REACTIVE_RETRIES = 1


def estimate_size(messages: list) -> int:
    """教学版用字符串长度粗略估算上下文大小。"""
    return len(json.dumps(messages, ensure_ascii=False, default=str))


def _message_has_tool_calls(message: dict) -> bool:
    """判断 assistant 消息是否携带 OpenAI/Qwen tool_calls。"""
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def _is_tool_message(message: dict) -> bool:
    """OpenAI/Qwen 工具结果格式是 role=tool，并通过 tool_call_id 关联调用。"""
    return message.get("role") == "tool"


def _is_tool_pair_boundary(messages: list, index: int) -> bool:
    """判断裁剪边界是否落在 assistant tool_calls 与后续 tool 结果之间。"""
    if index <= 0 or index >= len(messages):
        return False
    return _message_has_tool_calls(messages[index - 1]) and _is_tool_message(messages[index])


def snip_compact(messages: list, max_messages: int = 50) -> list:
    """L1：消息太多时裁掉中间部分，同时尽量不切断工具调用和工具结果。"""
    if len(messages) <= max_messages:
        return messages

    keep_head = 3
    keep_tail = max_messages - keep_head
    head_end = keep_head
    tail_start = len(messages) - keep_tail

    while _is_tool_pair_boundary(messages, head_end):
        head_end += 1

    if _is_tool_pair_boundary(messages, tail_start):
        tail_start -= 1

    if head_end >= tail_start:
        return messages

    snipped = tail_start - head_end
    marker = {"role": "user", "content": f"[snipped {snipped} messages]"}
    return messages[:head_end] + [marker] + messages[tail_start:]


def collect_tool_results(messages: list) -> list[tuple[int, dict]]:
    """收集所有 role=tool 的工具结果消息。"""
    return [(index, message) for index, message in enumerate(messages) if _is_tool_message(message)]


def micro_compact(messages: list) -> list:
    """L2：保留最近几个工具结果，较早的大结果替换成占位符。"""
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT:
        return messages

    for _, message in tool_results[:-KEEP_RECENT]:
        content = str(message.get("content", ""))
        if len(content) > 120:
            message["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


def persist_large_output(tool_call_id: str, output: str) -> str:
    """把超大工具输出写入磁盘，只把路径和预览留在上下文里。"""
    if len(output) <= PERSIST_THRESHOLD:
        return output

    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_call_id}.txt"
    if not path.exists():
        path.write_text(output, encoding="utf-8")

    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"


def tool_result_budget(messages: list, max_bytes: int = 200_000) -> list:
    """L3：如果最近一批工具结果总量过大，优先把最大结果落盘。"""
    last_tool_indices = []
    for index in range(len(messages) - 1, -1, -1):
        if _is_tool_message(messages[index]):
            last_tool_indices.append(index)
            continue
        break

    if not last_tool_indices:
        return messages

    blocks = [(index, messages[index]) for index in reversed(last_tool_indices)]
    total = sum(len(str(message.get("content", ""))) for _, message in blocks)
    if total <= max_bytes:
        return messages

    ranked = sorted(blocks, key=lambda pair: len(str(pair[1].get("content", ""))), reverse=True)
    for _, message in ranked:
        if total <= max_bytes:
            break

        content = str(message.get("content", ""))
        if len(content) <= PERSIST_THRESHOLD:
            continue

        tool_call_id = message.get("tool_call_id", "unknown")
        message["content"] = persist_large_output(tool_call_id, content)
        total = sum(len(str(message.get("content", ""))) for _, message in blocks)

    return messages


def write_transcript(messages: list) -> Path:
    """把完整历史写入 .transcripts，压缩后仍可回查原始上下文。"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    """L4：调用一次模型，把历史压成可继续工作的摘要。"""
    conversation = json.dumps(messages, ensure_ascii=False, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
        "4. remaining work, 5. user constraints.\n"
        "Be compact but concrete.\n\n"
        + conversation
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return (response.choices[0].message.content or "").strip() or "(empty summary)"


def compact_history(messages: list) -> list:
    """主动压缩：保存 transcript，再把消息历史替换为一条摘要消息。"""
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


def reactive_compact(messages: list) -> list:
    """应急压缩：保留摘要和最近几条消息，用来处理 API 上下文超限。"""
    write_transcript(messages)
    summary = summarize_history(messages)
    tail_start = max(0, len(messages) - 5)

    if _is_tool_pair_boundary(messages, tail_start):
        tail_start -= 1

    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]


def _is_context_too_long_error(exc: Exception) -> bool:
    """兼容不同服务商的上下文超限错误文案。"""
    text = str(exc).lower()
    needles = [
        "prompt_too_long",
        "too many tokens",
        "context length",
        "maximum context",
        "input token",
    ]
    return any(needle in text for needle in needles)


def agent_loop(messages: list) -> str:
    """持续调用模型，直到模型不再请求工具并返回最终文本。"""
    reactive_retries = 0

    while True:
        messages[:] = tool_result_budget(messages)
        messages[:] = snip_compact(messages)
        messages[:] = micro_compact(messages)

        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM}] + messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=8000,
            )
            reactive_retries = 0
        except Exception as exc:
            if _is_context_too_long_error(exc) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                messages[:] = reactive_compact(messages)
                reactive_retries += 1
                continue
            raise

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            content = message.content or ""
            messages.append({"role": "assistant", "content": content})
            return content

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [_dump_tool_call(tc) for tc in tool_calls],
            }
        )

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = _parse_tool_arguments(tool_call.function.arguments)
            block = ToolBlock(id=tool_call.id, name=name, input=args)
            print(f"\033[36m> {name}\033[0m")

            if "__json_error__" in args:
                output = args["__json_error__"]
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
                continue

            if name == "compact":
                messages[:] = compact_history(messages)
                output = "[Compacted. Conversation history has been summarized.]"
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})
                break

            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                output = str(blocked)
            else:
                handler = TOOL_HANDLERS.get(name)
                output = handler(**args) if handler else f"Unknown: {name}"
                trigger_hooks("PostToolUse", block, output)

            print(str(output)[:200])
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(output)})


if __name__ == "__main__":
    print("s08: Context Compact - 千问/OpenAI-compatible 四层压缩管线")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        answer = agent_loop(history)
        if answer:
            print(answer)
        print()
