#!/usr/bin/env python3
"""
s13: Background Tasks - thread-based async execution with Qwen/OpenAI-compatible tools.

Run:  python s13_background_tasks/code.py
Need: pip install openai python-dotenv

.env example:
    DASHSCOPE_API_KEY=sk-xxxx
    QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    MODEL_ID=qwen-plus

Changes from s12:
  - threading.Thread for background execution
  - background_tasks dict for lifecycle tracking (bg_id, command, status)
  - background_results dict + threading.Lock for thread-safe storage
  - should_run_background: model explicit request via run_in_background param
  - is_slow_operation: fallback heuristic when model doesn't specify
  - start_background_task: dispatch to daemon thread, return bg task id
  - collect_background_results: gather completed, return notifications
  - agent_loop: slow ops -> background + placeholder, inject notifications

Note: Teaching code keeps a basic agent loop to stay focused on background
tasks. S11's full error recovery is omitted.
"""

import json
import os
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
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

WORKDIR = Path.cwd()
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=(
        os.getenv("QWEN_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)
MODEL = os.environ.get("MODEL_ID", "qwen-plus")


# Task System

TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]


def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def create_task(
    subject: str,
    description: str = "",
    blockedBy: list[str] | None = None,
) -> Task:
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject,
        description=description,
        status="pending",
        owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    _task_path(task.id).write_text(
        json.dumps(asdict(task), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_task(task_id: str) -> Task:
    return Task(**json.loads(_task_path(task_id).read_text(encoding="utf-8")))


def list_tasks() -> list[Task]:
    return [
        Task(**json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(TASKS_DIR.glob("task_*.json"))
    ]


def get_task(task_id: str) -> str:
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2, ensure_ascii=False)


def can_start(task_id: str) -> bool:
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"

    if not can_start(task_id):
        deps = [
            dep_id
            for dep_id in task.blockedBy
            if not _task_path(dep_id).exists()
            or load_task(dep_id).status != "completed"
        ]
        return f"Blocked by: {deps}"

    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} -> in_progress (owner: {owner})\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"

    task.status = "completed"
    save_task(task)

    unblocked = [
        item.subject
        for item in list_tasks()
        if item.status == "pending" and item.blockedBy and can_start(item.id)
    ]

    print(f"  \033[32m[complete] {task.subject}\033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")
    return msg


# Prompt Assembly

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": (
        "Available tools: bash, read_file, write_file, create_task, "
        "list_tasks, get_task, claim_task, complete_task. "
        "The bash tool supports run_in_background=true for long-running commands."
    ),
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    sections = [
        PROMPT_SECTIONS["identity"],
        PROMPT_SECTIONS["tools"],
        PROMPT_SECTIONS["workspace"],
    ]
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)


_last_context_key = None
_last_prompt = None


def get_system_prompt(context: dict) -> str:
    global _last_context_key, _last_prompt

    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        return _last_prompt

    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


# Basic File Tools

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str, run_in_background: bool = False) -> str:
    # run_in_background is handled by agent_loop dispatch, not here.
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (result.stdout + result.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# Task Tools

def run_create_task(
    subject: str,
    description: str = "",
    blockedBy: list[str] | None = None,
) -> str:
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."

    markers = {"pending": "-", "in_progress": ">", "completed": "x"}
    lines = []
    for task in tasks:
        marker = markers.get(task.status, "?")
        deps = f" (blockedBy: {', '.join(task.blockedBy)})" if task.blockedBy else ""
        owner = f" [{task.owner}]" if task.owner else ""
        lines.append(
            f"  {marker} {task.id}: {task.subject} "
            f"[{task.status}]{owner}{deps}"
        )
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    try:
        return claim_task(task_id, owner="agent")
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_complete_task(task_id: str) -> str:
    try:
        return complete_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


# Qwen/OpenAI-compatible Tool Definitions

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "run_in_background": {"type": "boolean"},
                },
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
            "name": "create_task",
            "description": "Create a new task with optional blockedBy dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "blockedBy": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all tasks with status, owner, and dependencies.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task",
            "description": "Get full details of a specific task by ID.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "Claim a pending task. Sets owner and changes status to in_progress.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Complete an in-progress task and report unblocked downstream tasks.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "create_task": run_create_task,
    "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task,
    "complete_task": run_complete_task,
}


# Background Tasks

_bg_counter = 0
background_tasks: dict[str, dict] = {}
background_results: dict[str, str] = {}
background_lock = threading.Lock()


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    if tool_name != "bash":
        return False

    command = tool_input.get("command", "").lower()
    slow_keywords = [
        "install",
        "build",
        "test",
        "deploy",
        "compile",
        "docker build",
        "pip install",
        "npm install",
        "cargo build",
        "pytest",
        "make",
    ]
    return any(keyword in command for keyword in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    if tool_input.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, tool_input)


def execute_tool(tool_name: str, tool_input: dict) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Unknown tool: {tool_name}"

    try:
        return handler(**tool_input)
    except TypeError as e:
        return f"Error: bad tool arguments: {e}"
    except Exception as e:
        return f"Error: tool execution failed: {e}"


def start_background_task(tool_name: str, tool_call_id: str, tool_input: dict) -> str:
    global _bg_counter

    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    command = tool_input.get("command", tool_name)

    def worker():
        result = execute_tool(tool_name, tool_input)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_call_id": tool_call_id,
            "command": command,
            "status": "running",
        }

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    print(f"  \033[33m[background] dispatched {bg_id}: {command[:40]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    with background_lock:
        ready_ids = [
            bg_id
            for bg_id, task in background_tasks.items()
            if task["status"] == "completed"
        ]

    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")

        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            "<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            "  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            "</task_notification>"
        )
        print(
            f"  \033[32m[background done] {bg_id}: "
            f"{task['command'][:40]} ({len(output)} chars)\033[0m"
        )

    return notifications


# Context

def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def update_context(context: dict, messages: list) -> dict:
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


# Agent Loop - Qwen/OpenAI-compatible

def _dump_tool_call(tool_call) -> dict:
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def _append_background_notifications(messages: list):
    notifications = collect_background_results()
    if not notifications:
        return

    messages.append({
        "role": "user",
        "content": "\n\n".join(notifications),
    })
    print(f"  \033[32m[inject] {len(notifications)} background notification(s)\033[0m")


def agent_loop(messages: list, context: dict) -> str:
    system = get_system_prompt(context)

    while True:
        _append_background_notifications(messages)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}] + messages,
                tools=TOOLS,
                max_tokens=8000,
            )
        except Exception as e:
            return f"[Error] {type(e).__name__}: {e}"

        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

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

            if should_run_background(tool_name, args):
                bg_id = start_background_task(tool_name, tool_call.id, args)
                output = (
                    f"[Background task {bg_id} started] "
                    f"Command: {args.get('command', '')}. "
                    "Result will be injected as a task_notification when complete."
                )
            else:
                output = execute_tool(tool_name, args)
                print(str(output)[:300])

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })

        _append_background_notifications(messages)
        context = update_context(context, messages)
        system = get_system_prompt(context)


if __name__ == "__main__":
    print("s13: background tasks - Qwen")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    context = update_context({}, [])

    while True:
        try:
            query = input("\033[36ms13 >> \033[0m")
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
