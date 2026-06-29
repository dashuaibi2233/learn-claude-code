#!/usr/bin/env python3
"""
s15: Agent Teams - MessageBus + teammate threads with Qwen/OpenAI-compatible tools.

Run:  python s15_agent_teams/code.py
Need: pip install openai python-dotenv

.env example:
    DASHSCOPE_API_KEY=sk-xxxx
    QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    MODEL_ID=qwen-plus

Changes from s14:
  - MessageBus class: file-based mailboxes (.mailboxes/*.jsonl)
  - spawn_teammate_thread: creates teammate in background thread
  - Teammate runs own simplified agent_loop (bash, read, write, send_message)
  - Lead tools: spawn_teammate, send_message, check_inbox
  - Lead inbox: teammate messages injected into history
  - Teaching version: teammates limited to 10 rounds
"""

import json
import os
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
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


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


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
        "Available tools: bash, read_file, write_file, get_task, create_task, "
        "list_tasks, claim_task, complete_task, schedule_cron, list_crons, "
        "cancel_cron, spawn_teammate, send_message, check_inbox. "
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


# Task Tool Handlers

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


# Cron Scheduler

DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"


@dataclass
class CronJob:
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool


scheduled_jobs: dict[str, CronJob] = {}
cron_queue: list[CronJob] = []
cron_lock = threading.Lock()
_last_fired: dict[str, str] = {}


def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(_cron_field_matches(part.strip(), value) for part in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7

    minute_ok = _cron_field_matches(minute, dt.minute)
    hour_ok = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    if not (minute_ok and hour_ok and month_ok):
        return False

    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None
    if "-" in field:
        start, end = field.split("-", 1)
        if not start.isdigit() or not end.isdigit():
            return f"Invalid range: {field}"
        a, b = int(start), int(end)
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < lo or value > hi:
        return f"Value {value} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"

    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for field, (lo, hi), name in zip(fields, bounds, names):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    durable = [asdict(job) for job in scheduled_jobs.values() if job.durable]
    DURABLE_PATH.write_text(
        json.dumps(durable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_durable_jobs():
    if not DURABLE_PATH.exists():
        return

    try:
        jobs = json.loads(read_text_with_fallback(DURABLE_PATH))
        for item in jobs:
            job = CronJob(**item)
            err = validate_cron(job.cron)
            if err:
                print(f"  \033[31m[cron] skipping invalid job {job.id}: {err}\033[0m")
                continue
            scheduled_jobs[job.id] = job
        valid = [job for job in jobs if job["id"] in scheduled_jobs]
        if valid:
            print(f"  \033[35m[cron] loaded {len(valid)} durable job(s)\033[0m")
    except Exception:
        pass


def schedule_job(
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = True,
) -> CronJob | str:
    err = validate_cron(cron)
    if err:
        return err

    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron,
        prompt=prompt,
        recurring=recurring,
        durable=durable,
    )
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()
    print(f"  \033[35m[cron register] {job.id} '{cron}' -> {prompt[:40]}\033[0m")
    return job


def cancel_job(job_id: str) -> str:
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()
    print(f"  \033[31m[cron cancel] {job_id}\033[0m")
    return f"Cancelled {job_id}"


def cron_scheduler_loop():
    while True:
        time.sleep(1)
        now = datetime.now()
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)
                            _last_fired[job.id] = minute_marker
                            print(f"  \033[35m[cron fire] {job.id} -> {job.prompt[:40]}\033[0m")
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:
                    print(f"  \033[31m[cron error] {job.id}: {e}\033[0m")


def consume_cron_queue() -> list[CronJob]:
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


def run_schedule_cron(
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = True,
) -> str:
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' -> {prompt}"


def run_list_crons() -> str:
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs. Use schedule_cron to add one."

    lines = []
    for job in jobs:
        tag = "recurring" if job.recurring else "one-shot"
        durable = "durable" if job.durable else "session"
        lines.append(f"  {job.id}: '{job.cron}' -> {job.prompt[:40]} [{tag}, {durable}]")
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    return cancel_job(job_id)


# MessageBus

MAILBOX_DIR = WORKDIR / ".mailboxes"
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
    ):
        msg = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "ts": time.time(),
        }
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a", encoding="utf-8") as file:
            file.write(json.dumps(msg, ensure_ascii=False) + "\n")
        print(f"  \033[33m[bus] {from_agent} -> {to_agent}: {content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [
            json.loads(line)
            for line in read_text_with_fallback(inbox).splitlines()
            if line.strip()
        ]
        inbox.unlink()
        return msgs


BUS = MessageBus()
active_teammates: dict[str, bool] = {}


def qwen_tool(
    name: str,
    description: str,
    properties: dict | None = None,
    required: list[str] | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def _dump_tool_call(tool_call) -> dict:
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    system = (
        f"You are '{name}', a {role}. "
        "Use tools to complete tasks. "
        "Send results via send_message to 'lead'."
    )

    def run():
        messages = [{"role": "user", "content": prompt}]
        sub_tools = [
            qwen_tool(
                "bash",
                "Run a shell command.",
                {"command": {"type": "string"}},
                ["command"],
            ),
            qwen_tool(
                "read_file",
                "Read file contents.",
                {"path": {"type": "string"}},
                ["path"],
            ),
            qwen_tool(
                "write_file",
                "Write content to a file.",
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            ),
            qwen_tool(
                "send_message",
                "Send a message to another agent.",
                {"to": {"type": "string"}, "content": {"type": "string"}},
                ["to", "content"],
            ),
        ]
        sub_handlers = {
            "bash": run_bash,
            "read_file": run_read,
            "write_file": run_write,
            "send_message": lambda to, content: (
                BUS.send(name, to, content),
                "Sent",
            )[1],
        }

        for _ in range(10):
            inbox = BUS.read_inbox(name)
            if inbox:
                messages.append({
                    "role": "user",
                    "content": f"<inbox>{json.dumps(inbox, ensure_ascii=False)}</inbox>",
                })

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system}] + messages[-20:],
                    tools=sub_tools,
                    max_tokens=8000,
                )
            except Exception as e:
                BUS.send(name, "lead", f"[Error] {type(e).__name__}: {e}", "error")
                break

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
                break

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    output = f"Error: invalid JSON arguments: {e}"
                else:
                    handler = sub_handlers.get(tool_name)
                    if handler is None:
                        output = f"Unknown tool: {tool_name}"
                    else:
                        try:
                            output = handler(**args)
                        except TypeError as e:
                            output = f"Error: bad tool arguments: {e}"
                        except Exception as e:
                            output = f"Error: tool execution failed: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                })

        summary = "Done."
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                summary = msg["content"]
                break
        BUS.send(name, "lead", summary, "result")
        active_teammates.pop(name, None)
        print(f"  \033[32m[teammate] {name} finished\033[0m")

    active_teammates[name] = True
    threading.Thread(target=run, daemon=True).start()
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m")
    return f"Teammate '{name}' spawned as {role}"


def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    BUS.send("lead", to, content)
    return f"Sent to {to}"


def run_check_inbox() -> str:
    msgs = BUS.read_inbox("lead")
    if not msgs:
        return "(inbox empty)"
    return "\n".join(f"  [{msg['from']}] {msg['content'][:200]}" for msg in msgs)


# Tool Registry

TOOLS = [
    qwen_tool(
        "bash",
        "Run a shell command.",
        {
            "command": {"type": "string"},
            "run_in_background": {"type": "boolean"},
        },
        ["command"],
    ),
    qwen_tool(
        "read_file",
        "Read file contents.",
        {"path": {"type": "string"}, "limit": {"type": "integer"}},
        ["path"],
    ),
    qwen_tool(
        "write_file",
        "Write content to a file.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    qwen_tool(
        "create_task",
        "Create a new task with optional blockedBy dependencies.",
        {
            "subject": {"type": "string"},
            "description": {"type": "string"},
            "blockedBy": {"type": "array", "items": {"type": "string"}},
        },
        ["subject"],
    ),
    qwen_tool(
        "list_tasks",
        "List all tasks with status, owner, and dependencies.",
    ),
    qwen_tool(
        "get_task",
        "Get full details of a specific task by ID.",
        {"task_id": {"type": "string"}},
        ["task_id"],
    ),
    qwen_tool(
        "claim_task",
        "Claim a pending task. Sets owner and changes status to in_progress.",
        {"task_id": {"type": "string"}},
        ["task_id"],
    ),
    qwen_tool(
        "complete_task",
        "Complete an in-progress task and report unblocked downstream tasks.",
        {"task_id": {"type": "string"}},
        ["task_id"],
    ),
    qwen_tool(
        "schedule_cron",
        "Schedule a cron job. cron is 5-field: min hour dom month dow.",
        {
            "cron": {"type": "string", "description": "5-field cron expression"},
            "prompt": {"type": "string", "description": "Message to inject when fired"},
            "recurring": {"type": "boolean", "description": "True=recurring, False=one-shot"},
            "durable": {"type": "boolean", "description": "True=persist to disk"},
        },
        ["cron", "prompt"],
    ),
    qwen_tool("list_crons", "List all registered cron jobs."),
    qwen_tool(
        "cancel_cron",
        "Cancel a cron job by ID.",
        {"job_id": {"type": "string"}},
        ["job_id"],
    ),
    qwen_tool(
        "spawn_teammate",
        "Spawn a teammate agent in a background thread.",
        {
            "name": {"type": "string"},
            "role": {"type": "string"},
            "prompt": {"type": "string"},
        },
        ["name", "role", "prompt"],
    ),
    qwen_tool(
        "send_message",
        "Send a message to a teammate via MessageBus.",
        {"to": {"type": "string"}, "content": {"type": "string"}},
        ["to", "content"],
    ),
    qwen_tool("check_inbox", "Check Lead's inbox for teammate messages."),
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
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
    "spawn_teammate": run_spawn_teammate,
    "send_message": run_send_message,
    "check_inbox": run_check_inbox,
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

    threading.Thread(target=worker, daemon=True).start()
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


def _append_background_notifications(messages: list):
    notifications = collect_background_results()
    if not notifications:
        return

    messages.append({
        "role": "user",
        "content": "\n\n".join(notifications),
    })
    print(f"  \033[32m[inject] {len(notifications)} background notification(s)\033[0m")


# Context and Agent Loop

def update_context(context: dict, messages: list) -> dict:
    memories = ""
    if MEMORY_INDEX.exists():
        content = read_text_with_fallback(MEMORY_INDEX).strip()
        if content:
            memories = content

    return {
        "enabled_tools": [tool["function"]["name"] for tool in TOOLS],
        "workspace": str(WORKDIR),
        "memories": memories,
    }


def _append_cron_messages(messages: list):
    fired = consume_cron_queue()
    for job in fired:
        messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
        print(f"  \033[35m[inject cron] {job.prompt[:50]}\033[0m")


def _append_lead_inbox(messages: list):
    inbox = BUS.read_inbox("lead")
    if not inbox:
        return
    inbox_text = "\n".join(f"From {msg['from']}: {msg['content'][:200]}" for msg in inbox)
    messages.append({"role": "user", "content": f"[Inbox]\n{inbox_text}"})
    print(f"  \033[33m[inject inbox] {len(inbox)} message(s)\033[0m")


def agent_loop(messages: list, context: dict) -> str:
    system = get_system_prompt(context)

    while True:
        _append_cron_messages(messages)
        _append_background_notifications(messages)
        _append_lead_inbox(messages)

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

        context = update_context(context, messages)
        system = get_system_prompt(context)


load_durable_jobs()
threading.Thread(target=cron_scheduler_loop, daemon=True).start()
print("  \033[35m[cron] scheduler thread started\033[0m")


if __name__ == "__main__":
    print("s15: agent teams - Qwen")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    context = update_context({}, [])

    while True:
        try:
            query = input("\033[36ms15 >> \033[0m")
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
