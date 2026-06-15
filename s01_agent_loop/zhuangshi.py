#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

Usage:
    pip install openai python-dotenv langchain-core
    python s01_agent_loop/code.py
"""

import os
import json
import subprocess

from openai import OpenAI
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass


load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

MODEL = os.getenv("MODEL_ID", "qwen-plus")

if os.name == "nt":
    SHELL_NAME = "Windows cmd"
else:
    SHELL_NAME = "bash"

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    f"Use {SHELL_NAME} commands to solve tasks. "
    f"Act, don't explain."
)


# ── LangChain Tool definitions ───────────────────────────


@tool
def bash(command: str) -> str:
    """Run a shell command in the current working directory."""
    dangerous = ["sudo", "shutdown", "reboot", "> /dev/"]

    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )

        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


@tool
def read_file(path: str) -> str:
    """Read a text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:50000]
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written successfully: {path}"
    except Exception as e:
        return f"Error: {e}"


# ── Tool registry ────────────────────────────────────────

LANGCHAIN_TOOLS = [
    bash,
    read_file,
    write_file,
]

# 传给 OpenAI / DashScope 的 tools 格式
TOOLS = [convert_to_openai_tool(t) for t in LANGCHAIN_TOOLS]

# 本地执行时用：名字 -> LangChain Tool 对象
TOOL_MAP = {t.name: t for t in LANGCHAIN_TOOLS}


# ── Tool dispatcher ──────────────────────────────────────


def execute_tool_call(tool_call) -> str:
    """
    执行模型返回的一个 tool_call。
    """

    name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON arguments: {e}"

    selected_tool = TOOL_MAP.get(name)

    if selected_tool is None:
        return f"Error: unknown tool {name}"

    try:
        print(f"\033[33m$ tool: {name}({args})\033[0m")

        # LangChain Tool 的标准调用方式
        result = selected_tool.invoke(args)

        if result is None:
            return "(no output)"

        return str(result)

    except Exception as e:
        return f"Error while running tool {name}: {e}"


# ── Agent loop ───────────────────────────────────────────


def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        msg = response.choices[0].message

        # assistant 消息先放回 history
        messages.append(msg.model_dump(exclude_none=True))

        # 没有 tool_calls，说明模型回答完了
        if not msg.tool_calls:
            return

        # 执行所有工具调用
        for tool_call in msg.tool_calls:
            output = execute_tool_call(tool_call)

            print(output[:200])

            # 工具结果放回 history
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                }
            )


# ── Entry point ──────────────────────────────────────────


if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")
    print(f"当前目录: {os.getcwd()}")
    print(f"当前模型: {MODEL}")
    print(f"当前 Shell: {SHELL_NAME}\n")

    history = []

    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})

        agent_loop(history)

        if history and history[-1]["role"] == "assistant":
            print(history[-1].get("content") or "")

        print()