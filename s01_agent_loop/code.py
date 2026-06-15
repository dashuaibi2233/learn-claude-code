#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.

Usage:
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY=... python s01_agent_loop/code.py
"""

import os
import subprocess

try:
    import readline
    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = OpenAI(base_url=os.getenv("DASHSCOPE_BASE_URL"), api_key=os.getenv("DASHSCOPE_API_KEY"))
MODEL = os.getenv("MODEL_ID", "qwen-plus")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ── Tool definition: just bash ────────────────────────────
# TOOLS = [{
#     "name": "bash",
#     "description": "Run a shell command.",
#     "input_schema": {
#         "type": "object",
#         "properties": {"command": {"type": "string"}},
#         "required": ["command"],
#     },
# }]


TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"],
        },
    },
}]

# ── Tool execution ────────────────────────────────────────
def run_bash(command: str) -> str:
    dangerous = ["rm -rf","sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ── The core pattern: a while loop that calls tools until the model stops ──
def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        msg = response.choices[0].message

        # OpenAI 格式：assistant 消息要先塞回 history
        messages.append(msg.model_dump(exclude_none=True))

        # 没有 tool_calls，说明模型回答完了
        if not msg.tool_calls:
            return

        # 执行工具
        for tool_call in msg.tool_calls:
            if tool_call.function.name != "bash":
                continue

            args = json.loads(tool_call.function.arguments or "{}")
            command = args.get("command", "")

            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])

            # OpenAI 格式：工具结果 role 是 tool
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")
    print(f"当前目录: {os.getcwd()}\n")

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
        # Print the model's final text response
        if history and history[-1]["role"] == "assistant":
            print(history[-1].get("content") or "")
        print()
