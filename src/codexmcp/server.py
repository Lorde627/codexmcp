"""FastMCP server implementation for the Codex MCP project."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Dict, Generator, List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field
import shutil

# ---------------------------------------------------------------------------
# Logging — writes to stderr so it never contaminates MCP stdio transport
# ---------------------------------------------------------------------------
log = logging.getLogger("codexmcp")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[codexmcp %(levelname)s] %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)

mcp = FastMCP("Codex MCP Server-from guda.studio")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GLOBAL_TIMEOUT = 600       # 10 min hard cap per invocation
IDLE_TIMEOUT_INITIAL = 120 # 2 min before first output → fast startup-failure detection
IDLE_TIMEOUT_ACTIVE = 480  # 8 min after first output → allow deep reasoning
MAX_RETRIES = 2            # retry up to 2 times on transient failure
RETRY_DELAY = 3            # seconds between retries
GRACEFUL_SHUTDOWN_DELAY = 0.3

# Track child processes for cleanup on server exit
_child_processes: list[subprocess.Popen] = []
_child_lock = threading.Lock()


def _cleanup_children() -> None:
    """Kill any lingering codex child processes on server exit."""
    with _child_lock:
        for proc in _child_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except OSError:
                pass
        _child_processes.clear()


atexit.register(_cleanup_children)


# ---------------------------------------------------------------------------
# Subprocess runner with timeout
# ---------------------------------------------------------------------------
def _is_process_busy(pid: int) -> bool:
    """Check if a process is still consuming CPU (not just sleeping/hung).

    Uses /proc or ps to detect whether the child is actively working.
    Returns True if we can't determine status (fail-open).
    """
    try:
        result = subprocess.run(
            ["ps", "-o", "%cpu=", "-p", str(pid)],
            capture_output=True, text=True, timeout=3,
        )
        cpu = float(result.stdout.strip())
        return cpu > 0.1
    except Exception:
        return True  # fail-open: assume busy if we can't check


def run_shell_command(
    cmd: list[str],
    global_timeout: float = GLOBAL_TIMEOUT,
) -> Generator[str, None, None]:
    """Execute a command and stream its output line-by-line.

    Uses adaptive idle timeout: short before first output (fast failure
    detection), long after first output (allow deep model reasoning).
    When idle timeout is about to fire, checks process CPU to distinguish
    'still thinking' from 'truly hung'.

    Args:
        cmd: Command and arguments as a list.
        global_timeout: Maximum total seconds before force-killing.

    Yields:
        Output lines (stripped) from the command.
    """
    popen_cmd = cmd.copy()
    codex_path = shutil.which("codex") or cmd[0]
    popen_cmd[0] = codex_path

    # Use start_new_session on Unix so child gets its own process group,
    # allowing us to kill it and all its descendants cleanly.
    use_new_session = os.name != "nt"

    process = subprocess.Popen(
        popen_cmd,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        start_new_session=use_new_session,
    )

    with _child_lock:
        _child_processes.append(process)

    log.info("codex process started (pid=%d)", process.pid)

    output_queue: queue.Queue[str | None] = queue.Queue()
    abort_event = threading.Event()

    def is_turn_completed(line: str) -> bool:
        try:
            data = json.loads(line)
            return data.get("type") == "turn.completed"
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False

    def read_output() -> None:
        try:
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if abort_event.is_set():
                        break
                    stripped = line.strip()
                    if stripped:
                        output_queue.put(stripped)
                    if is_turn_completed(stripped):
                        time.sleep(GRACEFUL_SHUTDOWN_DELAY)
                        _terminate_process(process, use_new_session)
                        break
                process.stdout.close()
        except Exception as exc:
            log.warning("read_output thread error: %s", exc)
        finally:
            output_queue.put(None)

    thread = threading.Thread(target=read_output, daemon=True)
    thread.start()

    start_time = time.monotonic()
    last_output_time = start_time
    got_first_output = False
    timed_out = False

    while True:
        elapsed = time.monotonic() - start_time
        idle_elapsed = time.monotonic() - last_output_time
        idle_limit = IDLE_TIMEOUT_ACTIVE if got_first_output else IDLE_TIMEOUT_INITIAL

        if elapsed > global_timeout:
            timed_out = True
            break

        if idle_elapsed > idle_limit:
            # Before killing, check if process is still actively working
            if process.poll() is None and _is_process_busy(process.pid):
                log.info(
                    "idle timeout (%.0fs) reached but process pid=%d still busy, extending",
                    idle_elapsed, process.pid,
                )
                # Reset idle timer — process is still working
                last_output_time = time.monotonic()
                continue
            timed_out = True
            break

        try:
            line = output_queue.get(timeout=0.5)
            if line is None:
                break
            last_output_time = time.monotonic()
            if not got_first_output:
                got_first_output = True
            yield line
        except queue.Empty:
            if process.poll() is not None and not thread.is_alive():
                break

    # --- cleanup ---
    if timed_out:
        abort_event.set()
        timeout_kind = "global" if elapsed > global_timeout else "idle"
        log.warning(
            "codex process timed out (%s, pid=%d, elapsed=%.0fs, idle=%.0fs), killing",
            timeout_kind, process.pid, elapsed, idle_elapsed,
        )
        _kill_process(process, use_new_session)
        yield json.dumps({
            "type": "error",
            "message": (
                f"codex process killed: {timeout_kind} timeout "
                f"({global_timeout}s global / {idle_limit}s idle) exceeded"
            ),
        })
    else:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process(process, use_new_session)

    thread.join(timeout=5)

    # drain remaining
    while not output_queue.empty():
        try:
            line = output_queue.get_nowait()
            if line is not None:
                yield line
        except queue.Empty:
            break

    # surface non-zero exit code
    rc = process.returncode
    if rc and rc != 0:
        log.warning("codex process exited with code %d", rc)
        yield json.dumps({
            "type": "error",
            "message": f"codex process exited with code {rc}",
        })

    with _child_lock:
        try:
            _child_processes.remove(process)
        except ValueError:
            pass

    elapsed_total = time.monotonic() - start_time
    log.info("codex process finished (pid=%d, %.1fs)", process.pid, elapsed_total)


def _terminate_process(proc: subprocess.Popen, use_pg: bool) -> None:
    """Send SIGTERM, preferring the process group on Unix."""
    try:
        if use_pg:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (OSError, ProcessLookupError):
        pass


def _kill_process(proc: subprocess.Popen, use_pg: bool) -> None:
    """Escalate: SIGTERM → wait → SIGKILL."""
    _terminate_process(proc, use_pg)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            if use_pg:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def windows_escape(prompt: str) -> str:
    result = prompt.replace("\\", "\\\\")
    result = result.replace('"', '\\"')
    result = result.replace("\n", "\\n")
    result = result.replace("\r", "\\r")
    result = result.replace("\t", "\\t")
    result = result.replace("\b", "\\b")
    result = result.replace("\f", "\\f")
    result = result.replace("'", "\\'")
    return result


# ---------------------------------------------------------------------------
# Core parsing logic (extracted for retry) — runs in a thread, NOT on the
# asyncio event loop
# ---------------------------------------------------------------------------
def _parse_codex_output(cmd: list[str]) -> Dict[str, Any]:
    """Run a codex command and parse the JSONL output into a result dict."""
    all_messages: list[Dict[str, Any]] = []
    raw_lines: list[str] = []
    agent_messages = ""
    errors: list[str] = []
    thread_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    for line in run_shell_command(cmd):
        raw_lines.append(line)
        try:
            line_dict = json.loads(line.strip())
            all_messages.append(line_dict)

            line_type = line_dict.get("type", "")

            # extract thread_id
            if line_dict.get("thread_id") is not None:
                thread_id = line_dict["thread_id"]

            # extract agent text
            item = line_dict.get("item", {})
            if item.get("type") == "agent_message":
                agent_messages += item.get("text", "")

            # extract token usage from turn.completed
            if line_type == "turn.completed":
                turn_usage = line_dict.get("usage")
                if turn_usage:
                    usage = turn_usage

            # capture errors
            if "fail" in line_type:
                err_detail = line_dict.get("error", {}).get("message", "unknown failure")
                errors.append(f"[codex fail] {err_detail}")
            elif "error" in line_type:
                error_msg = line_dict.get("message", "")
                is_reconnecting = bool(
                    re.match(r"^Reconnecting\.\.\.\s+\d+/\d+", error_msg)
                )
                if not is_reconnecting:
                    errors.append(f"[codex error] {error_msg}")

        except json.JSONDecodeError:
            errors.append(f"[non-JSON output] {line}")
        except Exception as exc:
            errors.append(f"[unexpected error] {exc!r} — line: {line!r}")

    return {
        "success": bool(agent_messages) and thread_id is not None,
        "agent_messages": agent_messages,
        "thread_id": thread_id,
        "all_messages": all_messages,
        "errors": errors,
        "raw_lines": raw_lines,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------
@mcp.tool(
    name="codex",
    description="""
    Executes a non-interactive Codex session via CLI to perform AI-assisted coding tasks in a secure workspace.
    This tool wraps the `codex exec` command, enabling model-driven code generation, debugging, or automation based on natural language prompts.
    It supports resuming ongoing sessions for continuity and enforces sandbox policies to prevent unsafe operations. Ideal for integrating Codex into MCP servers for agentic workflows, such as code reviews or repo modifications.

    **Key Features:**
        - **Prompt-Driven Execution:** Send task instructions to Codex for step-by-step code handling.
        - **Workspace Isolation:** Operate within a specified directory, with optional Git repo skipping.
        - **Security Controls:** Three sandbox levels balance functionality and safety.
        - **Session Persistence:** Resume prior conversations via `SESSION_ID` for iterative tasks.

    **Edge Cases & Best Practices:**
        - Ensure `cd` exists and is accessible; tool fails silently on invalid paths.
        - For most repos, prefer "read-only" to avoid accidental changes.
        - If needed, set `return_all_messages` to `True` to parse "all_messages" for detailed tracing (e.g., reasoning, tool calls, etc.).
    """,
    meta={"version": "0.0.0", "author": "guda.studio"},
)
async def codex(
    PROMPT: Annotated[str, "Instruction for the task to send to codex."],
    cd: Annotated[Path, "Set the workspace root for codex before executing the task."],
    sandbox: Annotated[
        Literal["read-only", "workspace-write", "danger-full-access"],
        Field(
            description="Sandbox policy for model-generated commands. Defaults to `read-only`."
        ),
    ] = "read-only",
    SESSION_ID: Annotated[
        str,
        "Resume the specified session of the codex. Defaults to `None`, start a new session.",
    ] = "",
    skip_git_repo_check: Annotated[
        bool,
        "Allow codex running outside a Git repository (useful for one-off directories).",
    ] = True,
    return_all_messages: Annotated[
        bool,
        "Return all messages (e.g. reasoning, tool calls, etc.) from the codex session. Set to `False` by default, only the agent's final reply message is returned.",
    ] = False,
    image: Annotated[
        List[Path],
        Field(
            description="Attach one or more image files to the initial prompt. Separate multiple paths with commas or repeat the flag.",
        ),
    ] = [],
    model: Annotated[
        str,
        Field(
            description="The model to use for the codex session. This parameter is strictly prohibited unless explicitly specified by the user.",
        ),
    ] = "",
    yolo: Annotated[
        bool,
        Field(
            description="Run every command without approvals or sandboxing. Only use when `sandbox` couldn't be applied.",
        ),
    ] = False,
    profile: Annotated[
        str,
        "Configuration profile name to load from `~/.codex/config.toml`. This parameter is strictly prohibited unless explicitly specified by the user.",
    ] = "",
) -> Dict[str, Any]:
    """Execute a Codex CLI session and return the results."""

    # --- pre-flight validation ---
    cd_path = Path(cd)
    if not cd_path.exists():
        return {
            "success": False,
            "error": f"Working directory does not exist: {cd_path}",
        }

    codex_bin = shutil.which("codex")
    if not codex_bin:
        return {
            "success": False,
            "error": "codex CLI not found in PATH. Install it first.",
        }

    # --- build command ---
    cmd = ["codex", "exec", "--sandbox", sandbox, "--cd", str(cd_path), "--json"]

    if image:
        cmd.extend(["--image", ",".join(str(p) for p in image)])
    if model:
        cmd.extend(["--model", model])
    if profile:
        cmd.extend(["--profile", profile])
    if yolo:
        cmd.append("--yolo")
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    if SESSION_ID:
        cmd.extend(["resume", str(SESSION_ID)])

    prompt_text = windows_escape(PROMPT) if os.name == "nt" else PROMPT
    cmd += ["--", prompt_text]

    # --- execute with retry ---
    # CRITICAL: use asyncio.to_thread() so the blocking subprocess work
    # runs in a thread pool instead of freezing the MCP event loop.
    last_parsed: Optional[Dict[str, Any]] = None

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("codex attempt %d/%d", attempt, MAX_RETRIES)
        parsed = await asyncio.to_thread(_parse_codex_output, cmd)
        last_parsed = parsed

        if parsed["success"]:
            break

        # don't retry if we already got partial agent content
        if parsed["agent_messages"]:
            break

        if attempt < MAX_RETRIES:
            log.info(
                "codex attempt %d failed (%s), retrying in %ds",
                attempt,
                "; ".join(parsed["errors"][:3]) or "no output",
                RETRY_DELAY,
            )
            await asyncio.sleep(RETRY_DELAY)

    parsed = last_parsed  # type: ignore[assignment]

    # --- build response ---
    if parsed["success"]:
        result: Dict[str, Any] = {
            "success": True,
            "SESSION_ID": parsed["thread_id"],
            "agent_messages": parsed["agent_messages"],
        }
        if parsed["usage"]:
            result["usage"] = parsed["usage"]
        if parsed["errors"]:
            result["warnings"] = parsed["errors"]
    else:
        error_parts: list[str] = []
        if not parsed["thread_id"]:
            error_parts.append("Failed to obtain SESSION_ID from codex.")
        if not parsed["agent_messages"]:
            error_parts.append("No agent_messages received from codex.")
        if parsed["errors"]:
            error_parts.append("Errors encountered:")
            error_parts.extend(f"  - {e}" for e in parsed["errors"])

        result = {
            "success": False,
            "error": "\n".join(error_parts),
            "all_messages": parsed["all_messages"],
            "raw_lines": parsed["raw_lines"][-20:],
        }
        if parsed["agent_messages"]:
            result["agent_messages"] = parsed["agent_messages"]
        if parsed["thread_id"]:
            result["SESSION_ID"] = parsed["thread_id"]

    if return_all_messages:
        result["all_messages"] = parsed["all_messages"]

    return result


def run() -> None:
    """Start the MCP server over stdio transport."""
    mcp.run(transport="stdio")
