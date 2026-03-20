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
    """Check if a process or any of its descendants is consuming CPU.

    Uses pgrep to find all processes in the same process group, then sums
    their CPU usage.  Returns True (fail-open) if we can't determine status.
    """
    try:
        # Get process group ID — matches start_new_session=True
        pgid = os.getpgid(pid)
        # Find all PIDs in the process group
        pgrep = subprocess.run(
            ["pgrep", "-g", str(pgid)],
            capture_output=True, text=True, timeout=3,
        )
        pids = pgrep.stdout.strip()
        if not pids:
            return False

        # Sum CPU across all processes in the group
        pid_list = ",".join(pids.split())
        ps_result = subprocess.run(
            ["ps", "-o", "%cpu=", "-p", pid_list],
            capture_output=True, text=True, timeout=3,
        )
        total_cpu = sum(
            float(line.strip())
            for line in ps_result.stdout.strip().splitlines()
            if line.strip()
        )
        return total_cpu > 0.5
    except Exception:
        return True  # fail-open: assume busy if we can't check


def run_shell_command(
    cmd: list[str],
    stdin_text: Optional[str] = None,
    global_timeout: float = GLOBAL_TIMEOUT,
) -> Generator[str, None, None]:
    """Execute a command and stream its output line-by-line.

    Uses adaptive idle timeout: short before first output (fast failure
    detection), long after first output (allow deep model reasoning).
    When idle timeout is about to fire, checks process CPU to distinguish
    'still thinking' from 'truly hung'.

    Args:
        cmd: Command and arguments as a list.
        stdin_text: Optional text to feed to the process via stdin.
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
        stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        start_new_session=use_new_session,
    )

    # Write prompt to stdin and close — codex reads it then starts working
    if stdin_text and process.stdin:
        try:
            process.stdin.write(stdin_text)
            process.stdin.close()
        except OSError as exc:
            log.warning("failed to write stdin: %s", exc)

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

    thread.join(timeout=10)

    # drain remaining — thread may have enqueued lines between our last read
    # and the join; exhaust the queue before checking exit code
    while True:
        try:
            line = output_queue.get_nowait()
            if line is None:
                break
            yield line
        except queue.Empty:
            break

    # surface non-zero exit code — read after join to ensure wait() has been
    # called by cleanup above
    rc = process.returncode
    if rc is None:
        # Belt-and-suspenders: if somehow we didn't wait yet
        try:
            rc = process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            rc = -1
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




# ---------------------------------------------------------------------------
# Core parsing logic (extracted for retry) — runs in a thread, NOT on the
# asyncio event loop
# ---------------------------------------------------------------------------
def _parse_codex_output(cmd: list[str], prompt: str) -> Dict[str, Any]:
    """Run a codex command and parse the JSONL output into a result dict."""
    all_messages: list[Dict[str, Any]] = []
    raw_lines: list[str] = []
    agent_messages = ""
    errors: list[str] = []
    thread_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    for line in run_shell_command(cmd, stdin_text=prompt):
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
        Optional[List[Path]],
        Field(
            description="Attach one or more image files to the initial prompt. Separate multiple paths with commas or repeat the flag.",
        ),
    ] = None,
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
    reasoning_effort: Annotated[
        Optional[Literal["low", "medium", "high", "xhigh"]],
        Field(
            description=(
                "Override Codex config `model_reasoning_effort` (thinking budget). "
                "Allowed values: low, medium, high, xhigh. "
                "If omitted, uses the config/profile default or CODEX_REASONING_EFFORT env var."
            ),
        ),
    ] = None,
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
    # Use --ephemeral to avoid cluttering ~/.codex/sessions with temp files.
    # Prompt is delivered via stdin ("-") to avoid command-line length limits
    # and eliminate any need for shell escaping (shell=False + stdin pipe).
    cmd = [
        "codex", "exec",
        "--sandbox", sandbox,
        "--cd", str(cd_path),
        "--json",
        "--ephemeral",
    ]

    images = list(image) if image else []
    if images:
        cmd.extend(["--image", ",".join(str(p) for p in images)])
    if model:
        cmd.extend(["--model", model])

    # profile: explicit param > env var fallback
    effective_profile = profile or os.environ.get("CODEX_PROFILE", "")
    if effective_profile:
        cmd.extend(["--profile", effective_profile])

    # reasoning_effort: explicit param > env var fallback
    effective_effort = reasoning_effort or os.environ.get("CODEX_REASONING_EFFORT")
    if effective_effort:
        cmd.extend(["--config", f'model_reasoning_effort="{effective_effort}"'])

    if yolo:
        cmd.append("--yolo")
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")

    # resume is a subcommand: `resume SESSION_ID -` (stdin)
    # new exec: `-- -` (stdin)
    if SESSION_ID:
        cmd.extend(["resume", str(SESSION_ID), "-"])
    else:
        cmd += ["--", "-"]

    # --- execute with retry ---
    # CRITICAL: use asyncio.to_thread() so the blocking subprocess work
    # runs in a thread pool instead of freezing the MCP event loop.
    #
    # Retry policy: only retry transient/empty failures.  Skip retry when:
    #  - we got partial agent content (may have side-effects already)
    #  - resuming a session (replay could cause duplicate actions)
    #  - deterministic CLI errors (exit code 2 = usage error)
    is_resume = bool(SESSION_ID)
    last_parsed: Optional[Dict[str, Any]] = None

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("codex attempt %d/%d", attempt, MAX_RETRIES)
        parsed = await asyncio.to_thread(_parse_codex_output, cmd, PROMPT)
        last_parsed = parsed

        if parsed["success"]:
            break

        # don't retry if we got partial agent content (side-effects possible)
        if parsed["agent_messages"]:
            break

        # don't retry resume — replaying could duplicate tool actions
        if is_resume:
            break

        # don't retry deterministic errors (non-JSON = CLI usage/config error)
        has_only_deterministic_errors = (
            parsed["errors"]
            and all("[non-JSON output]" in e for e in parsed["errors"])
        )
        if has_only_deterministic_errors:
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
