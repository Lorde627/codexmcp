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
import uuid
from dataclasses import dataclass, field
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
TASK_RETENTION_SECONDS = 3600  # prune completed tasks after 1 hour
MAX_TRACKED_TASKS = 200        # hard cap on registry size

# Track child processes for cleanup on server exit
_child_processes: list[subprocess.Popen] = []
_child_lock = threading.Lock()


def _cleanup_children() -> None:
    """Kill any lingering codex child processes on server exit."""
    with _child_lock:
        for proc in list(_child_processes):
            try:
                if proc.poll() is None:
                    _kill_process(proc, os.name != "nt")
            except OSError:
                pass
        _child_processes.clear()


atexit.register(_cleanup_children)


# ---------------------------------------------------------------------------
# StreamProcessor — structured JSONL event parser
# ---------------------------------------------------------------------------
@dataclass
class TaskEvent:
    """A single parsed event from codex JSONL output."""
    type: str   # text, thinking, command, tool_call, tool_result, error
    text: str = ""
    tool_name: str = ""


class StreamProcessor:
    """Incrementally parses codex exec --json JSONL output into structured events.

    Thread-safe: all mutations go through self._lock.  Readers should call
    snapshot() to get a consistent cross-field view.
    """

    MAX_RECENT_EVENTS = 50
    NON_JSON_BUFFER_SIZE = 20
    MAX_RAW_LINES = 2000
    MAX_ALL_MESSAGES = 2000
    MAX_ERRORS = 200

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.agent_messages: str = ""
        self.thread_id: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.errors: list[str] = []
        self.done: bool = False
        self.all_messages: list[Dict[str, Any]] = []
        self.raw_lines: list[str] = []
        self._events: list[TaskEvent] = []
        self._non_json_lines: list[str] = []

    @staticmethod
    def _bounded_append(items: list, value: Any, limit: int) -> None:
        items.append(value)
        if len(items) > limit:
            del items[:-limit]

    def process_line(self, line: str) -> Optional[TaskEvent]:
        """Parse one JSONL line.  Updates internal state and returns an event."""
        with self._lock:
            self._bounded_append(self.raw_lines, line, self.MAX_RAW_LINES)

        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            with self._lock:
                self._bounded_append(
                    self._non_json_lines, line[:500], self.NON_JSON_BUFFER_SIZE
                )
                self._bounded_append(
                    self.errors, f"[non-JSON output] {line}", self.MAX_ERRORS
                )
            return None

        with self._lock:
            self._bounded_append(self.all_messages, data, self.MAX_ALL_MESSAGES)
            event_type = data.get("type", "")

            if data.get("thread_id") is not None:
                self.thread_id = data["thread_id"]

            if event_type == "turn.completed":
                turn_usage = data.get("usage")
                if turn_usage:
                    self.usage = turn_usage
                self.done = True
                return None

            if event_type in ("item.completed", "item.started"):
                return self._process_item_locked(data.get("item", {}))

            if "fail" in event_type:
                detail = data.get("error", {}).get("message", "unknown failure")
                self._bounded_append(
                    self.errors, f"[codex fail] {detail}", self.MAX_ERRORS
                )
                evt = TaskEvent(type="error", text=detail)
                self._push_event_locked(evt)
                return evt

            if "error" in event_type:
                msg = data.get("message", "")
                if not re.match(r"^Reconnecting\.\.\.\s+\d+/\d+", msg):
                    self._bounded_append(
                        self.errors, f"[codex error] {msg}", self.MAX_ERRORS
                    )
                    evt = TaskEvent(type="error", text=msg)
                    self._push_event_locked(evt)
                    return evt

        return None

    def _process_item_locked(self, item: dict) -> Optional[TaskEvent]:
        """Must be called while holding self._lock."""
        item_type = item.get("type", "")
        evt: Optional[TaskEvent] = None

        if item_type == "agent_message":
            text = item.get("text", "")
            self.agent_messages += text
            evt = TaskEvent(type="text", text=text[:500])
        elif item_type == "reasoning":
            thinking = item.get("summary", [])
            text = thinking[0].get("text", "") if thinking else ""
            evt = TaskEvent(type="thinking", text=text[:300])
        elif item_type == "command_execution":
            cmd_str = item.get("command", "")
            exit_code = item.get("exit_code")
            evt = TaskEvent(type="command", text=f"{cmd_str} (exit={exit_code})")
        elif item_type == "function_call":
            evt = TaskEvent(
                type="tool_call",
                tool_name=item.get("name", ""),
                text=str(item.get("arguments", ""))[:300],
            )
        elif item_type == "function_call_output":
            output = item.get("output", "")
            evt = TaskEvent(type="tool_result", text=str(output)[:300])

        if evt:
            self._push_event_locked(evt)
        return evt

    def _push_event_locked(self, evt: TaskEvent) -> None:
        self._events.append(evt)
        if len(self._events) > self.MAX_RECENT_EVENTS:
            self._events = self._events[-self.MAX_RECENT_EVENTS:]

    def snapshot(self) -> Dict[str, Any]:
        """Return a consistent cross-field snapshot for readers."""
        with self._lock:
            return {
                "success": bool(self.agent_messages) and self.thread_id is not None,
                "agent_messages": self.agent_messages,
                "thread_id": self.thread_id,
                "all_messages": list(self.all_messages),
                "errors": list(self.errors),
                "raw_lines": list(self.raw_lines),
                "usage": self.usage,
                "recent_events": [
                    {"type": e.type, "text": e.text, "tool_name": e.tool_name}
                    for e in self._events
                ],
                "events_count": len(self.all_messages),
            }

    @property
    def success(self) -> bool:
        with self._lock:
            return bool(self.agent_messages) and self.thread_id is not None

    def to_result(self) -> Dict[str, Any]:
        return self.snapshot()


# ---------------------------------------------------------------------------
# Background task registry
# ---------------------------------------------------------------------------
@dataclass
class _TaskState:
    task_id: str
    prompt_preview: str
    cd: str
    status: str = "running"      # running | completed | failed | cancelled
    started_at: float = field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    stream: StreamProcessor = field(default_factory=StreamProcessor)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _asyncio_task: Optional[asyncio.Task] = field(default=None, repr=False)


_task_registry: Dict[str, _TaskState] = {}
_registry_lock = threading.Lock()


def _allocate_task_id_locked() -> str:
    """Generate a unique task ID while holding _registry_lock."""
    while True:
        task_id = uuid.uuid4().hex[:12]
        if task_id not in _task_registry:
            return task_id


def _prune_registry_locked(now: Optional[float] = None) -> None:
    """Remove expired/overflow tasks while holding _registry_lock."""
    current = now or time.monotonic()
    # Remove tasks completed more than TASK_RETENTION_SECONDS ago
    expired = [
        tid for tid, s in _task_registry.items()
        if s.completed_at is not None
        and current - s.completed_at > TASK_RETENTION_SECONDS
    ]
    for tid in expired:
        _task_registry.pop(tid, None)
    # If still over limit, evict oldest completed tasks
    overflow = len(_task_registry) - MAX_TRACKED_TASKS
    if overflow > 0:
        finished = sorted(
            (s.completed_at or current, tid)
            for tid, s in _task_registry.items()
            if s.completed_at is not None
        )
        for _, tid in finished[:overflow]:
            _task_registry.pop(tid, None)


# ---------------------------------------------------------------------------
# Subprocess runner with timeout
# ---------------------------------------------------------------------------
def _is_process_busy(pid: int) -> bool:
    """Check if a process or any of its descendants is consuming CPU."""
    try:
        pgid = os.getpgid(pid)
        pgrep = subprocess.run(
            ["pgrep", "-g", str(pgid)],
            capture_output=True, text=True, timeout=3,
        )
        pids = pgrep.stdout.strip()
        if not pids:
            return False
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
        return True


def run_shell_command(
    cmd: list[str],
    stdin_text: Optional[str] = None,
    global_timeout: float = GLOBAL_TIMEOUT,
    cancel_event: Optional[threading.Event] = None,
) -> Generator[str, None, None]:
    """Execute a command and stream its output line-by-line.

    Uses adaptive idle timeout: short before first output (fast failure
    detection), long after first output (allow deep model reasoning).
    """
    popen_cmd = cmd.copy()
    codex_path = shutil.which("codex") or cmd[0]
    popen_cmd[0] = codex_path

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

    def read_output() -> None:
        try:
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if abort_event.is_set():
                        break
                    stripped = line.strip()
                    if stripped:
                        output_queue.put(stripped)
                    # Don't kill the process on turn.completed — let it exit
                    # naturally so we don't lose tail events or create spurious
                    # non-zero exit codes from SIGTERM.
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
    cancelled = False

    while True:
        # Check external cancel signal
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break

        elapsed = time.monotonic() - start_time
        idle_elapsed = time.monotonic() - last_output_time
        idle_limit = IDLE_TIMEOUT_ACTIVE if got_first_output else IDLE_TIMEOUT_INITIAL

        if elapsed > global_timeout:
            timed_out = True
            break

        if idle_elapsed > idle_limit:
            if process.poll() is None and _is_process_busy(process.pid):
                log.info(
                    "idle timeout (%.0fs) reached but pid=%d still busy, extending",
                    idle_elapsed, process.pid,
                )
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
    if timed_out or cancelled:
        abort_event.set()
        reason = "cancelled" if cancelled else (
            "global" if elapsed > global_timeout else "idle"
        )
        log.warning("codex process %s (pid=%d), killing", reason, process.pid)
        _kill_process(process, use_new_session)
        if timed_out:
            yield json.dumps({
                "type": "error",
                "message": (
                    f"codex process killed: {reason} timeout "
                    f"({global_timeout}s global / {idle_limit}s idle) exceeded"
                ),
            })
    else:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process(process, use_new_session)

    thread.join(timeout=10)

    while True:
        try:
            line = output_queue.get_nowait()
            if line is None:
                break
            yield line
        except queue.Empty:
            break

    rc = process.returncode
    if rc is None:
        try:
            rc = process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            rc = -1
    # Only report unexpected exit codes — skip if we already killed the process
    if rc and rc != 0 and not timed_out and not cancelled:
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
    try:
        if use_pg:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (OSError, ProcessLookupError):
        pass


def _kill_process(proc: subprocess.Popen, use_pg: bool) -> None:
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
# Core parsing logic — uses StreamProcessor, runs in a thread
# ---------------------------------------------------------------------------
def _parse_codex_output(
    cmd: list[str],
    prompt: str,
    stream: Optional[StreamProcessor] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run a codex command and parse JSONL output via StreamProcessor."""
    sp = stream or StreamProcessor()

    for line in run_shell_command(cmd, stdin_text=prompt, cancel_event=cancel_event):
        sp.process_line(line)

    return sp.to_result()


# ---------------------------------------------------------------------------
# Shared command builder
# ---------------------------------------------------------------------------
def _build_codex_cmd(
    sandbox: str,
    cd_path: Path,
    session_id: str = "",
    image: Optional[List[Path]] = None,
    model: str = "",
    profile: str = "",
    reasoning_effort: Optional[str] = None,
    yolo: bool = False,
    skip_git_repo_check: bool = True,
) -> list[str]:
    """Build the codex exec CLI command list."""
    cmd = [
        "codex", "exec",
        "--sandbox", sandbox,
        "--cd", str(cd_path),
        "--json",
    ]

    images = list(image) if image else []
    if images:
        cmd.extend(["--image", ",".join(str(p) for p in images)])
    if model:
        cmd.extend(["--model", model])

    effective_profile = profile or os.environ.get("CODEX_PROFILE", "")
    if effective_profile:
        cmd.extend(["--profile", effective_profile])

    effective_effort = reasoning_effort or os.environ.get("CODEX_REASONING_EFFORT")
    if effective_effort:
        cmd.extend(["--config", f'model_reasoning_effort="{effective_effort}"'])

    if yolo:
        cmd.append("--yolo")
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")

    if session_id:
        cmd.extend(["resume", session_id, "-"])
    else:
        cmd += ["--", "-"]

    return cmd


def _preflight(cd: Path) -> Optional[Dict[str, Any]]:
    """Run pre-flight checks.  Returns an error dict or None if OK."""
    if not cd.exists():
        return {"success": False, "error": f"Working directory does not exist: {cd}"}
    if not shutil.which("codex"):
        return {"success": False, "error": "codex CLI not found in PATH."}
    return None


def _build_response(parsed: Dict[str, Any], return_all: bool) -> Dict[str, Any]:
    """Build the final MCP response from parsed output."""
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

    if return_all:
        result["all_messages"] = parsed["all_messages"]

    return result


def _should_retry(parsed: Dict[str, Any], *, is_resume: bool) -> bool:
    """Return True when a failed run looks transient enough to retry."""
    if parsed["success"] or parsed["agent_messages"] or is_resume:
        return False
    if parsed["errors"] and all("[non-JSON output]" in e for e in parsed["errors"]):
        return False
    return True


# ---------------------------------------------------------------------------
# MCP tool: codex (blocking)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="codex",
    description="""
    Executes a non-interactive Codex session via CLI and **waits** for completion.
    Use this for tasks where you need the result immediately (code review, quick questions, etc.).
    For long-running tasks, use `codex_dispatch` instead.

    **Key Features:**
        - **Prompt-Driven Execution:** Send task instructions to Codex for step-by-step code handling.
        - **Workspace Isolation:** Operate within a specified directory, with optional Git repo skipping.
        - **Security Controls:** Three sandbox levels balance functionality and safety.
        - **Session Persistence:** Resume prior conversations via `SESSION_ID` for iterative tasks.
        - **Auto-Retry:** Transient failures are retried automatically (up to 2 attempts).
    """,
    meta={"version": "0.0.0", "author": "guda.studio"},
)
async def codex(
    PROMPT: Annotated[str, "Instruction for the task to send to codex."],
    cd: Annotated[Path, "Set the workspace root for codex before executing the task."],
    sandbox: Annotated[
        Literal["read-only", "workspace-write", "danger-full-access"],
        Field(description="Sandbox policy. Defaults to `read-only`."),
    ] = "read-only",
    SESSION_ID: Annotated[
        str, "Resume a previous session. Defaults to empty (new session).",
    ] = "",
    skip_git_repo_check: Annotated[
        bool, "Allow running outside a Git repository.",
    ] = True,
    return_all_messages: Annotated[
        bool, "Return all messages including reasoning and tool calls.",
    ] = False,
    image: Annotated[
        Optional[List[Path]],
        Field(description="Attach image files to the prompt."),
    ] = None,
    model: Annotated[
        str,
        Field(description="Model override. Prohibited unless user specifies."),
    ] = "",
    yolo: Annotated[
        bool,
        Field(description="Skip all approvals and sandboxing."),
    ] = False,
    profile: Annotated[
        str, "Config profile from ~/.codex/config.toml.",
    ] = "",
    reasoning_effort: Annotated[
        Optional[Literal["low", "medium", "high", "xhigh"]],
        Field(description="Thinking budget override (low/medium/high/xhigh)."),
    ] = None,
) -> Dict[str, Any]:
    """Execute a Codex CLI session and wait for the result."""

    cd_path = Path(cd)
    err = _preflight(cd_path)
    if err:
        return err

    cmd = _build_codex_cmd(
        sandbox=sandbox, cd_path=cd_path, session_id=SESSION_ID,
        image=image, model=model, profile=profile,
        reasoning_effort=reasoning_effort, yolo=yolo,
        skip_git_repo_check=skip_git_repo_check,
    )

    is_resume = bool(SESSION_ID)
    last_parsed: Optional[Dict[str, Any]] = None

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("codex attempt %d/%d", attempt, MAX_RETRIES)
        parsed = await asyncio.to_thread(_parse_codex_output, cmd, PROMPT)
        last_parsed = parsed

        if not _should_retry(parsed, is_resume=is_resume):
            break

        if attempt < MAX_RETRIES:
            log.info(
                "codex attempt %d failed (%s), retrying in %ds",
                attempt, "; ".join(parsed["errors"][:3]) or "no output", RETRY_DELAY,
            )
            await asyncio.sleep(RETRY_DELAY)

    parsed = last_parsed  # type: ignore[assignment]
    return _build_response(parsed, return_all_messages)


# ---------------------------------------------------------------------------
# MCP tool: codex_dispatch (background, non-blocking)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="codex_dispatch",
    description="""
    Dispatch a Codex task to run **in the background** and return immediately with a `task_id`.
    Use this for long-running tasks (large code reviews, complex implementations, multi-file refactors).
    Check progress with `codex_status` and cancel with `codex_cancel`.
    """,
)
async def codex_dispatch(
    PROMPT: Annotated[str, "Instruction for the task to send to codex."],
    cd: Annotated[Path, "Set the workspace root for codex."],
    sandbox: Annotated[
        Literal["read-only", "workspace-write", "danger-full-access"],
        Field(description="Sandbox policy. Defaults to `read-only`."),
    ] = "read-only",
    SESSION_ID: Annotated[str, "Resume a previous session."] = "",
    skip_git_repo_check: Annotated[bool, "Allow running outside Git repo."] = True,
    image: Annotated[
        Optional[List[Path]],
        Field(description="Attach image files to the prompt."),
    ] = None,
    model: Annotated[str, Field(description="Model override.")] = "",
    yolo: Annotated[bool, Field(description="Skip approvals.")] = False,
    profile: Annotated[str, "Config profile."] = "",
    reasoning_effort: Annotated[
        Optional[Literal["low", "medium", "high", "xhigh"]],
        Field(description="Thinking budget override."),
    ] = None,
) -> Dict[str, Any]:
    """Start a background Codex task and return its task_id immediately."""

    cd_path = Path(cd)
    err = _preflight(cd_path)
    if err:
        return err

    cmd = _build_codex_cmd(
        sandbox=sandbox, cd_path=cd_path, session_id=SESSION_ID,
        image=image, model=model, profile=profile,
        reasoning_effort=reasoning_effort, yolo=yolo,
        skip_git_repo_check=skip_git_repo_check,
    )

    with _registry_lock:
        _prune_registry_locked()
        if len(_task_registry) >= MAX_TRACKED_TASKS:
            running = sum(1 for t in _task_registry.values() if t.completed_at is None)
            return {
                "success": False,
                "error": (
                    f"Task registry full ({len(_task_registry)}/{MAX_TRACKED_TASKS}); "
                    f"{running} active task(s) still running. Cancel or wait for tasks to complete."
                ),
            }
        task_id = _allocate_task_id_locked()
        state = _TaskState(
            task_id=task_id,
            prompt_preview=PROMPT[:200],
            cd=str(cd_path),
        )
        _task_registry[task_id] = state

    async def _run() -> None:
        is_resume = bool(SESSION_ID)
        try:
            for attempt in range(1, MAX_RETRIES + 1):
                await asyncio.to_thread(
                    _parse_codex_output, cmd, PROMPT,
                    stream=state.stream,
                    cancel_event=state.cancel_event,
                )
                parsed = state.stream.to_result()
                if not _should_retry(parsed, is_resume=is_resume):
                    break
                if state.cancel_event.is_set():
                    break
                if attempt < MAX_RETRIES:
                    log.info(
                        "dispatch task %s attempt %d failed, retrying in %ds",
                        task_id, attempt, RETRY_DELAY,
                    )
                    await asyncio.sleep(RETRY_DELAY)
                    # Reset stream for next attempt
                    state.stream = StreamProcessor()

            with _registry_lock:
                if state.status == "running":
                    state.status = "completed" if state.stream.success else "failed"
        except Exception as exc:
            state.stream.process_line(json.dumps({
                "type": "error",
                "message": f"[dispatch error] {exc!r}",
            }))
            with _registry_lock:
                if state.status == "running":
                    state.status = "failed"
        finally:
            with _registry_lock:
                if state.completed_at is None:
                    state.completed_at = time.monotonic()
                state._asyncio_task = None
                _prune_registry_locked(state.completed_at)
            log.info("dispatch task %s finished: %s", task_id, state.status)

    state._asyncio_task = asyncio.create_task(_run())

    return {
        "task_id": task_id,
        "status": "running",
        "prompt_preview": state.prompt_preview,
        "cd": state.cd,
    }


# ---------------------------------------------------------------------------
# MCP tool: codex_status
# ---------------------------------------------------------------------------
@mcp.tool(
    name="codex_status",
    description="""
    Query the status of background Codex tasks.
    - With `task_id`: returns detailed status, progress events, and result (if complete).
    - Without `task_id`: lists all tracked tasks with their status.
    """,
)
async def codex_status(
    task_id: Annotated[
        str, "Task ID to query. Leave empty to list all tasks.",
    ] = "",
) -> Dict[str, Any]:
    """Query background task status."""

    if not task_id:
        with _registry_lock:
            _prune_registry_locked()
            tasks = [
                {
                    "task_id": t.task_id,
                    "status": t.status,
                    "prompt_preview": t.prompt_preview,
                    "cd": t.cd,
                    "elapsed_seconds": round(
                        (t.completed_at or time.monotonic()) - t.started_at, 1
                    ),
                }
                for t in _task_registry.values()
            ]
        return {"tasks": tasks}

    with _registry_lock:
        state = _task_registry.get(task_id)
        if not state:
            return {"success": False, "error": f"Task '{task_id}' not found."}
        status = state.status
        started_at = state.started_at
        completed_at = state.completed_at

    snap = state.stream.snapshot()
    elapsed = round((completed_at or time.monotonic()) - started_at, 1)

    result: Dict[str, Any] = {
        "task_id": task_id,
        "status": status,
        "prompt_preview": state.prompt_preview,
        "cd": state.cd,
        "elapsed_seconds": elapsed,
        "recent_events": snap["recent_events"][-20:],
        "errors": snap["errors"][-10:] if snap["errors"] else [],
    }

    if status != "running":
        result["agent_messages"] = snap["agent_messages"]
        result["SESSION_ID"] = snap["thread_id"]
        if snap["usage"]:
            result["usage"] = snap["usage"]
    else:
        if snap["agent_messages"]:
            result["agent_messages_preview"] = snap["agent_messages"][-500:]
        result["events_count"] = snap["events_count"]

    return result


# ---------------------------------------------------------------------------
# MCP tool: codex_cancel
# ---------------------------------------------------------------------------
@mcp.tool(
    name="codex_cancel",
    description="Cancel a running background Codex task by its `task_id`.",
)
async def codex_cancel(
    task_id: Annotated[str, "The task ID to cancel."],
) -> Dict[str, Any]:
    """Cancel a background task."""

    with _registry_lock:
        state = _task_registry.get(task_id)
        if not state:
            return {"success": False, "error": f"Task '{task_id}' not found."}
        if state.status != "running":
            return {
                "success": False,
                "error": f"Task '{task_id}' is not running (status: {state.status}).",
            }
        state.cancel_event.set()
        state.status = "cancelled"
        state.completed_at = time.monotonic()
        started_at = state.started_at
        task = state._asyncio_task

    # Wait briefly for the asyncio task to finish cleanup
    if task and not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    snap = state.stream.snapshot()
    elapsed = round((state.completed_at or time.monotonic()) - started_at, 1)
    log.info("task %s cancelled after %.1fs", task_id, elapsed)

    return {
        "task_id": task_id,
        "status": "cancelled",
        "elapsed_seconds": elapsed,
        "partial_agent_messages": snap["agent_messages"][-500:] if snap["agent_messages"] else "",
    }


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------
def run() -> None:
    """Start the MCP server over stdio transport."""
    mcp.run(transport="stdio")
