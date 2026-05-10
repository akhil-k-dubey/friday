"""
skills.py — Built-in skill engine for FRIDAY.

Every skill is a callable that returns a string result.
The LLM triggers skills by emitting [ACTION:skill_name:args] in its response.
The main loop intercepts these, runs the skill, and feeds the result back.

All tools are FREE FOREVER — no API keys required.
"""
from __future__ import annotations

import ast
import datetime
import json
import math
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from urllib import parse as urlparse, request as urlreq


# ── App registry — common Windows apps ──────────────────────────────────────

_APP_REGISTRY: Dict[str, str] = {
    "notepad":    "notepad.exe",
    "calculator": "calc.exe",
    "calc":       "calc.exe",
    "paint":      "mspaint.exe",
    "explorer":   "explorer.exe",
    "file explorer": "explorer.exe",
    "files":      "explorer.exe",
    "cmd":        "cmd.exe",
    "terminal":   "wt.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "taskmgr":    "taskmgr.exe",
    "settings":   "ms-settings:",
    "control panel": "control.exe",
    "chrome":     r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "firefox":    r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "edge":       "msedge.exe",
    "microsoft edge": "msedge.exe",
    "brave":      r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    "vscode":     "code",
    "vs code":    "code",
    "visual studio code": "code",
    "spotify":    r"C:\Users\{user}\AppData\Roaming\Spotify\Spotify.exe",
    "discord":    r"C:\Users\{user}\AppData\Local\Discord\Update.exe --processStart Discord.exe",
    "vlc":        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "obs":        r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
}


# ── Timer storage ───────────────────────────────────────────────────────────

_active_timers: List[dict] = []


# ── Skill implementations ──────────────────────────────────────────────────

def skill_open_app(args: str) -> str:
    """Open a Windows application by name."""
    app_name = args.strip().lower()
    if not app_name:
        return "No application name provided."

    exe = _APP_REGISTRY.get(app_name)
    if exe:
        # Replace {user} placeholder
        user = os.environ.get("USERNAME", os.environ.get("USER", ""))
        exe = exe.replace("{user}", user)
    else:
        # Try running it directly (works for PATH-accessible apps)
        exe = app_name

    try:
        if exe.startswith("ms-"):
            # UWP / ms-settings links
            os.startfile(exe)
        else:
            subprocess.Popen(
                exe, shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return f"Opened {app_name}."
    except Exception as exc:
        return f"Could not open {app_name}: {exc}"


def skill_close_app(args: str) -> str:
    """Close a running application by name."""
    app_name = args.strip().lower()
    if not app_name:
        return "No application name provided."
    try:
        result = subprocess.run(
            ["taskkill", "/IM", f"{app_name}.exe", "/F"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return f"Closed {app_name}."
        return f"Could not find {app_name} running."
    except Exception as exc:
        return f"Failed to close {app_name}: {exc}"


def skill_web_search(args: str) -> str:
    """Search the web using DuckDuckGo (free, no API key)."""
    query = args.strip()
    if not query:
        return "No search query provided."

    # Try the new 'ddgs' package first, then legacy 'duckduckgo_search'
    ddgs_cls = None
    try:
        from ddgs import DDGS as _DDGS  # type: ignore
        ddgs_cls = _DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _DDGS2  # type: ignore
            ddgs_cls = _DDGS2
        except ImportError:
            pass

    if ddgs_cls is not None:
        try:
            with ddgs_cls() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return _web_search_fallback(query)

            summary_parts = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body = r.get("body", "")
                summary_parts.append(f"{i}. {title}: {body}")

            return "Here is what I found. " + " ".join(summary_parts)
        except Exception as exc:
            return _web_search_fallback(query)
    else:
        return _web_search_fallback(query)


def _web_search_fallback(query: str) -> str:
    """Minimal DuckDuckGo search without extra packages."""
    try:
        url = f"https://api.duckduckgo.com/?q={urlparse.quote(query)}&format=json&no_html=1"
        req = urlreq.Request(url, headers={"User-Agent": "FRIDAY/1.0"})
        with urlreq.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        abstract = data.get("AbstractText", "")
        answer = data.get("Answer", "")
        related = data.get("RelatedTopics", [])

        parts = []
        if answer:
            parts.append(answer)
        if abstract:
            parts.append(abstract)
        for topic in related[:3]:
            text = topic.get("Text", "")
            if text:
                parts.append(text)

        if parts:
            return "Here is what I found. " + " ".join(parts)
        return f"I could not find a clear answer for: {query}"
    except Exception as exc:
        return f"Web search failed: {exc}"


def skill_system_info(args: str) -> str:
    """Get system information: CPU, RAM, disk, battery."""
    try:
        import psutil
    except ImportError:
        return _system_info_basic()

    parts = []
    cpu_pct = psutil.cpu_percent(interval=0.5)
    parts.append(f"CPU usage is {cpu_pct} percent")

    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    parts.append(f"RAM: {used_gb:.1f} of {total_gb:.1f} GB used ({mem.percent} percent)")

    disk = psutil.disk_usage("C:\\")
    free_gb = disk.free / (1024 ** 3)
    parts.append(f"C drive: {free_gb:.0f} GB free")

    bat = psutil.sensors_battery()
    if bat:
        plug = "plugged in" if bat.power_plugged else "on battery"
        parts.append(f"Battery: {bat.percent} percent, {plug}")

    return ". ".join(parts) + "."


def _system_info_basic() -> str:
    """Fallback system info without psutil."""
    info = platform.uname()
    return (
        f"System: {info.system} {info.release}, "
        f"Machine: {info.machine}, "
        f"Processor: {info.processor}."
    )


def skill_clipboard_read(args: str) -> str:
    """Read current clipboard contents."""
    try:
        import pyperclip
        content = pyperclip.paste()
        if content:
            # Truncate for voice
            if len(content) > 300:
                return f"Clipboard contains: {content[:300]}... (truncated)"
            return f"Clipboard contains: {content}"
        return "Clipboard is empty."
    except ImportError:
        # PowerShell fallback
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            content = result.stdout.strip()
            if content:
                if len(content) > 300:
                    return f"Clipboard contains: {content[:300]}... (truncated)"
                return f"Clipboard contains: {content}"
            return "Clipboard is empty."
        except Exception:
            return "Could not read clipboard."


def skill_clipboard_write(args: str) -> str:
    """Write text to clipboard."""
    text = args.strip()
    if not text:
        return "No text to copy."
    try:
        import pyperclip
        pyperclip.copy(text)
        return "Copied to clipboard."
    except ImportError:
        try:
            subprocess.run(
                ["powershell", "-Command", f"Set-Clipboard '{text}'"],
                timeout=5,
            )
            return "Copied to clipboard."
        except Exception:
            return "Could not write to clipboard."


def skill_volume(args: str) -> str:
    """Control system volume using PowerShell/nircmd."""
    action = args.strip().lower()
    if not action:
        return "Specify: mute, unmute, up, down, or a number 0-100."

    ps_base = (
        "Add-Type -TypeDefinition '"
        'using System.Runtime.InteropServices; '
        '[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] '
        "interface IAudioEndpointVolume { int f(); int g(); int h(); int i(); "
        "int SetMasterVolumeLevelScalar(float fLevel, System.Guid pguidEventContext); "
        "int j(); int GetMasterVolumeLevelScalar(out float pfLevel); }"
        "'"
    )

    try:
        if action == "mute":
            subprocess.run(
                ["powershell", "-Command",
                 "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"],
                timeout=5, capture_output=True,
            )
            return "Volume muted."
        elif action == "unmute":
            subprocess.run(
                ["powershell", "-Command",
                 "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"],
                timeout=5, capture_output=True,
            )
            return "Volume unmuted."
        elif action in ("up", "louder"):
            for _ in range(5):
                subprocess.run(
                    ["powershell", "-Command",
                     "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"],
                    timeout=5, capture_output=True,
                )
            return "Volume increased."
        elif action in ("down", "quieter", "lower"):
            for _ in range(5):
                subprocess.run(
                    ["powershell", "-Command",
                     "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"],
                    timeout=5, capture_output=True,
                )
            return "Volume decreased."
        else:
            return f"Unknown volume command: {action}. Use mute, unmute, up, down."
    except Exception as exc:
        return f"Volume control failed: {exc}"


def skill_timer(args: str) -> str:
    """Set a countdown timer. Args: seconds or 'Xm' for minutes."""
    raw = args.strip().lower()
    if not raw:
        return "Specify a duration, like 30 for 30 seconds or 5m for 5 minutes."

    try:
        if raw.endswith("m"):
            seconds = int(raw[:-1]) * 60
        elif raw.endswith("min"):
            seconds = int(raw[:-3]) * 60
        elif raw.endswith("h"):
            seconds = int(raw[:-1]) * 3600
        elif raw.endswith("s"):
            seconds = int(raw[:-1])
        else:
            seconds = int(raw)
    except ValueError:
        return f"Could not parse duration: {raw}"

    if seconds <= 0:
        return "Duration must be positive."
    if seconds > 86400:
        return "Maximum timer is 24 hours."

    def _alarm():
        time.sleep(seconds)
        # Play a beep via PowerShell
        try:
            subprocess.run(
                ["powershell", "-Command",
                 "[console]::beep(800,500); [console]::beep(800,500); [console]::beep(800,500)"],
                timeout=10, capture_output=True,
            )
        except Exception:
            pass
        print(f"\n⏰ Timer done! ({seconds} seconds elapsed)")

    t = threading.Thread(target=_alarm, daemon=True)
    t.start()
    _active_timers.append({"seconds": seconds, "started": time.time(), "thread": t})

    if seconds >= 60:
        mins = seconds // 60
        secs = seconds % 60
        label = f"{mins} minute{'s' if mins != 1 else ''}"
        if secs:
            label += f" {secs} second{'s' if secs != 1 else ''}"
    else:
        label = f"{seconds} second{'s' if seconds != 1 else ''}"

    return f"Timer set for {label}."


def skill_calculator(args: str) -> str:
    """Safely evaluate a math expression."""
    expr = args.strip()
    if not expr:
        return "No expression to evaluate."

    # Clean up spoken math
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = expr.replace("plus", "+").replace("minus", "-")
    expr = expr.replace("times", "*").replace("divided by", "/")
    expr = expr.replace("to the power of", "**")
    expr = expr.replace("mod", "%").replace("modulo", "%")

    # Allow only safe characters
    safe_pattern = re.compile(r'^[\d\s\+\-\*\/\.\(\)\%\,]+$')
    # Also allow math function names
    func_pattern = re.compile(r'\b(sqrt|sin|cos|tan|log|log2|log10|abs|pow|round|pi|e)\b')

    clean = func_pattern.sub('', expr)
    if not safe_pattern.match(clean.replace(' ', '').replace(',', '')):
        return f"I cannot safely evaluate: {expr}"

    # Replace math functions with math module calls
    expr = re.sub(r'\bsqrt\b', 'math.sqrt', expr)
    expr = re.sub(r'\bsin\b', 'math.sin', expr)
    expr = re.sub(r'\bcos\b', 'math.cos', expr)
    expr = re.sub(r'\btan\b', 'math.tan', expr)
    expr = re.sub(r'\blog10\b', 'math.log10', expr)
    expr = re.sub(r'\blog2\b', 'math.log2', expr)
    expr = re.sub(r'\blog\b', 'math.log', expr)
    expr = re.sub(r'\bpi\b', 'math.pi', expr)
    expr = re.sub(r'\be\b', 'math.e', expr)
    expr = re.sub(r'\babs\b', 'abs', expr)
    expr = re.sub(r'\bround\b', 'round', expr)

    try:
        result = eval(expr, {"__builtins__": {}, "math": math, "abs": abs, "round": round, "pow": pow})
        # Format nicely
        if isinstance(result, float):
            if result == int(result):
                return f"The answer is {int(result)}."
            return f"The answer is {result:.6g}."
        return f"The answer is {result}."
    except Exception as exc:
        return f"Could not calculate: {exc}"


def skill_weather(args: str) -> str:
    """Get weather info from wttr.in (free, no API key)."""
    location = args.strip() or "Indore"
    try:
        url = f"https://wttr.in/{urlparse.quote(location)}?format=j1"
        req = urlreq.Request(url, headers={"User-Agent": "FRIDAY/1.0"})
        with urlreq.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "?")
        desc = current.get("weatherDesc", [{}])[0].get("value", "unknown")
        humidity = current.get("humidity", "?")
        feels = current.get("FeelsLikeC", "?")
        wind = current.get("windspeedKmph", "?")

        area = data.get("nearest_area", [{}])[0]
        city = area.get("areaName", [{}])[0].get("value", location)

        return (
            f"Weather in {city}: {desc}, {temp_c} degrees celsius, "
            f"feels like {feels} degrees. "
            f"Humidity is {humidity} percent. "
            f"Wind speed is {wind} kilometers per hour."
        )
    except Exception as exc:
        return f"Could not get weather: {exc}"


def skill_screenshot(args: str) -> str:
    """Take a screenshot and save it to Desktop."""
    try:
        from PIL import ImageGrab
    except ImportError:
        # Fallback: use PowerShell/Snipping Tool
        try:
            subprocess.Popen(["snippingtool", "/clip"], shell=True)
            return "Opened Snipping Tool for screenshot."
        except Exception:
            return "Screenshot requires Pillow: pip install Pillow"

    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(desktop, f"screenshot_{ts}.png")
        img = ImageGrab.grab()
        img.save(path)
        return f"Screenshot saved to Desktop as screenshot_{ts}.png."
    except Exception as exc:
        return f"Screenshot failed: {exc}"


def skill_file_search(args: str) -> str:
    """Search for files by name pattern on common locations."""
    pattern = args.strip().lower()
    if not pattern:
        return "Specify a filename or pattern to search for."

    search_dirs = [
        os.path.expanduser("~\\Desktop"),
        os.path.expanduser("~\\Documents"),
        os.path.expanduser("~\\Downloads"),
    ]

    found = []
    for root_dir in search_dirs:
        if not os.path.isdir(root_dir):
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Limit depth to 3 levels
                depth = dirpath.replace(root_dir, "").count(os.sep)
                if depth > 3:
                    dirnames.clear()
                    continue
                for fn in filenames:
                    if pattern in fn.lower():
                        found.append(os.path.join(dirpath, fn))
                        if len(found) >= 10:
                            break
                if len(found) >= 10:
                    break
        except PermissionError:
            continue

    if not found:
        return f"No files matching '{pattern}' found in Desktop, Documents, or Downloads."

    file_list = ", ".join(os.path.basename(f) for f in found[:5])
    return f"Found {len(found)} file{'s' if len(found) > 1 else ''}: {file_list}."


def skill_datetime(args: str) -> str:
    """Get current date, time, or day information."""
    now = datetime.datetime.now()
    query = args.strip().lower()

    if "date" in query:
        return f"Today is {now.strftime('%A, %d %B %Y')}."
    elif "day" in query:
        return f"Today is {now.strftime('%A')}."
    else:
        return f"It is {now.strftime('%I:%M %p')} on {now.strftime('%A, %d %B %Y')}."


def skill_remind(args: str) -> str:
    """Set a quick reminder (same as timer but with a message)."""
    # Parse "in Xm message" or "X seconds message"
    match = re.match(r'(?:in\s+)?(\d+)\s*(s|sec|seconds?|m|min|minutes?|h|hours?)\s*(.*)', args.strip(), re.IGNORECASE)
    if not match:
        return "Format: remind in 5m check the oven"

    amount = int(match.group(1))
    unit = match.group(2).lower()
    message = match.group(3).strip() or "Time's up!"

    if unit.startswith("m"):
        seconds = amount * 60
    elif unit.startswith("h"):
        seconds = amount * 3600
    else:
        seconds = amount

    def _remind():
        time.sleep(seconds)
        print(f"\n🔔 Reminder: {message}")
        try:
            subprocess.run(
                ["powershell", "-Command",
                 f"[console]::beep(600,300); [console]::beep(800,300); "
                 f"[console]::beep(600,300)"],
                timeout=10, capture_output=True,
            )
        except Exception:
            pass

    threading.Thread(target=_remind, daemon=True).start()

    if seconds >= 60:
        label = f"{seconds // 60} minute{'s' if seconds // 60 != 1 else ''}"
    else:
        label = f"{seconds} second{'s' if seconds != 1 else ''}"

    return f"Reminder set: '{message}' in {label}."


# ── Skill Registry ──────────────────────────────────────────────────────────

@dataclass
class SkillEngine:
    """
    Dispatches [ACTION:skill:args] tags from LLM output.
    """

    debug_fn: Optional[Callable] = None
    skills: Dict[str, Callable[[str], str]] = field(default_factory=dict)

    # Pattern: [ACTION:skill_name:arguments]
    ACTION_PATTERN = re.compile(r'\[ACTION:(\w+):(.*?)\]', re.DOTALL)

    def __post_init__(self):
        self.skills = {
            "open_app":       skill_open_app,
            "close_app":      skill_close_app,
            "web_search":     skill_web_search,
            "system_info":    skill_system_info,
            "clipboard_read": skill_clipboard_read,
            "clipboard_write": skill_clipboard_write,
            "volume":         skill_volume,
            "timer":          skill_timer,
            "calculator":     skill_calculator,
            "weather":        skill_weather,
            "screenshot":     skill_screenshot,
            "file_search":    skill_file_search,
            "datetime":       skill_datetime,
            "remind":         skill_remind,
        }

    def _debug(self, msg: str) -> None:
        if self.debug_fn:
            self.debug_fn(f"[skills] {msg}")

    def list_skills(self) -> str:
        """Human-readable list of available skills."""
        lines = []
        for name, fn in self.skills.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            lines.append(f"  {name}: {doc}")
        return "Available skills:\n" + "\n".join(lines)

    def extract_and_execute(self, text: str) -> tuple[str, list[str]]:
        """
        Find all [ACTION:...] tags in text, execute them,
        and return (cleaned_text, list_of_results).
        """
        results = []
        matches = list(self.ACTION_PATTERN.finditer(text))

        if not matches:
            return text, []

        for match in matches:
            skill_name = match.group(1).strip()
            skill_args = match.group(2).strip()

            self._debug(f"Executing skill: {skill_name}({skill_args})")

            fn = self.skills.get(skill_name)
            if fn:
                try:
                    result = fn(skill_args)
                    results.append(result)
                    self._debug(f"Skill result: {result[:100]}")
                except Exception as exc:
                    error_msg = f"Skill {skill_name} failed: {exc}"
                    results.append(error_msg)
                    self._debug(error_msg)
            else:
                results.append(f"Unknown skill: {skill_name}")

        # Remove action tags from text
        cleaned = self.ACTION_PATTERN.sub("", text).strip()
        # Clean up leftover whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'  +', ' ', cleaned)

        return cleaned, results

    def get_skill_prompt(self) -> str:
        """
        Returns instruction block for the LLM system prompt
        so it knows how to invoke skills.
        """
        return (
            "TOOL USE:\n"
            "You can perform real actions by including action tags in your response.\n"
            "Format: [ACTION:skill_name:arguments]\n"
            "Available actions:\n"
            "  [ACTION:open_app:chrome] — open an application\n"
            "  [ACTION:close_app:notepad] — close an application\n"
            "  [ACTION:web_search:latest python news] — search the web\n"
            "  [ACTION:system_info:] — get CPU, RAM, battery info\n"
            "  [ACTION:clipboard_read:] — read clipboard\n"
            "  [ACTION:clipboard_write:text to copy] — write to clipboard\n"
            "  [ACTION:volume:up/down/mute/unmute] — control volume\n"
            "  [ACTION:timer:30s or 5m] — set a timer\n"
            "  [ACTION:calculator:2+2*3] — evaluate math\n"
            "  [ACTION:weather:Indore] — get weather\n"
            "  [ACTION:screenshot:] — take a screenshot\n"
            "  [ACTION:file_search:report.pdf] — find files\n"
            "  [ACTION:datetime:] — get current date/time\n"
            "  [ACTION:remind:5m check the oven] — set a reminder\n\n"
            "IMPORTANT: Only use action tags when the user explicitly asks you to DO something.\n"
            "For questions, just answer normally without action tags.\n"
            "You may include ONE action tag per response. Put it at the END of your message.\n"
            "After the action tag, do NOT add any more text.\n"
        )
