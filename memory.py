"""
memory.py — Short-term + persistent memory for FRIDAY.

ShortMemory: in-memory rolling buffer (unchanged).
PersistentMemory: saves conversation logs + user facts to disk.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ── Short-term memory (kept for backward compat) ───────────────────────────

@dataclass
class ShortMemory:
    messages: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages.clear()

    def recent(self, count: int = 5) -> list[str]:
        return self.messages[-count:]

    def count(self) -> int:
        return len(self.messages)


# ── Persistent memory ──────────────────────────────────────────────────────

_DEFAULT_MEMORY_DIR = os.path.join(os.path.expanduser("~"), ".friday")
_DEFAULT_MEMORY_FILE = os.path.join(_DEFAULT_MEMORY_DIR, "memory.json")
_DEFAULT_HISTORY_FILE = os.path.join(_DEFAULT_MEMORY_DIR, "history.jsonl")


@dataclass
class PersistentMemory:
    """
    Saves key user facts, preferences, and conversation history to disk.
    Loads automatically on init.
    """
    memory_dir: str = _DEFAULT_MEMORY_DIR
    memory_file: str = _DEFAULT_MEMORY_FILE
    history_file: str = _DEFAULT_HISTORY_FILE

    # In-memory stores
    user_facts: Dict[str, str] = field(default_factory=dict)
    preferences: Dict[str, str] = field(default_factory=dict)
    session_log: List[dict] = field(default_factory=list)
    max_session_log: int = 100

    def __post_init__(self):
        os.makedirs(self.memory_dir, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Load persisted facts/preferences from disk."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.user_facts = data.get("user_facts", {})
                self.preferences = data.get("preferences", {})
            except (json.JSONDecodeError, IOError):
                pass  # corrupted — start fresh

    def _save(self) -> None:
        """Persist facts/preferences to disk."""
        data = {
            "user_facts": self.user_facts,
            "preferences": self.preferences,
            "last_updated": datetime.now().isoformat(),
        }
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    def save_fact(self, key: str, value: str) -> None:
        """Store a user fact (e.g. 'favorite_color' → 'blue')."""
        self.user_facts[key.strip().lower()] = value.strip()
        self._save()

    def get_fact(self, key: str) -> Optional[str]:
        return self.user_facts.get(key.strip().lower())

    def save_preference(self, key: str, value: str) -> None:
        self.preferences[key.strip().lower()] = value.strip()
        self._save()

    def get_preference(self, key: str) -> Optional[str]:
        return self.preferences.get(key.strip().lower())

    def log_interaction(self, user_input: str, assistant_response: str) -> None:
        """Log a conversation turn to the session log and append to history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response,
        }
        self.session_log.append(entry)
        if len(self.session_log) > self.max_session_log:
            self.session_log = self.session_log[-self.max_session_log:]

        # Append to history file
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except IOError:
            pass

    def get_context_summary(self) -> str:
        """
        Build a short context string for the LLM system prompt.
        Includes known facts and recent session context.
        """
        parts = []
        if self.user_facts:
            facts = "; ".join(f"{k}: {v}" for k, v in list(self.user_facts.items())[:10])
            parts.append(f"Known facts about user: {facts}")
        if self.preferences:
            prefs = "; ".join(f"{k}: {v}" for k, v in list(self.preferences.items())[:5])
            parts.append(f"User preferences: {prefs}")
        return ". ".join(parts) if parts else ""

    def clear_facts(self) -> None:
        self.user_facts.clear()
        self.preferences.clear()
        self._save()

    def clear_session(self) -> None:
        self.session_log.clear()

    def recent_history(self, count: int = 5) -> List[dict]:
        """Last N conversation turns from this session."""
        return self.session_log[-count:]
