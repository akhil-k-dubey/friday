"""
brain.py — Local LLM brain powered by Ollama.

Major upgrades:
  - True sentence-level streaming (yields as each sentence completes)
  - Deduplication of repeated phrases
  - Tool-use prompt injection for skills
  - Conciseness enforcement
  - Dynamic context (time, date) per request
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from typing import Optional, Generator
from urllib import error, request


# Sentence-end regex: splits on . ! ? followed by space or end of string
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# Filler phrases to strip from the beginning of LLM responses
_FILLER_PREFIXES = [
    "sure,", "sure!", "of course,", "of course!", "absolutely,", "absolutely!",
    "certainly,", "certainly!", "great question,", "great question!",
    "that's a great question,", "that's a great question!",
    "well,", "so,", "okay,", "ok,", "alright,", "right,",
]


@dataclass
class Brain:
    """Local LLM brain powered by Ollama with true sentence streaming."""

    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    model: str = "auto"
    timeout_s: float = 60.0
    debug: bool = False
    active_model: Optional[str] = None
    conversation_history: list[dict] = field(default_factory=list)
    max_history: int = 20

    # Skill prompt injection (set by main.py)
    skill_prompt: str = ""

    # Persistent memory context (set by main.py)
    memory_context: str = ""

    def __post_init__(self) -> None:
        self._build_system_prompt()

    def _build_system_prompt(self) -> None:
        now = datetime.now()
        hour = now.hour
        if hour < 12:
            tod = "morning"
        elif hour < 17:
            tod = "afternoon"
        elif hour < 21:
            tod = "evening"
        else:
            tod = "night"

        self.system_prompt = (
            f"You are FRIDAY, a personal AI voice assistant running locally "
            f"on Akhil's Windows machine. "
            f"Current time: {now.strftime('%I:%M %p')}, "
            f"{now.strftime('%A %d %B %Y')} ({tod}).\n\n"

            "CRITICAL IDENTITY RULES:\n"
            "- You run on llama3 by Meta, served locally via Ollama\n"
            "- You are NOT cloud based, NOT Google, NOT BERT, NOT GPT\n"
            "- You do NOT use multiple models\n"
            "- If asked what model you use say: I run on llama3 locally via Ollama\n"
            "- If asked who made you say: Meta made llama3, Akhil built FRIDAY on top\n"
            "- Never invent model names\n\n"

            "VOICE OUTPUT RULES (CRITICAL):\n"
            "- No markdown, no bullet points, no backticks, no asterisks, no hashtags\n"
            "- For code write it plainly without any formatting symbols\n"
            "- NEVER use numbered lists or bullet lists\n"
            "- Speak in natural flowing sentences — as if talking to a friend\n"
            "- Keep answers concise. For simple questions, 1-2 sentences max\n"
            "- For complex topics, use 3-5 sentences. Rarely exceed this\n"
            "- Do NOT repeat what the user said back to them\n"
            "- Do NOT start with filler phrases like Sure, Of course, Absolutely, etc\n"
            "- Get straight to the answer\n"
            "- Call the user sir occasionally\n\n"

            "About the user Akhil:\n"
            "- CSE Data Science BTech student at SVVV Indore\n"
            "- Building DroidBox, a Linux VM runner for Android\n"
            "- Works with Python, C, Linux, Makefiles, zsh, Kali Linux\n"
            "- Has RTX 4050 GPU\n"
            "- Running FRIDAY on Windows, moving to Ubuntu VM soon\n"
        )

        # Inject skill prompt if available
        if self.skill_prompt:
            self.system_prompt += "\n" + self.skill_prompt

        # Inject persistent memory context
        if self.memory_context:
            self.system_prompt += f"\n{self.memory_context}\n"

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[brain-debug] {message}")

    def _get_json(self, path: str) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _resolve_model(self) -> Optional[str]:
        if self.model != "auto":
            return self.model
        try:
            tags = self._get_json("/api/tags")
            installed = [item.get("name", "") for item in tags.get("models", [])]
            self._debug(f"Installed models: {installed}")

            preferred = [
                "llama3:latest", "llama3", "llama3.1:latest", "llama3.1",
                "llama3.1:8b-instruct", "llama3.2:latest",
                "mistral:latest", "mistral", "mistral:7b-instruct",
                "qwen2.5:7b-instruct", "gemma2:9b", "phi3:latest", "phi3:mini",
            ]

            for candidate in preferred:
                if candidate in installed:
                    self._debug(f"Exact match: {candidate}")
                    return candidate

            for name in installed:
                if any(x in name.lower() for x in ["llama3", "mistral", "llama"]):
                    self._debug(f"Partial match: {name}")
                    return name

            if installed:
                return installed[0]

        except Exception as exc:
            self._debug(f"Model discovery failed: {exc}")
        return None

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting for voice output."""
        text = re.sub(r"```[\w]*\n?", "", text)
        text = re.sub(r"```", "", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _strip_filler(self, text: str) -> str:
        """Remove filler prefixes like 'Sure!', 'Of course,' etc."""
        lower = text.lstrip()
        for filler in _FILLER_PREFIXES:
            if lower.lower().startswith(filler):
                text = lower[len(filler):].lstrip()
                break
        return text

    def _deduplicate(self, text: str) -> str:
        """Remove consecutive duplicate sentences."""
        sentences = _SENTENCE_END.split(text)
        if len(sentences) <= 1:
            return text

        deduped = [sentences[0]]
        for sent in sentences[1:]:
            # Skip if this sentence is very similar to the previous one
            prev_lower = deduped[-1].strip().lower()
            curr_lower = sent.strip().lower()
            if curr_lower and curr_lower != prev_lower:
                deduped.append(sent)

        return " ".join(s.strip() for s in deduped if s.strip())

    def _clean_response(self, text: str) -> str:
        """Full cleaning pipeline for LLM output."""
        text = self._strip_markdown(text)
        text = self._strip_filler(text)
        text = self._deduplicate(text)
        # Remove excessive emoji
        text = re.sub(r'[\U0001F600-\U0001F64F]{2,}', '', text)
        # Collapse multiple spaces
        text = re.sub(r'  +', ' ', text)
        return text.strip()

    def stream_sentences(
        self, user_text: str, memory_hint: str = ""
    ) -> Generator[str, None, str]:
        """
        Stream response from Ollama token by token.
        Yields each SENTENCE as soon as it's complete.
        Returns the full response as the generator return value.
        """
        if not self.enabled:
            return ""

        if self.active_model is None:
            self.active_model = self._resolve_model()
            if self.active_model:
                self._debug(f"Using Ollama model: {self.active_model}")
            else:
                self._debug("No Ollama model available.")
                return ""

        # Rebuild system prompt with fresh time
        self._build_system_prompt()

        messages = [{"role": "system", "content": self.system_prompt}]

        if memory_hint:
            messages.append({
                "role": "system",
                "content": f"Recent conversation context: {memory_hint}"
            })

        messages.extend(self.conversation_history[-(self.max_history * 2):])
        messages.append({"role": "user", "content": user_text.strip()})

        payload = {
            "model": self.active_model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": 400,
                "temperature": 0.7,
                "num_gpu": 99,
                "num_thread": 8,
            }
        }

        url = f"{self.base_url.rstrip('/')}/api/chat"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        full_response = ""
        buffer = ""              # Accumulates tokens until a sentence boundary

        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_response += token
                        buffer += token
                        print(token, end="", flush=True)

                        # Check for sentence boundaries in buffer
                        while True:
                            # Find the first sentence-ending punctuation
                            match = re.search(r'[.!?]\s', buffer)
                            if match:
                                end_idx = match.end()
                                sentence = buffer[:end_idx].strip()
                                buffer = buffer[end_idx:]
                                if sentence:
                                    cleaned = self._clean_response(sentence)
                                    if cleaned:
                                        yield cleaned
                            else:
                                break

                    if chunk.get("done", False):
                        break

            print()  # newline after streaming

        except (error.URLError, TimeoutError, OSError) as exc:
            self._debug(f"Ollama stream failed: {exc}")
            self.active_model = None
            return ""

        # Yield any remaining text in buffer
        if buffer.strip():
            cleaned = self._clean_response(buffer.strip())
            if cleaned:
                yield cleaned

        # Store in conversation history
        if full_response:
            full_clean = self._clean_response(full_response)
            self.conversation_history.append({
                "role": "user", "content": user_text.strip()
            })
            self.conversation_history.append({
                "role": "assistant", "content": full_clean
            })
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = \
                    self.conversation_history[-(self.max_history * 2):]

        return full_response

    def generate(self, user_text: str, memory_hint: str = "") -> Optional[str]:
        """Non-streaming fallback — returns full response at once."""
        sentences = []
        for sentence in self.stream_sentences(user_text, memory_hint):
            sentences.append(sentence)
        full = " ".join(sentences)
        return full.strip() if full else None

    def clear_history(self) -> None:
        self.conversation_history = []
        self._debug("Conversation history cleared.")