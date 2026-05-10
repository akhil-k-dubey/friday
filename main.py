"""
main.py — FRIDAY Advanced AI Assistant.

Architecture:
  1. LLM streams tokens → accumulates sentences
  2. Each sentence is immediately queued for TTS generation
  3. TTS plays sentence-by-sentence with minimal gaps
  4. Skill actions are detected, executed, and results spoken
  5. Persistent memory logs every interaction

All tools are FREE FOREVER — no API keys required.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
import argparse
import sys

from assistant import Friday
from brain import Brain
from commands import is_shutdown_command, is_wake_command
from memory import PersistentMemory
from skills import SkillEngine
from voice import VoiceIO


def run(
    default_voice_mode: bool = True,
    debug_voice: bool = False,
    stt_backend: str = "google",
    whisper_model_size: str = "base.en",
    stt_language: str = "en-IN",
    tts_backend: str = "auto",
    llm_backend: str = "ollama",
    llm_model: str = "auto",
    llm_base_url: str = "http://127.0.0.1:11434",
    llm_timeout: float = 60.0,
    llm_enabled: bool = True,
    always_on: bool = False,
) -> None:
    # ── Initialize components ─────────────────────────────────────────────────
    persistent_memory = PersistentMemory()
    skill_engine = SkillEngine(debug_fn=lambda m: print(f"[skill-debug] {m}") if debug_voice else None)

    brain = Brain(
        enabled=llm_enabled and llm_backend == "ollama",
        base_url=llm_base_url,
        model=llm_model,
        timeout_s=llm_timeout,
        debug=debug_voice,
    )

    agent = Friday(
        name="friday",
        brain=brain,
        always_on=always_on,
        persistent_memory=persistent_memory,
        skills=skill_engine,
    )

    voice = VoiceIO(
        debug=debug_voice,
        stt_backend=stt_backend,
        whisper_model_size=whisper_model_size,
        stt_language=stt_language,
        tts_backend=tts_backend,
    )
    voice.voice_mode = default_voice_mode

    # ── Boot messages ─────────────────────────────────────────────────────────
    online_message = (
        f"{agent.name.title()} online. "
        f"Systems restored at {datetime.now().strftime('%I:%M %p')}."
    )
    print(online_message)
    voice.speak(online_message)

    if not always_on:
        standby_message = agent.standby()
        print(standby_message)
        voice.speak(standby_message)
    else:
        agent.activated = True
        ready_msg = f"{agent.name.title()} is ready and listening."
        print(ready_msg)

    # ── Voice status line ─────────────────────────────────────────────────────
    if voice.voice_mode and voice.mic_enabled and voice.tts_enabled:
        status_line = (
            f"Voice: READY "
            f"(mic={voice.mic_backend}, TTS={voice.tts_effective_backend})"
        )
    elif voice.voice_mode and (voice.mic_enabled or voice.tts_enabled):
        status_line = (
            "Voice: PARTIAL "
            f"(mic={voice.mic_backend}, "
            f"TTS={voice.tts_effective_backend if voice.tts_enabled else 'off'})"
        )
    elif voice.voice_mode:
        status_line = "Voice: FALLBACK (text-only; mic/TTS unavailable)"
    else:
        status_line = "Voice: DISABLED (text-only mode)"
    print(status_line)

    if brain.enabled:
        model_name = brain.active_model or brain.model
        print(f"Brain: Ollama ({model_name})")
    else:
        print("Brain: OFFLINE (rule-based mode)")

    print(f"Skills: {len(skill_engine.skills)} available")
    print("Say 'help' for commands or just talk naturally.\n")

    if debug_voice:
        print("[debug] Debug mode enabled.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while agent.awake:
        user_input: Optional[str] = None

        # ── Voice input ──
        if voice.voice_mode and voice.mic_enabled:
            spoken = voice.listen()
            if spoken:
                print(f"You (voice) > {spoken}")
                user_input = spoken

        # ── Text input fallback ──
        if not user_input and (not voice.voice_mode or not voice.mic_enabled):
            try:
                user_input = input("You > ")
            except (EOFError, KeyboardInterrupt):
                print("\nPowering down. Until next time.")
                break

        if not user_input:
            continue

        normalized = user_input.strip().lower()

        # ── Shutdown ──────────────────────────────────────────────────────────
        if is_shutdown_command(normalized):
            response = "Powering down. Until next time."
            agent.awake = False
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            persistent_memory.log_interaction(user_input, response)
            break

        # ── Wake word (standby mode) ──────────────────────────────────────────
        if not agent.activated:
            if is_wake_command(normalized, agent.name):
                response = agent.wake()
                print(f"{agent.name.title()} > {response}")
                voice.speak(response)
            else:
                response = f"In standby. Say 'wake up {agent.name}' to activate."
                print(f"{agent.name.title()} > {response}")
                voice.speak(response)
            continue

        # ── Already awake, wake word again ────────────────────────────────────
        if is_wake_command(normalized, agent.name):
            response = "I am already online and listening."
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            continue

        # ── Voice mode toggle ─────────────────────────────────────────────────
        if normalized in ("voice on", "enable voice", "turn voice on"):
            voice.voice_mode = True
            response = "Voice mode enabled."
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            continue

        if normalized in ("voice off", "disable voice", "turn voice off"):
            voice.voice_mode = False
            response = "Voice mode disabled."
            print(f"{agent.name.title()} > {response}")
            continue

        # ── Handle with streaming brain ───────────────────────────────────────
        if agent.brain and agent.brain.enabled:
            _handle_brain_response(agent, voice, user_input, persistent_memory)
        else:
            # Fallback — rule based responses from assistant.py
            response = agent.handle(user_input)
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            persistent_memory.log_interaction(user_input, response)

        if not agent.awake:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    voice.shutdown()
    print("FRIDAY shut down cleanly.")


def _handle_brain_response(
    agent: "Friday",
    voice: "VoiceIO",
    user_input: str,
    persistent_memory: "PersistentMemory",
) -> None:
    """
    Stream LLM response sentence-by-sentence.
    TTS generation starts on sentence 1 while sentence 2 is still being generated.
    Skill actions are detected and executed inline.
    """
    sentence_gen = agent.brain.stream_sentences(
        user_input,
        memory_hint=" | ".join(agent.memory.recent(5))
    )

    print(f"{agent.name.title()} > ", end="", flush=True)

    sentences: list[str] = []
    full_response = ""
    first_sentence_fired = False

    for sentence in sentence_gen:
        if not sentence.strip():
            continue

        sentences.append(sentence)
        full_response += sentence + " "

        # Fire TTS pre-generation on the FIRST sentence immediately
        # while llama3 is still generating the rest
        if not first_sentence_fired:
            voice.pregen_start(sentence)
            first_sentence_fired = True

    full_response = full_response.strip()

    # ── Skill dispatch ────────────────────────────────────────────────────
    skill_results = []
    if full_response:
        cleaned_response, skill_results = agent.process_skill_actions(full_response)
        if skill_results:
            full_response = cleaned_response
            for result in skill_results:
                print(f"\n  [Skill] {result}")
            for result in skill_results:
                if result:
                    sentences.append(result)

    # ── TTS — speak via pipeline ──────────────────────────────────────────
    if sentences:
        speak_sentences = [
            s for s in sentences
            if s.strip() and not s.startswith("[ACTION:")
        ]

        if speak_sentences:
            if first_sentence_fired and len(speak_sentences) == 1:
                # Only one sentence — play the pre-generated audio directly
                played = voice.speak_pregenerated()
                if not played:
                    voice.speak(speak_sentences[0])
            elif first_sentence_fired and len(speak_sentences) > 1:
                # Multiple sentences:
                # First sentence is already pre-generating in background
                # Play it first, then pipeline the rest
                played = voice.speak_pregenerated()
                if not played:
                    voice.speak(speak_sentences[0])
                # Speak remaining sentences via pipeline
                if len(speak_sentences) > 1:
                    voice.speak_sentence_pipeline(speak_sentences[1:])
            else:
                # Fallback — no pre-gen fired (empty response case)
                voice.speak_sentence_pipeline(speak_sentences)

    # ── Memory ────────────────────────────────────────────────────────────
    agent.memory.add(user_input)
    if full_response:
        persistent_memory.log_interaction(user_input, full_response)
        for result in skill_results:
            if result:
                persistent_memory.log_interaction("[skill_result]", result)

    # ── Skill dispatch ────────────────────────────────────────────────────
    skill_results = []
    if full_response:
        cleaned_response, skill_results = agent.process_skill_actions(full_response)
        if skill_results:
            full_response = cleaned_response
            # Print skill results
            for result in skill_results:
                print(f"  [Skill] {result}")
            # Add skill results to what gets spoken
            for result in skill_results:
                if result:
                    sentences.append(result)

    # ── TTS — speak all sentences via pipeline ────────────────────────────
    if sentences:
        # Filter out empty sentences and action tags
        speak_sentences = [
            s for s in sentences
            if s.strip() and not s.startswith("[ACTION:")
        ]

        if speak_sentences:
            voice.speak_sentence_pipeline(speak_sentences)

    # ── Memory ────────────────────────────────────────────────────────────
    agent.memory.add(user_input)
    if full_response:
        persistent_memory.log_interaction(user_input, full_response)
        if skill_results:
            for result in skill_results:
                persistent_memory.log_interaction("[skill_result]", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FRIDAY — Advanced AI Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                    # Default: voice + Ollama\n"
            "  python main.py --always-on         # Skip standby, always listen\n"
            "  python main.py --no-voice          # Text-only mode\n"
            "  python main.py --debug-voice       # Show voice diagnostics\n"
            "  python main.py --stt-backend whisper --whisper-model base.en\n"
        ),
    )
    parser.add_argument(
        "--voice",
        dest="voice",
        action="store_true",
        help="Start in voice mode (default).",
    )
    parser.add_argument(
        "--no-voice",
        dest="voice",
        action="store_false",
        help="Start in text-only mode.",
    )
    parser.add_argument(
        "--always-on",
        action="store_true",
        help="Skip standby mode — FRIDAY is always listening (Copilot-like).",
    )
    parser.add_argument(
        "--debug-voice",
        action="store_true",
        help="Print microphone/TTS/skill diagnostics.",
    )
    parser.add_argument(
        "--stt-backend",
        choices=["google", "whisper"],
        default="google",
        help="Speech-to-text backend.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="Whisper model size (tiny.en, base.en, small.en, etc).",
    )
    parser.add_argument(
        "--stt-language",
        default="en-IN",
        help="STT language locale (en-IN, hi-IN, en-US).",
    )
    parser.add_argument(
        "--tts-backend",
        choices=["auto", "pyttsx3", "powershell", "none"],
        default="auto",
        help="TTS backend. Default 'auto' uses Edge TTS with FRIDAY voice.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "none"],
        default="ollama",
        help="LLM backend.",
    )
    parser.add_argument(
        "--llm-model",
        default="auto",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for LLM requests.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM, use rule-based responses only.",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List available skills and exit.",
    )
    parser.set_defaults(voice=True)

    args = parser.parse_args(sys.argv[1:])

    # Quick skill list and exit
    if args.list_skills:
        engine = SkillEngine()
        print(engine.list_skills())
        sys.exit(0)

    run(
        default_voice_mode=args.voice,
        debug_voice=args.debug_voice,
        stt_backend=args.stt_backend,
        whisper_model_size=args.whisper_model,
        stt_language=args.stt_language,
        tts_backend=args.tts_backend,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_timeout=args.llm_timeout,
        llm_enabled=not args.no_llm,
        always_on=args.always_on,
    )