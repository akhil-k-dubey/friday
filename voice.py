"""
voice.py — Voice I/O pipeline for FRIDAY.

Upgrades:
  - Streaming mic input via StreamingVoiceCapture (real-time, Copilot-like)
  - speak_pregenerated() method (was missing, called from main.py)
  - Sentence-pipelined TTS: queue multiple sentences for back-to-back playback
  - Barge-in support: stop playback when user starts speaking
  - Cleaner edge TTS integration with better error handling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import asyncio
import os
import subprocess
import tempfile
import time
import threading
import queue

from ear import (
    np, sd, sr,
    setup_mic, setup_stt_backend, transcribe_audio_data,
    StreamingVoiceCapture, _try_import,
)

edge_tts = _try_import("edge_tts")

# ── FRIDAY voice config ────────────────────────────────────────────────────
FRIDAY_VOICE = "en-IE-EmilyNeural"
FRIDAY_RATE  = "+8%"
FRIDAY_PITCH = "+0Hz"
FFPLAY       = "ffplay"


@dataclass
class VoiceIO:
    """
    Full-duplex voice pipeline:
      IN  → StreamingVoiceCapture → Google STT / Whisper
      OUT → Edge TTS (sentence pipeline) → ffplay (instant)
    """

    voice_mode: bool = True
    mic_enabled: bool = False
    tts_enabled: bool = True
    recognizer: Optional[object] = None
    mic_backend: str = "none"
    stt_backend: str = "google"
    whisper_model_size: str = "base.en"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    stt_language: str = "en-IN"
    mic_phrase_seconds: int = 5
    min_input_rms: float = 80.0
    post_tts_cooldown_s: float = 0.8
    debug: bool = False
    whisper_model: Optional[object] = None
    tts_backend: str = "auto"

    # Internal state
    _last_tts_time: float = 0.0
    _pregen_queue: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _tts_pipeline: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _playback_stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _streaming_capture: Optional[StreamingVoiceCapture] = field(default=None, repr=False)

    # Expose for main.py status line
    tts_effective_backend: str = "edge"

    def __post_init__(self) -> None:
        self.recognizer, self.mic_enabled, self.mic_backend = setup_mic(
            self.recognizer, self._debug
        )
        self.stt_backend, self.whisper_model = setup_stt_backend(
            self.stt_backend,
            self.whisper_model_size,
            self.whisper_device,
            self.whisper_compute_type,
            self._debug,
        )

        if edge_tts is None:
            self.tts_enabled = False
            self.tts_effective_backend = "none"
            self._debug("edge-tts not installed. TTS disabled.")
        else:
            self._debug(f"TTS: Edge TTS ({FRIDAY_VOICE}) via ffplay — FRIDAY voice active.")

        # Initialize streaming capture if sounddevice is available
        if self.mic_enabled and self.mic_backend == "sounddevice_stream" and sd is not None:
            self._streaming_capture = StreamingVoiceCapture(
                stt_backend=self.stt_backend,
                whisper_model=self.whisper_model,
                recognizer=self.recognizer,
                stt_language=self.stt_language,
                debug=self._debug,
            )
            self._streaming_capture.start()
            self._debug("Streaming voice capture initialized and running.")

    def _debug(self, msg: str) -> None:
        if self.debug:
            print(f"[voice-debug] {msg}")

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._streaming_capture:
            self._streaming_capture.stop()
            self._streaming_capture = None
        self._playback_stop.set()

    # ── Pre-generation (background TTS) ───────────────────────────────────────

    def pregen_start(self, text: str) -> None:
        """
        Fire Edge TTS generation in a background thread.
        Call this the moment you have the response text.
        """
        if not text.strip() or not self.tts_enabled or edge_tts is None:
            return

        # Flush old results
        while not self._pregen_queue.empty():
            try:
                old = self._pregen_queue.get_nowait()
                if old and os.path.exists(old):
                    try:
                        os.unlink(old)
                    except OSError:
                        pass
            except queue.Empty:
                break

        def _run():
            async def _gen():
                communicate = edge_tts.Communicate(
                    text, voice=FRIDAY_VOICE,
                    rate=FRIDAY_RATE, pitch=FRIDAY_PITCH,
                )
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    path = f.name
                await communicate.save(path)
                return path

            try:
                path = asyncio.run(_gen())
                self._pregen_queue.put(path)
                self._debug(f"Pre-gen done: {os.path.basename(path)}")
            except Exception as e:
                self._debug(f"Pre-gen failed: {e}")
                self._pregen_queue.put(None)

        threading.Thread(target=_run, daemon=True).start()
        self._debug("Pre-gen started.")

    def speak_pregenerated(self, timeout: float = 15.0) -> bool:
        """
        Wait for pre-generated audio then play it instantly.
        Returns True on success.
        """
        if not self.voice_mode or not self.tts_enabled:
            return False
        try:
            path = self._pregen_queue.get(timeout=timeout)
        except queue.Empty:
            self._debug("Pre-gen timed out.")
            return False

        if not path or not os.path.exists(path):
            return False

        try:
            self._ffplay(path)
            self._last_tts_time = time.monotonic()
            return True
        except Exception as e:
            self._debug(f"Playback failed: {e}")
            return False
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ── Sentence pipeline TTS ─────────────────────────────────────────────────

    def speak_sentence_pipeline(self, sentences: list[str]) -> None:
        """
        Generate and play multiple sentences with minimal gap.
        Generates sentence N+1 while playing sentence N.
        """
        if not self.voice_mode or not self.tts_enabled or not sentences:
            return

        self._playback_stop.clear()
        paths: list[Optional[str]] = [None] * len(sentences)
        gen_done = threading.Event()

        def _generate_all():
            for i, sentence in enumerate(sentences):
                if self._playback_stop.is_set():
                    break
                try:
                    path = self._generate_tts_sync(sentence)
                    paths[i] = path
                except Exception as e:
                    self._debug(f"Pipeline TTS gen failed for sentence {i}: {e}")
            gen_done.set()

        # Start generation in background
        gen_thread = threading.Thread(target=_generate_all, daemon=True)
        gen_thread.start()

        # Play each sentence as soon as it's ready
        for i in range(len(sentences)):
            if self._playback_stop.is_set():
                break

            # Wait for this sentence to be generated
            while paths[i] is None and not gen_done.is_set():
                time.sleep(0.05)
                if self._playback_stop.is_set():
                    break

            if self._playback_stop.is_set():
                break

            path = paths[i]
            if path and os.path.exists(path):
                try:
                    self._ffplay(path)
                except Exception as e:
                    self._debug(f"Pipeline playback failed: {e}")
                finally:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        self._last_tts_time = time.monotonic()

        # Cleanup any unplayed files
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def stop_playback(self) -> None:
        """Barge-in: signal playback to stop."""
        self._playback_stop.set()
        # Kill any running ffplay process
        try:
            subprocess.run(
                ["taskkill", "/IM", "ffplay.exe", "/F"],
                capture_output=True, timeout=3,
            )
        except Exception:
            pass

    # ── Speak (direct, single-shot) ───────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synchronous speak — generates and plays in one shot."""
        if not self.voice_mode or not text.strip() or not self.tts_enabled:
            return

        try:
            path = self._generate_tts_sync(text)
            if path:
                self._ffplay(path)
                self._last_tts_time = time.monotonic()
                self._debug(f"Spoke: {text[:60]}")
        except Exception as e:
            self._debug(f"speak() failed: {e}")
        finally:
            try:
                if 'path' in locals() and path:
                    os.unlink(path)
            except OSError:
                pass

    def _generate_tts_sync(self, text: str) -> Optional[str]:
        """Generate TTS audio file synchronously. Returns file path."""
        if edge_tts is None:
            return None

        async def _gen():
            communicate = edge_tts.Communicate(
                text, voice=FRIDAY_VOICE,
                rate=FRIDAY_RATE, pitch=FRIDAY_PITCH,
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                path = f.name
            await communicate.save(path)
            return path

        return asyncio.run(_gen())

    def _ffplay(self, path: str) -> None:
        """Play MP3 instantly — no window, no delay."""
        subprocess.run(
            [FFPLAY, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            check=True,
        )

    # ── Listen (streaming or legacy) ──────────────────────────────────────────

    def listen(self) -> Optional[str]:
        """
        Listen for user speech. Uses streaming capture if available,
        falls back to legacy SR mic recording.
        """
        if not self.voice_mode or not self.mic_enabled or sr is None:
            return None

        # Don't listen while still speaking (cooldown)
        if time.monotonic() - self._last_tts_time < self.post_tts_cooldown_s:
            return None

        # ── Streaming path (preferred) ──
        if self._streaming_capture:
            self._streaming_capture.flush()  # Clear any stale audio
            return self._streaming_capture.listen_for_utterance()

        # ── Legacy SR mic path ──
        if self.mic_backend == "speech_recognition_mic":
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(
                        source, timeout=3, phrase_time_limit=6
                    )
                return transcribe_audio_data(
                    self.recognizer, audio,
                    self.stt_backend, self.whisper_model,
                    self.stt_language, self._debug,
                )
            except Exception as e:
                self._debug(f"STT (sr mic) failed: {e}")
                return None

        # ── Legacy sounddevice record path ──
        if self.mic_backend in ("sounddevice_record", "sounddevice_stream") and not self._streaming_capture:
            try:
                sample_rate = 16000
                frames = int(sample_rate * self.mic_phrase_seconds)
                audio = sd.rec(
                    frames, samplerate=sample_rate,
                    channels=1, dtype=np.int16
                )
                sd.wait()
                pcm16 = audio.reshape(-1)
                rms = float(np.sqrt(np.mean(np.square(pcm16.astype(np.float32)))))
                if rms < self.min_input_rms:
                    self._debug(f"STT skipped: low input level (rms={rms:.1f})")
                    return None
                audio_data = sr.AudioData(
                    pcm16.tobytes(), sample_rate=sample_rate, sample_width=2
                )
                return transcribe_audio_data(
                    self.recognizer, audio_data,
                    self.stt_backend, self.whisper_model,
                    self.stt_language, self._debug,
                )
            except Exception as e:
                self._debug(f"STT (sounddevice) failed: {e}")
                return None

        return None