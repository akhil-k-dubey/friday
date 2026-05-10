"""
ear.py — Microphone setup, streaming voice capture, and STT.

Upgrades:
  - StreamingVoiceCapture now supports Google STT (not just whisper)
  - Faster silence detection for snappier responses
  - In-memory audio processing (no temp files)
"""
from __future__ import annotations

import importlib
import io
import wave
import queue
import threading
import time
from typing import Any, Optional, Callable


def _try_import(module_name: str, attribute: str | None = None) -> Any:
    """Import a module or attribute, returning None if unavailable."""
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attribute) if attribute else mod
    except (ImportError, AttributeError):
        return None


sr = _try_import("speech_recognition")
sd = _try_import("sounddevice")
np = _try_import("numpy")
WhisperModel = _try_import("faster_whisper", "WhisperModel")


# ──────────────────────────────────────────────
# 1.  FASTER MIC SETUP — eager init, less probing
# ──────────────────────────────────────────────
def setup_mic(
    recognizer: Optional[sr.Recognizer],
    debug: Callable,
    sample_rate: int = 16000,
    channels: int = 1,
    blocksize: int = 1600,
) -> tuple[Optional[sr.Recognizer], bool, str]:

    if sr is None:
        debug("SpeechRecognition not installed.")
        return None, False, "none"

    # --- Prefer sounddevice for low-latency streaming ---
    if sd is not None and np is not None:
        try:
            sd.check_input_settings(
                device=None,
                channels=channels,
                samplerate=sample_rate,
                dtype="float32",
            )
            debug(f"Mic backend: sounddevice_stream.")
            return sr.Recognizer(), True, "sounddevice_stream"
        except Exception:
            debug("sounddevice not available for this device, falling back to SR.")

    # --- Fallback: speech_recognition microphone ---
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5
        with sr.Microphone(sample_rate=sample_rate):
            pass
        debug("Mic backend: speech_recognition_mic.")
        return recognizer, True, "speech_recognition_mic"
    except Exception:
        debug("Mic setup failed completely.")
        return None, False, "none"


# ──────────────────────────────────────────────
# 2.  STT SETUP
# ──────────────────────────────────────────────
def setup_stt_backend(
    stt_backend: str,
    whisper_model_size: str,
    whisper_device: str,
    whisper_compute_type: str,
    debug: Callable,
) -> tuple[str, Optional[object]]:
    if stt_backend != "whisper":
        debug("STT backend: google")
        return "google", None

    if WhisperModel is None:
        debug("faster-whisper not installed. Falling back to google.")
        return "google", None

    try:
        model = WhisperModel(
            whisper_model_size,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
        debug(
            f"STT backend: whisper ({whisper_model_size}, "
            f"device={whisper_device}, compute={whisper_compute_type})"
        )
        return "whisper", model
    except Exception as exc:
        debug(f"Whisper init failed, fallback to google: {exc}")
        return "google", None


# ──────────────────────────────────────────────
# 3.  STREAMING VOICE CAPTURE
#     Now supports both Whisper AND Google STT
# ──────────────────────────────────────────────
class StreamingVoiceCapture:
    """
    Captures audio via sounddevice in small chunks, detects silence
    boundaries, and transcribes utterances in real-time.
    Gives the Copilot-like feel of responding as you speak.
    """

    def __init__(
        self,
        stt_backend: str,
        whisper_model: Optional[object],
        recognizer: Optional[sr.Recognizer],
        stt_language: str,
        debug: Callable,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 1600,          # 100 ms chunks
        silence_seconds: float = 0.5,   # shorter = snappier
        min_audio_duration: float = 0.3,
        rms_threshold: float = 0.008,   # voice activity threshold
    ):
        self.stt_backend = stt_backend
        self.whisper_model = whisper_model
        self.recognizer = recognizer
        self.stt_language = stt_language
        self.debug = debug
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.silence_threshold = silence_seconds
        self.min_audio_duration = min_audio_duration
        self.rms_threshold = rms_threshold

        self._audio_queue: queue.Queue = queue.Queue()
        self._stream = None
        self._running = False

    # ── sounddevice callback ──
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.debug(f"Stream status: {status}")
        self._audio_queue.put(indata.copy())

    # ── start/stop ──
    def start(self):
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self.debug("Streaming voice capture started.")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.debug("Streaming voice capture stopped.")

    def flush(self):
        """Drain any buffered audio (call after TTS finishes)."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    # ── listen for one utterance (blocking) ──
    def listen_for_utterance(self) -> Optional[str]:
        """
        Blocks until the user stops speaking for `silence_threshold` seconds,
        then transcribes the entire utterance.
        """
        if np is None:
            return None

        buffer: list = []
        silence_start: Optional[float] = None
        utterance_started = False

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.05)
            except queue.Empty:
                if buffer and utterance_started:
                    if silence_start is None:
                        silence_start = time.perf_counter()
                    elif (time.perf_counter() - silence_start) >= self.silence_threshold:
                        return self._transcribe_buffer(buffer)
                continue

            rms = float(np.sqrt(np.mean(chunk**2)))

            if rms < self.rms_threshold:
                # Silence
                if utterance_started:
                    if silence_start is None:
                        silence_start = time.perf_counter()
                    elif (time.perf_counter() - silence_start) >= self.silence_threshold:
                        return self._transcribe_buffer(buffer)
                    # Keep a bit of trailing silence
                    if silence_start and (time.perf_counter() - silence_start) < 0.2:
                        buffer.append(chunk)
            else:
                # Speech detected
                utterance_started = True
                silence_start = None
                buffer.append(chunk)

        # Stream stopped mid-utterance
        if buffer and utterance_started:
            return self._transcribe_buffer(buffer)
        return None

    # ── transcribe buffered audio ──
    def _transcribe_buffer(self, buffer: list) -> Optional[str]:
        if not buffer or np is None:
            return None

        audio_np = np.concatenate(buffer, axis=0).flatten()

        # Check minimum duration
        duration = len(audio_np) / self.sample_rate
        if duration < self.min_audio_duration:
            self.debug(f"Audio too short ({duration:.2f}s), skipping.")
            return None

        # ── Whisper path ──
        if self.stt_backend == "whisper" and self.whisper_model is not None:
            return self._transcribe_whisper(audio_np)

        # ── Google STT path (via speech_recognition) ──
        if self.recognizer is not None:
            return self._transcribe_google(audio_np)

        self.debug("No STT backend available for streaming.")
        return None

    def _transcribe_whisper(self, audio_np) -> Optional[str]:
        """Transcribe with faster-whisper (in-memory)."""
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                int16_audio = (audio_np * 32767).astype(np.int16)
                wav.writeframes(int16_audio.tobytes())
            buf.seek(0)

            segments, _ = self.whisper_model.transcribe(
                buf, language="en",
                beam_size=1, best_of=1, vad_filter=True,
            )
            transcript = " ".join(s.text.strip() for s in segments).strip()
            if transcript:
                self.debug(f"STT (streaming whisper): {transcript}")
                return transcript
            self.debug("STT (streaming whisper): <empty>")
            return None
        except Exception as exc:
            self.debug(f"STT whisper streaming failed: {exc}")
            return None

    def _transcribe_google(self, audio_np) -> Optional[str]:
        """Transcribe with Google STT via speech_recognition (in-memory)."""
        if sr is None or self.recognizer is None:
            return None
        try:
            # Convert float32 to int16 PCM
            int16_audio = (audio_np * 32767).astype(np.int16)
            audio_data = sr.AudioData(
                int16_audio.tobytes(),
                sample_rate=self.sample_rate,
                sample_width=2,
            )
            transcript = self.recognizer.recognize_google(
                audio_data, language=self.stt_language
            )
            if transcript:
                self.debug(f"STT (streaming google): {transcript}")
                return transcript.strip()
            return None
        except sr.UnknownValueError:
            self.debug("STT (streaming google): could not understand")
            return None
        except Exception as exc:
            self.debug(f"STT (streaming google) failed: {exc}")
            return None


# ──────────────────────────────────────────────
# 4.  LEGACY bridge — same interface as original
# ──────────────────────────────────────────────
def transcribe_audio_data(
    recognizer: Optional[sr.Recognizer],
    audio_data: sr.AudioData,
    stt_backend: str,
    whisper_model: Optional[object],
    stt_language: str,
    debug: Callable,
) -> Optional[str]:
    # In-memory whisper path
    if stt_backend == "whisper" and whisper_model is not None:
        try:
            raw = audio_data.get_raw_data()
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(audio_data.sample_rate)
                wav.writeframes((samples * 32767).astype(np.int16).tobytes())
            buf.seek(0)

            segments, _ = whisper_model.transcribe(
                buf, language="en",
                beam_size=1, best_of=1, vad_filter=True,
            )
            transcript = " ".join(s.text.strip() for s in segments).strip()
            if transcript:
                debug(f"STT transcript (whisper in-memory): {transcript}")
                return transcript
            debug("STT transcript (whisper in-memory): <empty>")
            return None
        except Exception as exc:
            debug(f"STT whisper in-memory failed: {exc}")
            return None

    # Google fallback
    if recognizer is None:
        return None
    try:
        transcript = recognizer.recognize_google(audio_data, language=stt_language)
        debug(f"STT transcript (google): {transcript}")
        return transcript
    except Exception as exc:
        debug(f"STT google failed: {type(exc).__name__}: {exc}")
        return None