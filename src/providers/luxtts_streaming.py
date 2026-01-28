"""LuxTTS streaming provider - sentence-level streaming for lower TTFB."""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Generator
import numpy as np

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


class LuxTTSStreamingProvider(TTSProvider):
    """LuxTTS with sentence-level streaming for lower time-to-first-byte.
    
    LuxTTS doesn't support native token-level streaming, but we can achieve
    lower TTFB by splitting text into sentences and generating each separately.
    """

    DEFAULT_PROMPT_PATH = Path(__file__).parent.parent.parent / "outputs" / "audio" / "azure_tts" / "basic.wav"

    def __init__(self):
        config = ProviderConfig(
            name="LuxTTS Streaming",
            pricing_per_1m_chars=0.00,
            supported_languages=["en"],
            default_voice_en="default",
            default_voice_cn="",
            max_chars_per_request=2000,
            supports_streaming=True,
        )
        super().__init__(config)
        self._model = None
        self._device = "cpu"
        self._encoded_prompt = None

    def initialize(self) -> None:
        """Initialize LuxTTS model."""
        vendor_path = Path(__file__).parent.parent.parent / "vendor" / "LuxTTS"
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))

        from zipvoice.luxvoice import LuxTTS

        print("Loading LuxTTS model for streaming...")
        self._model = LuxTTS(
            model_path='YatharthS/LuxTTS',
            device=self._device,
            threads=4,
        )

        prompt_path = self.DEFAULT_PROMPT_PATH
        if not prompt_path.exists():
            raise FileNotFoundError(f"Voice prompt not found: {prompt_path}")

        self._encoded_prompt = self._model.encode_prompt(str(prompt_path), duration=5, rms=0.001)
        self._is_initialized = True

    def _split_into_chunks(self, text: str, chunk_mode: str = "sentence") -> list[str]:
        """Split text into chunks for streaming.

        Args:
            text: Input text to split
            chunk_mode: "sentence" for sentence-level, "phrase" for phrase-level (lower TTFB)

        Minimum chunk size is ~40 chars to ensure stable audio generation.
        """
        MIN_CHUNK_SIZE = 40  # Minimum chars per chunk for stable audio

        if chunk_mode == "phrase":
            # Split by phrases (commas, semicolons, sentence endings)
            # This gives lower TTFB but may affect prosody
            parts = re.split(r'(?<=[,;.!?])\s+', text.strip())
        else:
            # Split by sentences only
            parts = re.split(r'(?<=[.!?])\s+', text.strip())

        # Merge short chunks
        result = []
        buffer = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            candidate = (buffer + " " + part).strip() if buffer else part

            if len(candidate) < MIN_CHUNK_SIZE:
                buffer = candidate
            else:
                if buffer and len(buffer) >= MIN_CHUNK_SIZE:
                    result.append(buffer)
                    buffer = part
                else:
                    result.append(candidate)
                    buffer = ""

        # Add remaining buffer
        if buffer:
            if result and len(buffer) < MIN_CHUNK_SIZE:
                result[-1] = result[-1] + " " + buffer
            else:
                result.append(buffer)

        return result if result else [text]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences (default mode)."""
        return self._split_into_chunks(text, chunk_mode="sentence")

    def generate_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> Generator[bytes, None, None]:
        """Generate speech in streaming chunks.

        Args:
            chunk_mode: "sentence" (default) or "phrase" for lower TTFB
        """
        if not self._is_initialized:
            self.initialize()

        if language != "en":
            raise ValueError(f"LuxTTS only supports English, got: {language}")

        chunk_mode = kwargs.get("chunk_mode", "sentence")
        chunks = self._split_into_chunks(text, chunk_mode=chunk_mode)

        for chunk in chunks:
            audio_tensor = self._model.generate_speech(
                chunk,
                self._encoded_prompt,
                num_steps=kwargs.get("num_steps", 4),
                guidance_scale=kwargs.get("guidance_scale", 3.0),
                t_shift=kwargs.get("t_shift", 0.5),
                speed=kwargs.get("speed", 1.0),
                return_smooth=False
            )
            
            audio_np = audio_tensor.numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            
            audio_np = audio_np / np.max(np.abs(audio_np) + 1e-8)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            yield audio_int16.tobytes()

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech with streaming, measuring TTFB."""
        if not self._is_initialized:
            self.initialize()

        audio_chunks = []
        sample_rate = 48000

        with TimingContext() as timing:
            for chunk in self.generate_stream(text, voice, language, **kwargs):
                if timing.first_byte_time is None:
                    timing.mark_first_byte()
                audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)
        duration = len(audio_data) // 2 / sample_rate  # 16-bit audio

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=timing.ttfb_ms or timing.total_ms,
            ttfb_ms=timing.ttfb_ms,
            characters=len(text),
            provider=self.name,
            voice="cloned",
            language=language,
        )

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        if language and language != "en":
            return []
        return [{"id": "default", "name": "Cloned Voice (Streaming)", "language": "en"}]
