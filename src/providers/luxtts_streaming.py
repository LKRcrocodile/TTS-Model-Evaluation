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

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences for streaming.

        Merges short sentences to avoid audio generation issues with very short text.
        Minimum chunk size is ~60 chars to ensure stable audio generation.
        """
        # Simple split on sentence-ending punctuation followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # Merge short chunks (min 60 chars per chunk for stable audio)
        MIN_CHUNK_SIZE = 60
        result = []
        buffer = ""

        for s in sentences:
            s = s.strip()
            if not s:
                continue

            candidate = (buffer + " " + s).strip() if buffer else s

            if len(candidate) < MIN_CHUNK_SIZE:
                # Keep accumulating
                buffer = candidate
            else:
                # Buffer is long enough, or adding this would make it too long
                if buffer and len(buffer) >= MIN_CHUNK_SIZE:
                    result.append(buffer)
                    buffer = s
                else:
                    # Merge and add
                    result.append(candidate)
                    buffer = ""

        # Add remaining buffer
        if buffer:
            if result and len(buffer) < MIN_CHUNK_SIZE:
                result[-1] = result[-1] + " " + buffer
            else:
                result.append(buffer)

        return result if result else [text]

    def generate_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> Generator[bytes, None, None]:
        """Generate speech in streaming chunks (sentence by sentence)."""
        if not self._is_initialized:
            self.initialize()

        if language != "en":
            raise ValueError(f"LuxTTS only supports English, got: {language}")

        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            audio_tensor = self._model.generate_speech(
                sentence,
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
