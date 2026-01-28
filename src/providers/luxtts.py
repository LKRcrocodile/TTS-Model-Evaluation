"""LuxTTS local provider (CPU-compatible)."""

import os
import io
from typing import Optional
import numpy as np

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


class LuxTTSProvider(TTSProvider):
    """LuxTTS local Text-to-Speech provider (runs on CPU)."""

    def __init__(self):
        config = ProviderConfig(
            name="LuxTTS",
            pricing_per_1m_chars=0.00,  # Local - compute cost only
            supported_languages=["en"],
            default_voice_en="default",
            default_voice_cn="",  # Limited Chinese support
            max_chars_per_request=2000,
            supports_streaming=False,
        )
        super().__init__(config)
        self._model = None
        self._device = "cpu"

    def initialize(self) -> None:
        """Initialize LuxTTS model."""
        try:
            from zipvoice.luxvoice import LuxTTS
        except ImportError:
            raise ImportError(
                "LuxTTS not installed. Install with:\n"
                "pip install git+https://github.com/ysharma3501/LuxTTS.git"
            )

        # Use CPU for inference (no GPU required)
        self._model = LuxTTS(
            model_id="YatharthS/LuxTTS",
            device=self._device,
            threads=4,  # Use multiple CPU threads
        )
        self._is_initialized = True

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech using LuxTTS."""
        if not self._is_initialized:
            self.initialize()

        if language != "en":
            raise ValueError(f"LuxTTS only supports English, got: {language}")

        with TimingContext() as timing:
            # LuxTTS returns audio as numpy array
            audio_np, sample_rate = self._model.tts(text)

        # Convert to bytes (16-bit PCM)
        if audio_np.dtype != np.int16:
            # Normalize and convert to int16
            audio_np = (audio_np * 32767).astype(np.int16)

        audio_data = audio_np.tobytes()
        duration = len(audio_np) / sample_rate

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=timing.total_ms,
            characters=len(text),
            provider=self.name,
            voice="default",
            language=language,
        )

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available voices (LuxTTS has one default voice)."""
        if language and language != "en":
            return []

        return [
            {
                "id": "default",
                "name": "Default",
                "language": "en",
            }
        ]
