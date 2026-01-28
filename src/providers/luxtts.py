"""LuxTTS local provider (CPU-compatible voice cloning)."""

import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


class LuxTTSProvider(TTSProvider):
    """LuxTTS local Text-to-Speech provider with voice cloning (runs on CPU).

    LuxTTS is a voice cloning model that requires a reference audio file.
    It uses a default voice prompt from our benchmark audio samples.
    """

    # Default voice prompt (uses Azure TTS output as reference voice)
    DEFAULT_PROMPT_PATH = Path(__file__).parent.parent.parent / "outputs" / "audio" / "azure_tts" / "basic.wav"

    def __init__(self):
        config = ProviderConfig(
            name="LuxTTS",
            pricing_per_1m_chars=0.00,  # Local - compute cost only
            supported_languages=["en"],
            default_voice_en="default",
            default_voice_cn="",  # English only
            max_chars_per_request=2000,
            supports_streaming=False,
        )
        super().__init__(config)
        self._model = None
        self._device = "cpu"
        self._encoded_prompt = None

    def initialize(self) -> None:
        """Initialize LuxTTS model and encode voice prompt."""
        # Add vendor/LuxTTS to path
        vendor_path = Path(__file__).parent.parent.parent / "vendor" / "LuxTTS"
        if str(vendor_path) not in sys.path:
            sys.path.insert(0, str(vendor_path))

        try:
            from zipvoice.luxvoice import LuxTTS
        except ImportError as e:
            raise ImportError(
                f"LuxTTS not installed correctly. Error: {e}\n"
                "Install with: cd vendor/LuxTTS && pip install -r requirements.txt"
            )

        print("Loading LuxTTS model (this may take a moment)...")

        # Use CPU for inference (no GPU required)
        self._model = LuxTTS(
            model_path='YatharthS/LuxTTS',
            device=self._device,
            threads=4,  # Use multiple CPU threads
        )

        # Encode the default voice prompt
        prompt_path = self.DEFAULT_PROMPT_PATH
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Voice prompt not found: {prompt_path}\n"
                "Run Azure TTS benchmark first to generate reference audio."
            )

        print(f"Encoding voice prompt from: {prompt_path}")
        self._encoded_prompt = self._model.encode_prompt(
            str(prompt_path),
            duration=5,
            rms=0.001
        )

        self._is_initialized = True
        print("LuxTTS initialized successfully!")

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech using LuxTTS voice cloning."""
        if not self._is_initialized:
            self.initialize()

        if language != "en":
            raise ValueError(f"LuxTTS only supports English, got: {language}")

        # Generation parameters
        num_steps = kwargs.get("num_steps", 4)
        guidance_scale = kwargs.get("guidance_scale", 3.0)
        t_shift = kwargs.get("t_shift", 0.5)
        speed = kwargs.get("speed", 1.0)

        with TimingContext() as timing:
            # Generate speech using voice cloning
            audio_tensor = self._model.generate_speech(
                text,
                self._encoded_prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=False  # Return 48kHz audio
            )

            # Convert tensor to numpy
            audio_np = audio_tensor.numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()

        # LuxTTS outputs 48kHz audio
        sample_rate = 48000

        # Normalize and convert to int16
        audio_np = audio_np / np.max(np.abs(audio_np) + 1e-8)  # Normalize to [-1, 1]
        audio_int16 = (audio_np * 32767).astype(np.int16)

        audio_data = audio_int16.tobytes()
        duration = len(audio_int16) / sample_rate

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=timing.total_ms,
            characters=len(text),
            provider=self.name,
            voice="cloned",
            language=language,
        )

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available voices (LuxTTS clones from reference audio)."""
        if language and language != "en":
            return []

        return [
            {
                "id": "default",
                "name": "Cloned Voice (from Azure TTS)",
                "language": "en",
                "note": "Voice cloned from reference audio"
            }
        ]
