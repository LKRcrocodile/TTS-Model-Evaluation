"""ElevenLabs TTS provider - Turbo model (streaming/low latency)."""

import os
from typing import Optional, Iterator
from elevenlabs import ElevenLabs

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


class ElevenLabsTurboProvider(TTSProvider):
    """ElevenLabs Turbo model - optimized for low latency streaming."""

    # Default ElevenLabs voice IDs (premade voices)
    DEFAULT_VOICES = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "sarah": "EXAVITQu4vr4xnSDxMaL",
        "drew": "29vD33N1CtxCmqQRPOHJ",
    }

    def __init__(self):
        config = ProviderConfig(
            name="ElevenLabs Turbo",
            pricing_per_1m_chars=165.00,
            supported_languages=["en", "zh", "es", "fr", "de", "ja", "ko", "pt", "it", "multilingual"],
            default_voice_en="sarah",
            default_voice_cn="sarah",
            max_chars_per_request=5000,
            supports_streaming=True,
        )
        super().__init__(config)
        self._client: Optional[ElevenLabs] = None
        self._voices_cache: dict = {}

    def initialize(self) -> None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

        # Use global endpoint for auto-routing to closest server (US/Netherlands/Singapore)
        base_url = os.environ.get("ELEVENLABS_BASE_URL", "https://api-global-preview.elevenlabs.io")
        self._client = ElevenLabs(api_key=api_key, base_url=base_url)
        self._voices_cache = self.DEFAULT_VOICES.copy()
        self.config.default_voice_en = self.DEFAULT_VOICES["sarah"]
        self.config.default_voice_cn = self.DEFAULT_VOICES["sarah"]
        self._is_initialized = True

    def _get_voice_id(self, voice_name: str) -> str:
        return self._voices_cache.get(voice_name.lower(), voice_name)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        if not self._is_initialized:
            self.initialize()

        voice = voice or self.get_default_voice(language)
        voice_id = self._get_voice_id(voice)
        model = "eleven_turbo_v2_5"  # Turbo model for low latency

        with TimingContext() as timing:
            audio_generator = self._client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model,
                output_format="pcm_24000",
            )
            audio_chunks = []
            for chunk in audio_generator:
                if timing.first_byte_time is None:
                    timing.mark_first_byte()
                audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)
        sample_rate = 24000
        duration = len(audio_data) / (sample_rate * 2)

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=timing.ttfb_ms or timing.total_ms,  # TTFB for streaming
            ttfb_ms=timing.ttfb_ms,
            characters=len(text),
            provider=self.name,
            voice=voice,
            language=language,
        )

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        return [{"id": v, "name": k, "language": "multilingual"} for k, v in self.DEFAULT_VOICES.items()]
