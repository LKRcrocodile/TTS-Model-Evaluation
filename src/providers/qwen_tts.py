"""Qwen3-TTS provider via DashScope API."""

import os
from typing import Optional
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


# Available Qwen3-TTS voices (shared across providers)
QWEN_VOICES = {
    # Multilingual voices
    "loongstella": {"name": "Stella", "language": "multilingual", "gender": "female"},
    "loongbella": {"name": "Bella", "language": "multilingual", "gender": "female"},
    "loongman": {"name": "Man", "language": "multilingual", "gender": "male"},
    # Chinese voices
    "longxiaochun": {"name": "Xiaochun", "language": "zh", "gender": "female"},
    "longxiaoxia": {"name": "Xiaoxia", "language": "zh", "gender": "female"},
    "longyuan": {"name": "Yuan", "language": "zh", "gender": "male"},
    "longwan": {"name": "Wan", "language": "zh", "gender": "female"},
    "longjing": {"name": "Jing", "language": "zh", "gender": "female"},
    "longcheng": {"name": "Cheng", "language": "zh", "gender": "male"},
}


class QwenTTSProvider(TTSProvider):
    """Qwen3-TTS Standard via DashScope API (non-streaming, higher quality)."""

    def __init__(self):
        config = ProviderConfig(
            name="Qwen3-TTS",
            pricing_per_1m_chars=10.00,  # DashScope pricing
            supported_languages=["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "multilingual"],
            default_voice_en="loongstella",
            default_voice_cn="longxiaochun",
            max_chars_per_request=5000,
            supports_streaming=False,
        )
        super().__init__(config)

    def initialize(self) -> None:
        """Initialize DashScope client."""
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        dashscope.api_key = api_key
        self._is_initialized = True

    def _get_voice(self, voice: Optional[str], language: str) -> str:
        """Get appropriate voice for language."""
        if voice:
            return voice
        if language == "multilingual" or language == "en":
            return "loongstella"
        elif language == "zh":
            return "longxiaochun"
        return self.get_default_voice(language)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech using Qwen3-TTS (non-streaming, wait for complete audio)."""
        if not self._is_initialized:
            self.initialize()

        voice = self._get_voice(voice, language)
        model = kwargs.get("model", "qwen3-tts-flash")

        audio_chunks = []
        sample_rate = 24000

        with TimingContext() as timing:
            synthesizer = SpeechSynthesizer(
                model=model,
                voice=voice,
            )

            # Collect all audio (non-streaming behavior - report total time)
            for chunk in synthesizer.streaming_call(text):
                if chunk:
                    audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)
        duration = len(audio_data) / (sample_rate * 2)

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=timing.total_ms,  # Total time (non-streaming)
            characters=len(text),
            provider=self.name,
            voice=voice,
            language=language,
        )

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available Qwen3-TTS voices."""
        voices = []
        for voice_id, info in QWEN_VOICES.items():
            if language is None or language in info["language"] or info["language"] == "multilingual":
                voices.append({
                    "id": voice_id,
                    "name": info["name"],
                    "language": info["language"],
                    "gender": info["gender"],
                })
        return voices


class QwenStreamingProvider(TTSProvider):
    """Qwen3-TTS Streaming via DashScope API (realtime, lower latency)."""

    def __init__(self):
        config = ProviderConfig(
            name="Qwen3-TTS Streaming",
            pricing_per_1m_chars=10.00,  # Same pricing
            supported_languages=["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "multilingual"],
            default_voice_en="loongstella",
            default_voice_cn="longxiaochun",
            max_chars_per_request=5000,
            supports_streaming=True,
        )
        super().__init__(config)

    def initialize(self) -> None:
        """Initialize DashScope client."""
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        dashscope.api_key = api_key
        self._is_initialized = True

    def _get_voice(self, voice: Optional[str], language: str) -> str:
        """Get appropriate voice for language."""
        if voice:
            return voice
        if language == "multilingual" or language == "en":
            return "loongstella"
        elif language == "zh":
            return "longxiaochun"
        return self.get_default_voice(language)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech using Qwen3-TTS Realtime (streaming, measure TTFB)."""
        if not self._is_initialized:
            self.initialize()

        voice = self._get_voice(voice, language)
        # Use realtime model for streaming
        model = kwargs.get("model", "qwen3-tts-flash-realtime")

        audio_chunks = []
        sample_rate = 24000

        with TimingContext() as timing:
            synthesizer = SpeechSynthesizer(
                model=model,
                voice=voice,
            )

            # Streaming - measure TTFB
            for chunk in synthesizer.streaming_call(text):
                if timing.first_byte_time is None:
                    timing.mark_first_byte()
                if chunk:
                    audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)
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
        """List available Qwen3-TTS voices."""
        voices = []
        for voice_id, info in QWEN_VOICES.items():
            if language is None or language in info["language"] or info["language"] == "multilingual":
                voices.append({
                    "id": voice_id,
                    "name": info["name"],
                    "language": info["language"],
                    "gender": info["gender"],
                })
        return voices
