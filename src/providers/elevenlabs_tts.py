"""ElevenLabs TTS provider."""

import os
import io
import time
from typing import Optional, Iterator
from elevenlabs import ElevenLabs

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs Text-to-Speech provider - Standard model (higher quality, higher latency)."""

    def __init__(self):
        config = ProviderConfig(
            name="ElevenLabs Standard",
            pricing_per_1m_chars=165.00,  # Scale tier
            supported_languages=["en", "zh", "es", "fr", "de", "ja", "ko", "pt", "it", "multilingual"],
            default_voice_en="Rachel",
            default_voice_cn="Rachel",  # Multilingual
            max_chars_per_request=5000,
            supports_streaming=True,
        )
        super().__init__(config)
        self._client: Optional[ElevenLabs] = None
        self._voices_cache: dict = {}

    # Default ElevenLabs voice IDs (premade voices)
    DEFAULT_VOICES = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "drew": "29vD33N1CtxCmqQRPOHJ",
        "clyde": "2EiwWnXFnvU5JabPnv8n",
        "paul": "5Q0t7uMcjvnagumLfvZi",
        "domi": "AZnzlk1XvdvUeBnXmlld",
        "dave": "CYw3kZ02Hs0563khs1Fj",
        "fin": "D38z5RcWu1voky8WS1ja",
        "sarah": "EXAVITQu4vr4xnSDxMaL",
        "antoni": "ErXwobaYiN019PkySvjV",
        "thomas": "GBv7mTt0atIp3Br8iCZE",
    }

    def initialize(self) -> None:
        """Initialize ElevenLabs client."""
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

        # Use global endpoint for auto-routing to closest server (US/Netherlands/Singapore)
        # This reduces latency for users outside the US
        base_url = os.environ.get("ELEVENLABS_BASE_URL", "https://api-global-preview.elevenlabs.io")
        self._client = ElevenLabs(api_key=api_key, base_url=base_url)

        # Use default voice IDs (don't require voices_read permission)
        self._voices_cache = self.DEFAULT_VOICES.copy()
        self.config.default_voice_en = self.DEFAULT_VOICES["sarah"]
        self.config.default_voice_cn = self.DEFAULT_VOICES["sarah"]
        self._is_initialized = True

    def _get_voice_id(self, voice_name: str) -> str:
        """Convert voice name to voice ID."""
        return self._voices_cache.get(voice_name.lower(), voice_name)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech using ElevenLabs."""
        if not self._is_initialized:
            self.initialize()

        voice = voice or self.get_default_voice(language)
        voice_id = self._get_voice_id(voice)
        model = kwargs.get("model", "eleven_multilingual_v2")  # Standard model (higher quality)

        with TimingContext() as timing:
            audio_generator = self._client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model,
                output_format="pcm_24000",  # 24kHz PCM
            )
            # Collect audio chunks with streaming timing
            audio_chunks = []
            for chunk in audio_generator:
                if timing.first_byte_time is None:
                    timing.mark_first_byte()
                audio_chunks.append(chunk)

        audio_data = b"".join(audio_chunks)
        sample_rate = 24000
        duration = len(audio_data) / (sample_rate * 2)  # 16-bit mono

        # For standard model, report total latency (wait for complete audio)
        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=timing.total_ms,  # Total time to receive all audio
            ttfb_ms=timing.ttfb_ms,
            characters=len(text),
            provider=self.name,
            voice=voice,
            language=language,
        )

    def generate_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> Iterator[bytes]:
        """Stream audio generation."""
        if not self._is_initialized:
            self.initialize()

        voice = voice or self.get_default_voice(language)
        voice_id = self._get_voice_id(voice)
        model = kwargs.get("model", "eleven_multilingual_v2")  # Standard model (higher quality)

        audio_generator = self._client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model,
            output_format="pcm_24000",
        )

        for chunk in audio_generator:
            yield chunk

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available ElevenLabs voices."""
        if not self._is_initialized:
            self.initialize()

        voices_response = self._client.voices.get_all()
        voices = []

        for voice in voices_response.voices:
            voices.append({
                "id": voice.voice_id,
                "name": voice.name,
                "language": "multilingual",
                "category": voice.category,
            })

        return voices
