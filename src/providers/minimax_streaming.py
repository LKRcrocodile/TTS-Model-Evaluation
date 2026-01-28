"""MiniMax TTS Streaming provider."""

import os
import io
import base64
import json
from typing import Optional
import requests

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


class MiniMaxStreamingProvider(TTSProvider):
    """MiniMax Text-to-Speech provider with streaming support."""

    # Use the T2A v2 endpoint
    API_URL = "https://api.minimax.io/v1/t2a_v2"

    # Preset voices for different languages (v2 API voice IDs)
    VOICES = {
        "en": {
            "female": "Calm_Woman",
            "male": "presenter_male",
        },
        "zh": {
            "female": "female-shaonv",
            "male": "male-qn-qingse",
        },
    }

    def __init__(self):
        config = ProviderConfig(
            name="MiniMax Streaming",
            pricing_per_1m_chars=60.00,  # speech-2.6-turbo (official pricing)
            supported_languages=["en", "zh", "ja", "ko", "es", "fr", "de", "pt", "it", "ru", "ar", "multilingual"],
            default_voice_en="Calm_Woman",
            default_voice_cn="female-shaonv",
            max_chars_per_request=200000,
            supports_streaming=True,
        )
        super().__init__(config)
        self._api_key: Optional[str] = None
        self._group_id: Optional[str] = None

    def initialize(self) -> None:
        """Initialize MiniMax client."""
        self._api_key = os.environ.get("MINIMAX_API_KEY")
        self._group_id = os.environ.get("MINIMAX_GROUP_ID")

        if not self._api_key:
            raise ValueError("MINIMAX_API_KEY environment variable not set")
        if not self._group_id:
            raise ValueError("MINIMAX_GROUP_ID environment variable not set")

        self._is_initialized = True

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech using MiniMax with streaming (measure TTFB)."""
        if not self._is_initialized:
            self.initialize()

        # Use Chinese voice for multilingual (handles EN+CN mixed text well)
        if language == "multilingual":
            voice = voice or "female-shaonv"
        else:
            voice = voice or self.get_default_voice(language)
        model = kwargs.get("model", "speech-2.6-turbo")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate",  # Avoid brotli encoding issues
        }

        payload = {
            "model": model,
            "text": text,
            "stream": True,  # Enable streaming
            "voice_setting": {
                "voice_id": voice,
                "speed": kwargs.get("speed", 1.0),
                "vol": kwargs.get("volume", 1.0),
                "pitch": kwargs.get("pitch", 0),
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",  # Streaming only supports MP3
            },
        }

        audio_chunks = []
        sample_rate = 32000

        with TimingContext() as timing:
            response = requests.post(
                f"{self.API_URL}?GroupId={self._group_id}",
                headers=headers,
                json=payload,
                timeout=120,
                stream=True,  # Enable response streaming
            )
            response.raise_for_status()

            # Process streaming response (SSE format)
            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8')

                # Skip SSE event lines
                if line_str.startswith('event:'):
                    continue

                # Parse data lines
                if line_str.startswith('data:'):
                    data_str = line_str[5:].strip()
                    if not data_str or data_str == '[DONE]':
                        continue

                    try:
                        data = json.loads(data_str)

                        # Collect audio chunks (audio is HEX encoded, not base64)
                        # IMPORTANT: Skip status=2 chunks - they contain the FULL audio again,
                        # which would duplicate the incremental chunks we already collected
                        if 'data' in data and 'audio' in data['data']:
                            status = data['data'].get('status', 1)
                            if status == 2:
                                # Final chunk contains complete audio - skip to avoid duplication
                                continue
                            audio_hex = data['data']['audio']
                            if audio_hex:  # Skip empty audio chunks
                                # Mark first byte time
                                if timing.first_byte_time is None:
                                    timing.mark_first_byte()
                                # Decode hex to bytes
                                audio_chunk = bytes.fromhex(audio_hex)
                                audio_chunks.append(audio_chunk)
                    except (json.JSONDecodeError, ValueError):
                        continue

        audio_data = b"".join(audio_chunks)

        # Calculate duration (MP3, approximate)
        # For MP3 at 128kbps: duration = bytes * 8 / 128000
        duration = len(audio_data) * 8 / 128000 if audio_data else 0

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=timing.ttfb_ms or timing.total_ms,  # Report TTFB for streaming
            ttfb_ms=timing.ttfb_ms,
            characters=len(text),
            provider=self.name,
            voice=voice,
            language=language,
        )

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available MiniMax voices."""
        voices = []

        for lang, lang_voices in self.VOICES.items():
            if language is None or lang.startswith(language):
                for gender, voice_id in lang_voices.items():
                    voices.append({
                        "id": voice_id,
                        "name": f"{lang.upper()} {gender.title()}",
                        "language": lang,
                        "gender": gender,
                    })

        return voices
