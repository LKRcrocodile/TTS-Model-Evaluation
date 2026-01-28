"""Qwen3-TTS provider for self-hosted API on Azure GPU VM."""

import os
from typing import Optional
import requests

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


class QwenSelfHostedProvider(TTSProvider):
    """Qwen3-TTS via self-hosted API on Azure GPU VM."""

    def __init__(self):
        config = ProviderConfig(
            name="Qwen3-TTS (Self-Hosted)",
            pricing_per_1m_chars=0.00,  # Self-hosted - compute cost only
            supported_languages=["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it", "multilingual"],
            default_voice_en="default",
            default_voice_cn="default",
            max_chars_per_request=5000,
            supports_streaming=False,
        )
        super().__init__(config)
        self._api_url: Optional[str] = None

    def initialize(self) -> None:
        """Initialize connection to self-hosted API."""
        self._api_url = os.environ.get("QWEN_TTS_API_URL")
        if not self._api_url:
            raise ValueError(
                "QWEN_TTS_API_URL environment variable not set. "
                "Set it to your Azure VM endpoint, e.g., http://<VM_IP>:8000"
            )

        # Test connection
        try:
            response = requests.get(f"{self._api_url}/health", timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Qwen3-TTS API at {self._api_url}: {e}")

        self._is_initialized = True

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech using self-hosted Qwen3-TTS API."""
        if not self._is_initialized:
            self.initialize()

        voice = voice or "default"
        sample_rate = 24000

        with TimingContext() as timing:
            response = requests.post(
                f"{self._api_url}/synthesize",
                json={
                    "text": text,
                    "voice": voice,
                    "language": language,
                },
                timeout=120,
            )
            response.raise_for_status()
            audio_data = response.content

            # Try to get metrics from headers
            server_latency = response.headers.get("X-Latency-Ms")
            server_duration = response.headers.get("X-Duration-Seconds")

        # Calculate duration from audio data (16-bit PCM)
        duration = float(server_duration) if server_duration else len(audio_data) / (sample_rate * 2)

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            latency_ms=timing.total_ms,
            characters=len(text),
            provider=self.name,
            voice=voice,
            language=language,
        )

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available voices from self-hosted API."""
        if not self._is_initialized:
            try:
                self.initialize()
            except Exception:
                return [{"id": "default", "name": "Default", "language": "multilingual"}]

        try:
            response = requests.get(f"{self._api_url}/voices", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("voices", [])
        except Exception:
            return [{"id": "default", "name": "Default", "language": "multilingual"}]
