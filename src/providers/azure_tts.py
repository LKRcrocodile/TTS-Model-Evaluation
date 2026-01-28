"""Azure Cognitive Services TTS provider."""

import os
import io
from typing import Optional
import azure.cognitiveservices.speech as speechsdk

from .base import TTSProvider, TTSResult, ProviderConfig, TimingContext


class AzureTTSProvider(TTSProvider):
    """Azure Cognitive Services Text-to-Speech provider."""

    def __init__(self):
        config = ProviderConfig(
            name="Azure TTS",
            pricing_per_1m_chars=16.00,
            supported_languages=["en", "zh", "es", "fr", "de", "ja", "ko", "multilingual"],
            default_voice_en="en-US-JennyNeural",
            default_voice_cn="zh-CN-XiaoxiaoNeural",
            max_chars_per_request=10000,
            supports_streaming=True,
        )
        super().__init__(config)
        self._speech_config: Optional[speechsdk.SpeechConfig] = None
        self._multilingual_voice = "en-US-JennyMultilingualNeural"

    def initialize(self) -> None:
        """Initialize Azure Speech SDK with credentials."""
        speech_key = os.environ.get("AZURE_SPEECH_KEY")
        speech_region = os.environ.get("AZURE_SPEECH_REGION", "eastus")

        if not speech_key:
            raise ValueError("AZURE_SPEECH_KEY environment variable not set")

        self._speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region
        )
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        self._is_initialized = True

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """Generate speech using Azure TTS."""
        if not self._is_initialized:
            self.initialize()

        # Use multilingual voice for mixed language content
        if language == "multilingual":
            voice = self._multilingual_voice
        else:
            voice = voice or self.get_default_voice(language)
        self._speech_config.speech_synthesis_voice_name = voice

        # Use in-memory audio output
        audio_config = None  # Will use default (in-memory)

        with TimingContext() as timing:
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config,
                audio_config=audio_config
            )
            result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            # Calculate duration from audio data (24kHz, 16-bit mono)
            sample_rate = 24000
            duration = len(audio_data) / (sample_rate * 2)  # 2 bytes per sample

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
        else:
            cancellation = result.cancellation_details
            raise RuntimeError(
                f"Azure TTS failed: {cancellation.reason} - {cancellation.error_details}"
            )

    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available Azure voices."""
        if not self._is_initialized:
            self.initialize()

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=None
        )
        result = synthesizer.get_voices_async().get()

        voices = []
        for voice in result.voices:
            if language is None or voice.locale.startswith(language):
                voices.append({
                    "id": voice.short_name,
                    "name": voice.local_name,
                    "language": voice.locale,
                    "gender": voice.gender.name,
                })
        return voices
