"""Azure Cognitive Services TTS provider with streaming support."""

import os
import time
from typing import Optional
import azure.cognitiveservices.speech as speechsdk

from .base import TTSProvider, TTSResult, ProviderConfig


class AzureStreamingProvider(TTSProvider):
    """Azure TTS with streaming - measures Time to First Byte (TTFB)."""

    def __init__(self):
        config = ProviderConfig(
            name="Azure Streaming",
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
        """Generate speech using Azure TTS with streaming to capture TTFB."""
        if not self._is_initialized:
            self.initialize()

        # Use multilingual voice for mixed language content
        if language == "multilingual":
            voice = self._multilingual_voice
        else:
            voice = voice or self.get_default_voice(language)
        self._speech_config.speech_synthesis_voice_name = voice

        # Track timing
        start_time = time.perf_counter()
        first_byte_time = None
        audio_chunks = []

        # Create synthesizer
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=None  # In-memory
        )

        # Set up streaming callback to capture TTFB
        def on_synthesizing(evt):
            nonlocal first_byte_time
            if first_byte_time is None and evt.result.audio_data:
                first_byte_time = time.perf_counter()
            if evt.result.audio_data:
                audio_chunks.append(evt.result.audio_data)

        # Connect the callback
        synthesizer.synthesizing.connect(on_synthesizing)

        # Start synthesis (non-blocking) and wait for completion
        result = synthesizer.speak_text_async(text).get()
        end_time = time.perf_counter()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            sample_rate = 24000
            duration = len(audio_data) / (sample_rate * 2)  # 2 bytes per sample

            # Calculate TTFB and total time
            total_ms = (end_time - start_time) * 1000
            ttfb_ms = (first_byte_time - start_time) * 1000 if first_byte_time else total_ms

            return TTSResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration_seconds=duration,
                latency_ms=ttfb_ms,  # Report TTFB as primary latency for streaming
                ttfb_ms=ttfb_ms,
                characters=len(text),
                provider=self.name,
                voice=voice,
                language=language,
            )
        else:
            cancellation = result.cancellation_details
            raise RuntimeError(
                f"Azure TTS Streaming failed: {cancellation.reason} - {cancellation.error_details}"
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
