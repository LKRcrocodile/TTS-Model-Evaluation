"""Base class for TTS providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Iterator
import time
from pathlib import Path


@dataclass
class TTSResult:
    """Result from TTS generation."""
    audio_data: bytes
    sample_rate: int
    duration_seconds: float
    latency_ms: float  # Total generation time
    ttfb_ms: Optional[float] = None  # Time to first byte (for streaming)
    characters: int = 0
    provider: str = ""
    voice: str = ""
    language: str = ""

    @property
    def realtime_factor(self) -> float:
        """Ratio of audio duration to generation time. >1 means faster than realtime."""
        if self.latency_ms <= 0:
            return 0
        return self.duration_seconds / (self.latency_ms / 1000)

    @property
    def chars_per_second(self) -> float:
        """Characters processed per second."""
        if self.latency_ms <= 0:
            return 0
        return self.characters / (self.latency_ms / 1000)


@dataclass
class ProviderConfig:
    """Configuration for a TTS provider."""
    name: str
    pricing_per_1m_chars: float  # USD per 1 million characters
    supported_languages: list[str] = field(default_factory=list)
    default_voice_en: str = ""
    default_voice_cn: str = ""
    max_chars_per_request: int = 5000
    supports_streaming: bool = False


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._is_initialized = False

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider (load credentials, connect to API, etc.)."""
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> TTSResult:
        """
        Generate speech from text.

        Args:
            text: The text to synthesize
            voice: Voice ID/name (uses default if None)
            language: Language code (en, zh, etc.)
            **kwargs: Provider-specific options

        Returns:
            TTSResult with audio data and metrics
        """
        pass

    def generate_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> Iterator[bytes]:
        """
        Generate speech with streaming (if supported).

        Default implementation falls back to non-streaming generate().
        """
        result = self.generate(text, voice, language, **kwargs)
        yield result.audio_data

    @abstractmethod
    def list_voices(self, language: Optional[str] = None) -> list[dict]:
        """List available voices, optionally filtered by language."""
        pass

    def get_default_voice(self, language: str = "en") -> str:
        """Get the default voice for a language."""
        if language.startswith("zh"):
            return self.config.default_voice_cn
        return self.config.default_voice_en

    def calculate_cost(self, characters: int) -> float:
        """Calculate cost in USD for given character count."""
        return (characters / 1_000_000) * self.config.pricing_per_1m_chars

    def save_audio(
        self,
        result: TTSResult,
        output_path: Path,
        format: str = "wav"
    ) -> Path:
        """Save audio result to file."""
        import soundfile as sf
        import io
        import numpy as np

        # Convert bytes to numpy array
        audio_io = io.BytesIO(result.audio_data)

        # Handle different input formats
        try:
            data, sr = sf.read(audio_io)
        except Exception:
            # Try reading as raw PCM
            audio_io.seek(0)
            data = np.frombuffer(result.audio_data, dtype=np.int16)
            data = data.astype(np.float32) / 32768.0
            sr = result.sample_rate

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_path), data, sr, format=format)
        return output_path


class TimingContext:
    """Context manager for measuring execution time."""

    def __init__(self):
        self.start_time: float = 0
        self.first_byte_time: Optional[float] = None
        self.end_time: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    def mark_first_byte(self):
        """Mark when first byte of audio was received."""
        if self.first_byte_time is None:
            self.first_byte_time = time.perf_counter()

    @property
    def total_ms(self) -> float:
        """Total elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def ttfb_ms(self) -> Optional[float]:
        """Time to first byte in milliseconds."""
        if self.first_byte_time is None:
            return None
        return (self.first_byte_time - self.start_time) * 1000
