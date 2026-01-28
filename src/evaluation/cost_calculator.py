"""Cost calculation utilities for TTS providers."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class CostProjection:
    """Cost projection at different usage levels."""
    provider: str
    per_1k_chars: float
    monthly_100k: float
    monthly_500k: float
    monthly_1m: float
    notes: str = ""


class CostCalculator:
    """Calculate and compare costs across TTS providers."""

    # Pricing per 1 million characters (USD)
    PRICING = {
        "Azure TTS": 16.00,
        "Azure Streaming": 16.00,  # Same price, streaming mode
        "ElevenLabs Standard": 165.00,  # Scale tier
        "ElevenLabs Turbo": 165.00,  # Same price, faster
        "MiniMax": 60.00,  # speech-2.6-turbo (official pricing)
        "MiniMax Streaming": 60.00,  # Same price, streaming mode
        "MiniMax PCM": 60.00,  # Same price, PCM output for WebRTC
        "Qwen3-TTS": 10.00,  # DashScope pricing
        "Qwen3-TTS Streaming": 10.00,  # Same price, realtime model
        "LuxTTS": 0.00,  # Local only
    }

    NOTES = {
        "Azure TTS": "en-US-AvaMultilingualNeural, non-streaming",
        "Azure Streaming": "en-US-AvaMultilingualNeural, streaming",
        "ElevenLabs Standard": "eleven_multilingual_v2, high quality",
        "ElevenLabs Turbo": "eleven_turbo_v2_5, low latency",
        "MiniMax": "speech-2.6-turbo, non-streaming",
        "MiniMax Streaming": "speech-2.6-turbo, streaming MP3",
        "MiniMax PCM": "speech-2.6-turbo, streaming PCM",
        "Qwen3-TTS": "qwen3-tts-flash, non-streaming",
        "Qwen3-TTS Streaming": "qwen3-tts-flash-realtime, streaming",
        "LuxTTS": "Local CPU - no API cost",
    }

    @classmethod
    def calculate_cost(cls, provider: str, characters: int) -> float:
        """Calculate cost in USD for given character count."""
        price_per_1m = cls.PRICING.get(provider, 0)
        return (characters / 1_000_000) * price_per_1m

    @classmethod
    def get_projection(cls, provider: str) -> CostProjection:
        """Get cost projection for a provider at standard usage levels."""
        price_per_1m = cls.PRICING.get(provider, 0)

        return CostProjection(
            provider=provider,
            per_1k_chars=price_per_1m / 1000,
            monthly_100k=(100_000 / 1_000_000) * price_per_1m,
            monthly_500k=(500_000 / 1_000_000) * price_per_1m,
            monthly_1m=price_per_1m,
            notes=cls.NOTES.get(provider, ""),
        )

    @classmethod
    def get_all_projections(cls) -> Dict[str, CostProjection]:
        """Get cost projections for all providers."""
        return {
            provider: cls.get_projection(provider)
            for provider in cls.PRICING.keys()
        }

    @classmethod
    def format_comparison_table(cls) -> str:
        """Format a comparison table for display."""
        projections = cls.get_all_projections()

        lines = [
            "| Provider | Per 1K chars | 100K/mo | 500K/mo | 1M/mo | Notes |",
            "|----------|--------------|---------|---------|-------|-------|",
        ]

        for name, proj in projections.items():
            lines.append(
                f"| {name} | ${proj.per_1k_chars:.4f} | "
                f"${proj.monthly_100k:.2f} | ${proj.monthly_500k:.2f} | "
                f"${proj.monthly_1m:.2f} | {proj.notes} |"
            )

        return "\n".join(lines)
