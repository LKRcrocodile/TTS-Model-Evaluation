from .base import TTSProvider, TTSResult
from .azure_tts import AzureTTSProvider
from .azure_streaming import AzureStreamingProvider
from .elevenlabs_tts import ElevenLabsTTSProvider
from .elevenlabs_turbo import ElevenLabsTurboProvider
from .minimax_tts import MiniMaxTTSProvider
from .minimax_streaming import MiniMaxStreamingProvider
from .minimax_pcm_streaming import MiniMaxPCMStreamingProvider
from .qwen_tts import QwenTTSProvider, QwenStreamingProvider
from .qwen_selfhosted import QwenSelfHostedProvider
from .luxtts import LuxTTSProvider
from .luxtts_streaming import LuxTTSStreamingProvider

__all__ = [
    "TTSProvider",
    "TTSResult",
    "AzureTTSProvider",
    "AzureStreamingProvider",
    "ElevenLabsTTSProvider",
    "ElevenLabsTurboProvider",
    "MiniMaxTTSProvider",
    "MiniMaxStreamingProvider",
    "MiniMaxPCMStreamingProvider",
    "QwenTTSProvider",
    "QwenStreamingProvider",
    "QwenSelfHostedProvider",
    "LuxTTSProvider",
    "LuxTTSStreamingProvider",
]
