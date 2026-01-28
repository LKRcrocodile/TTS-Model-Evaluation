# TTS Model Evaluation Plan for Kira

## Objective
Evaluate TTS models to find better alternatives than Microsoft/AI Foundry, comparing quality, latency, and cost across commercial and open-source options.

## Models to Evaluate

**Language Priority: English + Chinese**

### Commercial (3 models)
| Model | Pricing | EN/CN Support | Notes |
|-------|---------|---------------|-------|
| Azure TTS | $16/1M chars | Yes/Yes | Current baseline |
| ElevenLabs | ~$165/1M chars | Yes/Yes | Highest quality, premium cost |
| MiniMax | $60/1M (intl), ~$28 (China) | Yes/Yes | speech-2.6-turbo, Chinese-native |

### Open-Source (2 models)
| Model | Deployment | EN/CN Support | Notes |
|-------|------------|---------------|-------|
| Qwen3-TTS | DashScope API | Yes/Yes | Use cloud API (no local GPU) |
| LuxTTS | Local CPU | Yes/Limited | 150x realtime, runs on CPU |

*PersonaPlex excluded - it's speech-to-speech, not pure TTS*

## Project Structure

```
TTS speech model/
├── src/
│   ├── providers/          # TTS integrations (base.py, azure.py, elevenlabs.py, etc.)
│   ├── evaluation/         # Benchmarking (latency.py, cost_calculator.py, benchmark_runner.py)
│   └── presentation/       # Dashboard generator
├── config/
│   ├── test_texts.yaml     # Sample texts for demos
│   └── models.yaml         # Model configurations
├── outputs/
│   ├── audio/              # Generated samples by model
│   ├── metrics/            # JSON/CSV results
│   └── dashboard/          # HTML presentation
├── scripts/
│   ├── run_evaluation.py   # Main runner
│   └── download_models.py  # Open-source model setup
└── notebooks/
    └── stakeholder_demo.ipynb  # Interactive demo
```

## Implementation Steps

### Phase 1: Setup
- [ ] Create project structure and install dependencies
- [ ] Configure API credentials (Azure, ElevenLabs, MiniMax, DashScope for Qwen3)
- [ ] Install LuxTTS for local CPU inference

### Phase 2: Provider Integration (5 models)
- [ ] Implement base provider interface (`src/providers/base.py`)
- [ ] Integrate Azure TTS (REST API)
- [ ] Integrate ElevenLabs (Python SDK)
- [ ] Integrate MiniMax (REST API)
- [ ] Integrate Qwen3-TTS via DashScope API (cloud)
- [ ] Integrate LuxTTS (local CPU)

### Phase 3: Test Content (5 Samples)

| # | Type | Language | Text |
|---|------|----------|------|
| 1 | Basic | EN | "Hello, how can I help you today?" |
| 2 | Complex | EN | "Dr. Smith's API at api.example.com returns JSON for the Q4 NYSE report." |
| 3 | Numbers | EN | "Your balance is $12,847.53, payment of $299.99 due January 15th, 2026." |
| 4 | Chinese+EN | ZH | "欢迎使用Kira智能助手，我是您的AI客服，请问有什么可以帮您？" |
| 5 | English+CN | EN | "Welcome to Kira智能助手, your AI客服 for 24/7 support." |

### Phase 4: Benchmarking
- [ ] Implement latency measurement (TTFB, total generation time)
- [ ] Calculate cost projections at different usage levels
- [ ] Generate 10-20 second audio samples for each model
- [ ] Run 10+ iterations per model for statistical reliability

### Phase 5: Presentation
- [ ] Build HTML dashboard with:
  - Side-by-side audio comparison player
  - Traffic-light comparison matrix (Quality/Latency/Cost)
  - Cost projection charts
  - Recommendations section
- [ ] Normalize all audio to -16 LUFS for fair comparison

## Metrics to Capture

| Metric | Description |
|--------|-------------|
| **TTFB** | Time to first audio byte (ms) |
| **Total latency** | Full generation time (ms) |
| **Realtime factor** | Audio duration / generation time |
| **Cost/1K chars** | Normalized cost comparison |
| **Monthly estimate** | Based on usage assumptions |

## Deliverables

1. **Audio samples**: 10-20 sec demos per model (WAV + MP3)
2. **Metrics report**: JSON/CSV with latency and cost data
3. **HTML dashboard**: Self-contained, browser-viewable comparison
4. **Recommendations**: Summary for stakeholders

## Key Dependencies

```
# Core
torch, torchaudio, transformers, accelerate

# Commercial APIs
azure-cognitiveservices-speech, elevenlabs, requests

# Open-source models
qwen-tts (pip install)
LuxTTS (from GitHub)

# Audio & visualization
soundfile, pyloudnorm, plotly, jinja2
```

## Verification Plan

1. Run `scripts/run_evaluation.py` to execute full benchmark
2. Check `outputs/audio/` for generated samples (verify playback)
3. Open `outputs/dashboard/index.html` in browser
4. Verify all 5 models have comparable samples and metrics
5. Review cost projections against documented pricing

## Decisions Made

- **PersonaPlex**: Excluded (speech-to-speech, not pure TTS)
- **Language priority**: English + Chinese
- **GPU resources**: None locally - using cloud APIs (DashScope) and CPU-only LuxTTS

## Cost Projection (Multiple Tiers)

Dashboard will show costs at 100K, 500K, and 1M+ chars/month:

| Model | 100K/mo | 500K/mo | 1M/mo | Notes |
|-------|---------|---------|-------|-------|
| Azure TTS | ~$1.60 | ~$8 | ~$16 | |
| ElevenLabs | ~$16.50 | ~$82.50 | ~$165 | |
| MiniMax | ~$6 | ~$30 | ~$60 | intl pricing; ~$28/1M in China |
| Qwen3 (DashScope) | ~$1 | ~$5 | ~$10 | |
| LuxTTS (CPU) | $0 | $0 | $0 | compute cost only |
