# TTS Model Evaluation for Kira

Evaluate Text-to-Speech models to find better alternatives for Kira voice quality.

## Models Evaluated

| Provider | Type | Languages | Notes |
|----------|------|-----------|-------|
| Azure TTS | Commercial | EN, CN + 140 more | Current baseline |
| ElevenLabs | Commercial | EN, CN + 27 more | Highest quality |
| MiniMax | Commercial | EN, CN + 9 more | Best value, Chinese-native |
| Qwen3-TTS | Open-source (API) | EN, CN + 8 more | Via DashScope |
| LuxTTS | Open-source (Local) | EN only | Runs on CPU |

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your API keys:
# - AZURE_SPEECH_KEY
# - AZURE_SPEECH_REGION
# - ELEVENLABS_API_KEY
# - MINIMAX_API_KEY
# - MINIMAX_GROUP_ID
# - DASHSCOPE_API_KEY
```

### 3. Run Evaluation

```bash
# Full evaluation (all providers, EN + CN)
python scripts/run_evaluation.py

# Specific providers only
python scripts/run_evaluation.py --providers azure minimax

# English only
python scripts/run_evaluation.py --languages en

# Generate dashboard from existing results
python scripts/run_evaluation.py --skip-benchmark
```

### 4. View Results

Open the dashboard in a browser:
```bash
open outputs/dashboard/index.html
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/stakeholder_demo.ipynb
```

## Project Structure

```
TTS speech model/
├── src/
│   ├── providers/      # TTS provider integrations
│   ├── evaluation/     # Benchmarking framework
│   └── presentation/   # Dashboard generator
├── config/
│   ├── test_texts.yaml # Sample texts for testing
│   └── models.yaml     # Provider configurations
├── outputs/
│   ├── audio/          # Generated samples
│   ├── metrics/        # JSON results
│   └── dashboard/      # HTML dashboard
├── scripts/
│   └── run_evaluation.py
└── notebooks/
    └── stakeholder_demo.ipynb
```

## Cost Comparison

| Provider | 100K/mo | 500K/mo | 1M/mo |
|----------|---------|---------|-------|
| Azure TTS | $1.60 | $8.00 | $16 |
| ElevenLabs | $16.50 | $82.50 | $165 |
| MiniMax | $5.00 | $25.00 | $50 |
| Qwen3-TTS | $1.00 | $5.00 | $10 |
| LuxTTS | $0 | $0 | $0 |
