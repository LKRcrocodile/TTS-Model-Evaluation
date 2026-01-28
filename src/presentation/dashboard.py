"""HTML Dashboard generator for TTS evaluation results."""

import json
import base64
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from ..evaluation.cost_calculator import CostCalculator


class DashboardGenerator:
    """Generate an interactive HTML dashboard for TTS comparison."""

    # Commercial vs Open-source categorization
    COMMERCIAL_PROVIDERS = ["azure", "azure_streaming", "elevenlabs", "elevenlabs_turbo", "minimax", "minimax_streaming", "minimax_pcm"]
    OPENSOURCE_PROVIDERS = ["qwen3", "qwen3_streaming", "luxtts", "luxtts_streaming"]

    COMMERCIAL_GROUPS = {
        "Azure": ["azure", "azure_streaming"],
        "ElevenLabs": ["elevenlabs", "elevenlabs_turbo"],
        "MiniMax": ["minimax", "minimax_streaming", "minimax_pcm"],
    }

    OPENSOURCE_GROUPS = {
        "Qwen3-TTS": ["qwen3", "qwen3_streaming"],
        "LuxTTS": ["luxtts", "luxtts_streaming"],
    }

    # Languages supported by each provider
    LANGUAGES_SUPPORTED = {
        "Azure TTS": 140,
        "Azure Streaming": 140,
        "ElevenLabs Standard": 29,
        "ElevenLabs Turbo": 29,
        "MiniMax": 11,
        "MiniMax Streaming": 11,
        "MiniMax PCM": 11,
        "Qwen3-TTS": 10,
        "Qwen3-TTS Streaming": 10,
        "LuxTTS": 1,
        "LuxTTS Streaming": 1,
    }

    # Map provider keys to display names
    KEY_TO_DISPLAY = {
        "azure": "Azure TTS",
        "azure_streaming": "Azure Streaming",
        "elevenlabs": "ElevenLabs Standard",
        "elevenlabs_turbo": "ElevenLabs Turbo",
        "qwen3": "Qwen3-TTS",
        "qwen3_streaming": "Qwen3-TTS Streaming",
        "minimax": "MiniMax",
        "minimax_streaming": "MiniMax Streaming",
        "minimax_pcm": "MiniMax PCM",
        "luxtts": "LuxTTS",
        "luxtts_streaming": "LuxTTS Streaming",
    }

    def __init__(self, results_path: Path, output_dir: Path):
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.results = self._load_results()

    def _load_results(self) -> dict:
        """Load benchmark results from JSON."""
        with open(self.results_path) as f:
            return json.load(f)

    def _get_rating(self, value: float, thresholds: tuple, reverse: bool = False) -> str:
        """Get traffic light rating (green/yellow/red)."""
        low, high = thresholds
        if reverse:
            if value <= low:
                return "green"
            elif value <= high:
                return "yellow"
            return "red"
        else:
            if value >= high:
                return "green"
            elif value >= low:
                return "yellow"
            return "red"

    def _generate_comparison_table(self, provider_keys: List[str]) -> str:
        """Generate comparison table for specific providers."""
        providers = self.results.get("providers", {})

        rows = []
        for key in provider_keys:
            if key not in providers:
                continue
            data = providers[key]
            display_name = self.KEY_TO_DISPLAY.get(key, key)
            latency = data.get("total_latency_mean_ms", 0)
            rtf = data.get("avg_realtime_factor", 0)

            latency_rating = self._get_rating(latency, (500, 1500), reverse=True)
            rtf_rating = self._get_rating(rtf, (5, 20))

            lang_count = self.LANGUAGES_SUPPORTED.get(display_name, 0)
            lang_rating = "green" if lang_count >= 20 else "yellow" if lang_count >= 5 else "red"

            rows.append(f"""
            <tr>
                <td class="provider-name">{display_name}</td>
                <td class="rating {latency_rating}">{latency:.0f}ms</td>
                <td class="rating {rtf_rating}">{rtf:.1f}x</td>
                <td class="rating {lang_rating}">{lang_count}</td>
            </tr>
            """)

        return "\n".join(rows) if rows else "<tr><td colspan='4'>No data available</td></tr>"

    def _generate_audio_section(self, provider_groups: Dict[str, List[str]], section_class: str = "") -> str:
        """Generate audio comparison section for specific provider groups."""
        providers = self.results.get("providers", {})

        # Group by sample ID
        samples_by_id: Dict[str, Dict[str, dict]] = {}
        for name, data in providers.items():
            for result in data.get("results", []):
                sample_id = result["sample_id"]
                if sample_id not in samples_by_id:
                    samples_by_id[sample_id] = {}
                samples_by_id[sample_id][name] = {
                    "provider": name,
                    "audio_file": result.get("audio_file", ""),
                    "latency": result["latency_stats"]["mean_ms"],
                    "text": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"],
                }

        def get_audio_src(audio_file: str) -> str:
            if audio_file and Path(audio_file).exists():
                with open(audio_file, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:audio/wav;base64,{audio_data}"
            return ""

        sections = []
        for sample_id, sample_providers in samples_by_id.items():
            text_preview = list(sample_providers.values())[0]["text"] if sample_providers else ""

            group_rows = []
            for group_name, group_provider_keys in provider_groups.items():
                available = [p for p in group_provider_keys if p in sample_providers]
                if not available:
                    continue

                cards = []
                for provider_key in available:
                    s = sample_providers[provider_key]
                    audio_src = get_audio_src(s["audio_file"])
                    display_name = self.KEY_TO_DISPLAY.get(provider_key, provider_key)

                    if "pcm" in provider_key:
                        mode_label = "PCM Streaming"
                        mode_class = "streaming"
                    elif "streaming" in provider_key or "turbo" in provider_key:
                        mode_label = "Streaming"
                        mode_class = "streaming"
                    else:
                        mode_label = "Non-Streaming"
                        mode_class = "non-streaming"

                    cards.append(f"""
                    <div class="audio-card {mode_class}">
                        <div class="mode-badge">{mode_label}</div>
                        <h4>{display_name}</h4>
                        <audio controls>
                            <source src="{audio_src}" type="audio/wav">
                        </audio>
                        <p class="latency">Latency: {s["latency"]:.0f}ms</p>
                    </div>
                    """)

                group_rows.append(f"""
                <div class="provider-group {section_class}">
                    <h4 class="group-title">{group_name}</h4>
                    <div class="audio-pair">
                        {"".join(cards)}
                    </div>
                </div>
                """)

            if group_rows:
                sections.append(f"""
                <div class="sample-section">
                    <h3>{sample_id}</h3>
                    <p class="sample-text">"{text_preview}"</p>
                    <div class="groups-container">
                        {"".join(group_rows)}
                    </div>
                </div>
                """)

        return "\n".join(sections) if sections else "<p>No audio samples available</p>"

    def _generate_cost_table(self, provider_type: str = "all") -> str:
        """Generate cost comparison table."""
        projections = CostCalculator.get_all_projections()

        if provider_type == "commercial":
            filter_names = ["Azure TTS", "Azure Streaming", "ElevenLabs Standard", "ElevenLabs Turbo", "MiniMax", "MiniMax Streaming", "MiniMax PCM"]
        elif provider_type == "opensource":
            filter_names = ["Qwen3-TTS", "Qwen3-TTS Streaming", "LuxTTS"]
        else:
            filter_names = None

        rows = []
        for name, proj in projections.items():
            if filter_names and name not in filter_names:
                continue
            rows.append(f"""
            <tr>
                <td>{name}</td>
                <td>${proj.monthly_100k:.2f}</td>
                <td>${proj.monthly_500k:.2f}</td>
                <td>${proj.monthly_1m:.2f}</td>
                <td class="notes">{proj.notes}</td>
            </tr>
            """)

        return "\n".join(rows) if rows else "<tr><td colspan='5'>No data available</td></tr>"

    def generate(self) -> Path:
        """Generate the complete HTML dashboard."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TTS Model Evaluation - Kira</title>
    <style>
        :root {{
            --bg-color: #f5f5f5;
            --card-bg: #ffffff;
            --text-color: #333;
            --border-color: #ddd;
            --green: #4caf50;
            --yellow: #ff9800;
            --red: #f44336;
            --commercial-color: #1976d2;
            --opensource-color: #388e3c;
            --analysis-color: #7b1fa2;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        header h1 {{ font-size: 2rem; margin-bottom: 8px; }}
        header p {{ color: #666; }}

        /* Navigation */
        .nav {{
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}

        .nav a {{
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s;
        }}

        .nav a:hover {{ transform: translateY(-2px); }}

        .nav-commercial {{ background: #e3f2fd; color: var(--commercial-color); }}
        .nav-opensource {{ background: #e8f5e9; color: var(--opensource-color); }}
        .nav-formats {{ background: #fff3e0; color: #e65100; }}
        .nav-analysis {{ background: #f3e5f5; color: var(--analysis-color); }}

        /* Section styling */
        section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        section h2 {{
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid var(--border-color);
        }}

        .section-commercial h2 {{ border-color: var(--commercial-color); color: var(--commercial-color); }}
        .section-opensource h2 {{ border-color: var(--opensource-color); color: var(--opensource-color); }}
        .section-formats h2 {{ border-color: #e65100; color: #e65100; }}
        .section-analysis h2 {{ border-color: var(--analysis-color); color: var(--analysis-color); }}

        /* Category headers */
        .category-header {{
            text-align: center;
            padding: 16px;
            margin: 40px 0 20px;
            border-radius: 12px;
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .category-commercial {{ background: #e3f2fd; color: var(--commercial-color); }}
        .category-opensource {{ background: #e8f5e9; color: var(--opensource-color); }}
        .category-formats {{ background: #fff3e0; color: #e65100; }}
        .category-analysis {{ background: #f3e5f5; color: var(--analysis-color); }}

        /* Tables */
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: var(--bg-color); font-weight: 600; }}
        .provider-name {{ font-weight: 600; }}

        .rating {{ text-align: center; border-radius: 20px; padding: 4px 12px; font-weight: 500; }}
        .rating.green {{ background: #e8f5e9; color: var(--green); }}
        .rating.yellow {{ background: #fff3e0; color: var(--yellow); }}
        .rating.red {{ background: #ffebee; color: var(--red); }}

        .notes {{ font-size: 0.85rem; color: #666; }}

        /* Audio sections */
        .sample-section {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }}
        .sample-section:last-child {{ border-bottom: none; }}

        .sample-text {{
            font-style: italic;
            color: #666;
            margin: 10px 0 20px;
            padding: 10px;
            background: var(--bg-color);
            border-radius: 8px;
        }}

        .groups-container {{ display: flex; flex-direction: column; gap: 24px; }}

        .provider-group {{
            background: var(--bg-color);
            border-radius: 12px;
            padding: 16px;
        }}

        .group-title {{
            font-size: 1.1rem;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #2196f3;
            color: #2196f3;
        }}

        .provider-group.commercial .group-title {{ border-color: var(--commercial-color); color: var(--commercial-color); }}
        .provider-group.opensource .group-title {{ border-color: var(--opensource-color); color: var(--opensource-color); }}

        .audio-pair {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }}

        @media (max-width: 600px) {{ .audio-pair {{ grid-template-columns: 1fr; }} }}

        .audio-card {{
            background: white;
            padding: 16px;
            border-radius: 8px;
            position: relative;
        }}

        .audio-card h4 {{ margin-bottom: 10px; }}
        .audio-card audio {{ width: 100%; margin-bottom: 8px; }}
        .audio-card .latency {{ font-size: 0.9rem; color: #666; }}

        .mode-badge {{
            position: absolute;
            top: 8px;
            right: 8px;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
            text-transform: uppercase;
        }}

        .audio-card.streaming .mode-badge {{ background: #e3f2fd; color: #1976d2; }}
        .audio-card.non-streaming .mode-badge {{ background: #fce4ec; color: #c2185b; }}
        .audio-card.streaming {{ border-left: 3px solid #1976d2; }}
        .audio-card.non-streaming {{ border-left: 3px solid #c2185b; }}

        /* Analysis cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}

        .metric-card {{
            background: var(--bg-color);
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid var(--analysis-color);
        }}

        .metric-card h4 {{ color: var(--analysis-color); margin-bottom: 8px; }}
        .metric-card p {{ font-size: 0.9rem; color: #555; }}

        .metric-card.orange {{ border-left-color: #ff9800; }}
        .metric-card.orange h4 {{ color: #ff9800; }}

        /* Info boxes */
        .info-box {{
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}

        .info-box h3 {{ margin-bottom: 12px; }}
        .info-box ul {{ margin-left: 20px; }}
        .info-box li {{ margin-bottom: 8px; }}

        .info-yellow {{ background: #fff8e1; }}
        .info-green {{ background: #e8f5e9; }}
        .info-blue {{ background: #e3f2fd; }}
        .info-orange {{ background: #fff3e0; }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>TTS Model Evaluation for Kira</h1>
            <p>Comparing commercial and open-source TTS models across quality, latency, and cost</p>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </header>

        <!-- Navigation -->
        <nav class="nav">
            <a href="#commercial" class="nav-commercial">Commercial Models</a>
            <a href="#opensource" class="nav-opensource">Open-Source Models</a>
            <a href="#formats" class="nav-formats">Audio Formats</a>
            <a href="#analysis" class="nav-analysis">Analysis</a>
        </nav>

        <!-- ==================== COMMERCIAL SECTION ==================== -->
        <div class="category-header category-commercial" id="commercial">
            Commercial TTS Models
        </div>

        <section class="section-commercial">
            <h2>Commercial Models - Performance Overview</h2>
            <p style="margin-bottom: 16px; color: #666;">Azure, ElevenLabs, and MiniMax - paid API services with enterprise support</p>
            <table>
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>Avg Latency</th>
                        <th>Realtime Factor</th>
                        <th>Languages</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_comparison_table(self.COMMERCIAL_PROVIDERS)}
                </tbody>
            </table>
        </section>

        <section class="section-commercial">
            <h2>Commercial Models - Audio Comparison</h2>
            <p style="margin-bottom: 20px;">Listen and compare voice quality across commercial providers:</p>
            {self._generate_audio_section(self.COMMERCIAL_GROUPS, "commercial")}
        </section>

        <section class="section-commercial">
            <h2>Commercial Models - Cost Projection</h2>
            <table>
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>100K chars/mo</th>
                        <th>500K chars/mo</th>
                        <th>1M chars/mo</th>
                        <th>Model</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_cost_table("commercial")}
                </tbody>
            </table>
        </section>

        <!-- ==================== AUDIO FORMATS SECTION ==================== -->
        <div class="category-header category-formats" id="formats">
            Audio Formats & Connection Compatibility
        </div>

        <section class="section-formats">
            <h2>Audio Format Details by Provider</h2>
            <p style="margin-bottom: 16px; color: #666;">Output format comparison for WebSocket/WebRTC integration</p>
            <table>
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>Mode</th>
                        <th>Format</th>
                        <th>Sample Rate</th>
                        <th>Bit Depth</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Azure TTS</td>
                        <td>Non-Streaming</td>
                        <td>RIFF/WAV PCM</td>
                        <td>24 kHz</td>
                        <td>16-bit Mono</td>
                    </tr>
                    <tr>
                        <td>Azure Streaming</td>
                        <td>Streaming</td>
                        <td>RIFF/WAV PCM</td>
                        <td>24 kHz</td>
                        <td>16-bit Mono</td>
                    </tr>
                    <tr>
                        <td>ElevenLabs Standard</td>
                        <td>Non-Streaming</td>
                        <td>Raw PCM</td>
                        <td>24 kHz</td>
                        <td>16-bit Mono</td>
                    </tr>
                    <tr>
                        <td>ElevenLabs Turbo</td>
                        <td>Streaming</td>
                        <td>Raw PCM</td>
                        <td>24 kHz</td>
                        <td>16-bit Mono</td>
                    </tr>
                    <tr>
                        <td>MiniMax</td>
                        <td>Non-Streaming</td>
                        <td>WAV</td>
                        <td>32 kHz</td>
                        <td>16-bit Mono</td>
                    </tr>
                    <tr style="background: #fff3e0;">
                        <td><strong>MiniMax Streaming</strong></td>
                        <td>Streaming (MP3)</td>
                        <td>MP3 (128kbps)</td>
                        <td>32 kHz</td>
                        <td>Compressed</td>
                    </tr>
                    <tr style="background: #e8f5e9;">
                        <td><strong>MiniMax PCM</strong></td>
                        <td>Streaming (PCM)</td>
                        <td>Raw PCM</td>
                        <td>24 kHz</td>
                        <td>16-bit Mono</td>
                    </tr>
                    <tr>
                        <td>Qwen3-TTS</td>
                        <td>Both</td>
                        <td>Raw PCM</td>
                        <td>24 kHz</td>
                        <td>16-bit Mono</td>
                    </tr>
                </tbody>
            </table>
            <div class="info-box info-orange" style="margin-top: 20px;">
                <h3>MiniMax Format Comparison</h3>
                <p><strong>MiniMax Streaming (MP3):</strong> Uses lossy MP3 compression. Requires MP3 decoding before WebRTC transmission, adding latency and complexity.</p>
                <p style="margin-top: 8px;"><strong>MiniMax PCM:</strong> Raw PCM output at 24kHz - directly compatible with WebRTC/WebSocket. No transcoding needed.</p>
            </div>

            <h3 style="margin: 30px 0 12px; color: #e65100;">WebSocket/WebRTC Compatibility</h3>
            <div class="metrics-grid">
                <div class="metric-card" style="border-color: #4caf50;">
                    <h4 style="color: #4caf50;">WebRTC Requirements</h4>
                    <p><strong>Status:</strong> Testing requirement</p>
                    <ul style="margin-top: 8px; margin-left: 16px; font-size: 0.9rem;">
                        <li>Natively supports <strong>Opus</strong> and <strong>PCM</strong> codecs</li>
                        <li>MP3 is <strong>not</strong> a native WebRTC codec</li>
                        <li>MP3 → PCM decoding adds latency and complexity</li>
                        <li>Recommended: Use PCM-outputting providers (Azure, ElevenLabs, MiniMax PCM)</li>
                    </ul>
                </div>
                <div class="metric-card" style="border-color: #1976d2;">
                    <h4 style="color: #1976d2;">WebSocket Requirements</h4>
                    <p><strong>Status:</strong> Production use</p>
                    <ul style="margin-top: 8px; margin-left: 16px; font-size: 0.9rem;">
                        <li>Can transport any binary format (MP3, WAV, PCM)</li>
                        <li>For real-time playback, <strong>raw PCM</strong> is preferred</li>
                        <li>No decoding overhead = lower client-side latency</li>
                        <li>MiniMax Streaming (MP3) works but requires client-side decode</li>
                    </ul>
                </div>
            </div>

            <div class="info-box info-blue" style="margin-top: 20px;">
                <h3>Recommendation for Kira</h3>
                <ul>
                    <li><strong>Lowest Latency (EN):</strong> Azure Streaming (~210ms) or ElevenLabs Turbo (~240ms) - both output PCM, WebRTC compatible</li>
                    <li><strong>Best Practice:</strong> Use PCM-outputting providers (Azure, ElevenLabs, MiniMax PCM) for both WebSocket and WebRTC compatibility</li>
                </ul>
            </div>
        </section>

        <!-- ==================== ANALYSIS SECTION ==================== -->
        <div class="category-header category-analysis" id="analysis">
            Analysis & Recommendations
        </div>

        <section class="section-analysis">
            <h2>Understanding the Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Latency (ms)</h4>
                    <p>Time to generate audio from text. Lower is better. Under 500ms feels instant, 500-1500ms is acceptable, over 1500ms feels slow.</p>
                </div>
                <div class="metric-card">
                    <h4>Realtime Factor</h4>
                    <p>How much faster than playback speed. Example: 15x means a 3-second clip generates in 0.2 seconds. Need at least 1x for real-time apps.</p>
                </div>
                <div class="metric-card">
                    <h4>Cost (per 1M chars)</h4>
                    <p>Price per 1 million characters. 1M chars ≈ 250 pages of text or ~170 minutes of speech.</p>
                </div>
                <div class="metric-card">
                    <h4>Languages Supported</h4>
                    <p>Total number of languages the provider supports. More languages means broader global coverage.</p>
                </div>
            </div>
        </section>

        <section class="section-analysis">
            <h2>Streaming vs Non-Streaming</h2>
            <div class="info-box info-yellow">
                <p><strong>Key Insight:</strong> Streaming is a <em>delivery optimization</em>, not a quality setting. The audio quality is identical - only the timing differs.</p>
                <p style="margin-top: 8px;"><strong>Note:</strong> Audio durations may vary slightly between runs because TTS synthesis is non-deterministic and each API call generates audio independently.</p>
            </div>
            <div class="metrics-grid" style="margin-top: 20px;">
                <div class="metric-card">
                    <h4>Same TTS Model</h4>
                    <p>Both modes use the exact same voice synthesis model. The audio generation algorithm is identical.</p>
                </div>
                <div class="metric-card">
                    <h4>Different Delivery</h4>
                    <p><strong>Non-streaming:</strong> Wait for complete audio, return all at once.<br><strong>Streaming:</strong> Return audio in chunks as it's generated.</p>
                </div>
                <div class="metric-card">
                    <h4>Latency Benefit</h4>
                    <p>Streaming lets audio playback start sooner (e.g., 180ms vs 260ms), reducing perceived wait time.</p>
                </div>
                <div class="metric-card">
                    <h4>Analogy</h4>
                    <p>Like streaming vs downloading a video - the quality is identical, you just start watching sooner with streaming.</p>
                </div>
            </div>

            <h3 style="margin: 30px 0 12px; color: var(--analysis-color);">Why Audio Lengths May Differ</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Non-Deterministic Synthesis</h4>
                    <p>TTS models don't produce identical output every time. Each API call generates audio independently with slight variations in pacing.</p>
                </div>
                <div class="metric-card">
                    <h4>Different Models (ElevenLabs)</h4>
                    <p>ElevenLabs Standard vs Turbo use completely different models with different speaking rates, causing noticeable duration differences.</p>
                </div>
                <div class="metric-card">
                    <h4>Expected Behavior</h4>
                    <p>Small duration differences (±10%) between runs are normal. The content and quality remain consistent.</p>
                </div>
            </div>

        </section>

        <section class="section-analysis">
            <h2>Why Latency Varies by Provider</h2>

            <h3 style="margin: 20px 0 12px; color: #4caf50;">Azure (~260ms non-streaming / ~210ms streaming)</h3>
            <div class="info-box info-green">
                <p><strong>Why only ~20% difference?</strong> Azure is already highly optimized with global infrastructure.</p>
            </div>
            <div class="metrics-grid" style="margin-top: 16px;">
                <div class="metric-card" style="border-color: #4caf50;">
                    <h4 style="color: #4caf50;">Global CDN</h4>
                    <p>140+ regions worldwide. Requests are served from the nearest data center, minimizing network latency.</p>
                </div>
                <div class="metric-card" style="border-color: #4caf50;">
                    <h4 style="color: #4caf50;">Already Fast</h4>
                    <p>Base latency is already low (~260ms). Streaming improves by ~50ms - noticeable but not dramatic.</p>
                </div>
                <div class="metric-card" style="border-color: #4caf50;">
                    <h4 style="color: #4caf50;">Short Text Samples</h4>
                    <p>For short utterances, total generation time is minimal. Streaming benefit increases with longer text.</p>
                </div>
                <div class="metric-card" style="border-color: #4caf50;">
                    <h4 style="color: #4caf50;">Streaming Benefit is Relative</h4>
                    <p>Compare: MiniMax streaming saves 65%, ElevenLabs Turbo saves 79%. Azure saves 20% because it's already optimized.</p>
                </div>
            </div>

            <h3 style="margin: 30px 0 12px; color: #1976d2;">ElevenLabs (Standard ~1100ms / Turbo ~250ms)</h3>
            <div class="info-box info-blue">
                <p><strong>Two Models Available:</strong></p>
                <ul style="margin-top: 8px;">
                    <li><strong>Standard</strong> (<code>eleven_multilingual_v2</code>): ~1100ms - Best quality, higher latency</li>
                    <li><strong>Turbo</strong> (<code>eleven_turbo_v2_5</code>): ~250ms - Near-Azure speed, slightly lower quality</li>
                </ul>
            </div>
            <div class="metrics-grid" style="margin-top: 16px;">
                <div class="metric-card">
                    <h4>Standard Model (~1100ms)</h4>
                    <p>Premium quality with advanced neural models. Best for pre-generated content where latency isn't critical.</p>
                </div>
                <div class="metric-card">
                    <h4>Turbo Model (~250ms)</h4>
                    <p>Optimized for real-time streaming. Quality is still excellent - most users won't notice the difference.</p>
                </div>
                <div class="metric-card">
                    <h4>Regional Endpoints</h4>
                    <p><code>api.elevenlabs.io</code> → US only<br><code>api-global-preview.elevenlabs.io</code> → Auto-routes to closest (US/EU/Singapore)</p>
                </div>
                <div class="metric-card">
                    <h4>Recommendation</h4>
                    <p>Use <strong>Turbo</strong> for real-time applications. Use <strong>Standard</strong> only when quality is paramount and latency is acceptable.</p>
                </div>
            </div>

            <h3 style="margin: 30px 0 12px; color: #ff9800;">MiniMax (~4550ms)</h3>
            <div class="info-box info-orange">
                <p><strong>Key Factor:</strong> MiniMax is a Chinese company. We tested using <code>api.minimax.io</code> (international endpoint), which still routes to China servers.</p>
                <p style="margin-top: 8px;"><strong>Note:</strong> MiniMax also offers <code>api.minimax.chat</code> (China domestic endpoint) which may have lower latency for users in China.</p>
            </div>
            <div class="metrics-grid" style="margin-top: 16px;">
                <div class="metric-card orange">
                    <h4>No Overseas Servers</h4>
                    <p>Both endpoints (<code>api.minimax.io</code> and <code>api.minimax.chat</code>) route to servers in mainland China. MiniMax has no data centers outside China.</p>
                </div>
                <div class="metric-card orange">
                    <h4>International Endpoint ≠ International Servers</h4>
                    <p>The <code>.io</code> domain is just a gateway for overseas access. Requests still travel to China and back, adding 200-400ms+ network latency.</p>
                </div>
                <div class="metric-card orange">
                    <h4>No Global CDN</h4>
                    <p>Unlike Azure (140+ global regions) or ElevenLabs (US/EU servers), MiniMax lacks distributed infrastructure.</p>
                </div>
                <div class="metric-card orange">
                    <h4>Streaming Helps</h4>
                    <p>~4550ms → ~1600ms TTFB (65% faster). Always use streaming mode for MiniMax to reduce perceived wait time.</p>
                </div>
            </div>
            <div class="info-box info-yellow" style="margin-top: 20px;">
                <h3>MiniMax Considerations</h3>
                <ul>
                    <li><strong>High latency (International):</strong> ~1600-1970ms TTFB from overseas - not ideal for real-time outside China</li>
                    <li><strong>Recommended for Chinese:</strong> Native Chinese language support, ideal for Chinese content and users in China</li>
                    <li><strong>China pricing (Recommended):</strong> speech-2.6-turbo costs 2 CNY/10K chars (~200 CNY/1M ≈ $28 USD) - much cheaper than international $60/1M</li>
                </ul>
            </div>
        </section>

        <section class="section-analysis">
            <h2>Recommendations (Commercial TTS)</h2>
            <div class="info-box info-blue">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Best for Real-time (EN):</strong> Azure Streaming (~210ms) or ElevenLabs Turbo (~240ms) - both excellent for low-latency applications</li>
                    <li><strong>Best Quality:</strong> ElevenLabs Standard - most natural sounding, but higher latency (~1100ms) and cost ($165/1M)</li>
                    <li><strong>Best for Chinese Users:</strong> MiniMax - native Chinese support, affordable in China (2 CNY/10K chars ≈ $28/1M USD)</li>
                    <li><strong>Most Languages:</strong> Azure TTS - supports 140 languages with consistent quality</li>
                </ul>
            </div>

            <div class="metrics-grid" style="margin-top: 20px;">
                <div class="metric-card" style="border-color: var(--commercial-color);">
                    <h4 style="color: var(--commercial-color);">Azure</h4>
                    <p>Best overall: lowest latency (~210ms), 140 languages, $16/1M. Ideal for real-time global applications.</p>
                </div>
                <div class="metric-card" style="border-color: var(--commercial-color);">
                    <h4 style="color: var(--commercial-color);">ElevenLabs</h4>
                    <p>Best quality: use Turbo (~240ms) for real-time, Standard (~1100ms) for premium quality. $165/1M.</p>
                </div>
                <div class="metric-card" style="border-color: var(--commercial-color);">
                    <h4 style="color: var(--commercial-color);">MiniMax</h4>
                    <p>Best for Chinese: native Mandarin support. High latency internationally (~1600ms), recommended for China users.</p>
                </div>
            </div>
        </section>

        <!-- ==================== OPEN-SOURCE SECTION ==================== -->
        <div class="category-header category-opensource" id="opensource">
            Open-Source TTS Models
        </div>

        <section class="section-opensource">
            <h2>Open-Source Models - Performance Overview</h2>
            <p style="margin-bottom: 16px; color: #666;">Qwen3-TTS and LuxTTS - free/low-cost alternatives with self-hosting options</p>
            <table>
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>Avg Latency</th>
                        <th>Realtime Factor</th>
                        <th>Languages</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_comparison_table(self.OPENSOURCE_PROVIDERS)}
                </tbody>
            </table>
        </section>

        <section class="section-opensource">
            <h2>Open-Source Models - Audio Comparison</h2>
            <p style="margin-bottom: 20px;">Listen and compare voice quality across open-source providers:</p>
            {self._generate_audio_section(self.OPENSOURCE_GROUPS, "opensource")}
        </section>

        <section class="section-opensource">
            <h2>Open-Source Models - Cost Projection</h2>
            <table>
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>100K chars/mo</th>
                        <th>500K chars/mo</th>
                        <th>1M chars/mo</th>
                        <th>Model</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_cost_table("opensource")}
                </tbody>
            </table>
            <div class="info-box info-green" style="margin-top: 20px;">
                <h3>Open-Source Advantage</h3>
                <ul>
                    <li><strong>Qwen3-TTS:</strong> Uses DashScope API (~$10/1M chars) or can be self-hosted for free</li>
                    <li><strong>LuxTTS:</strong> Runs locally on CPU - zero API cost, only compute resources</li>
                    <li><strong>No vendor lock-in:</strong> Full control over models and data</li>
                </ul>
            </div>
        </section>

        <footer>
            <p>TTS Evaluation Framework | Kira Voice Quality R&D</p>
        </footer>
    </div>
</body>
</html>
"""

        # Write dashboard
        dashboard_dir = self.output_dir / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        output_path = dashboard_dir / "index.html"

        with open(output_path, "w") as f:
            f.write(html)

        print(f"Dashboard generated: {output_path}")
        return output_path
