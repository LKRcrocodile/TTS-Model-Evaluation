"""Benchmark runner for TTS evaluation."""

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from ..providers.base import TTSProvider, TTSResult
from .cost_calculator import CostCalculator


@dataclass
class LatencyStats:
    """Latency statistics from multiple runs."""
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    std_dev_ms: float
    ttfb_mean_ms: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single text sample."""
    provider: str
    sample_id: str
    text: str
    language: str
    category: str

    # Latency metrics
    latency_stats: LatencyStats
    iterations: int

    # Audio info
    duration_seconds: float
    sample_rate: int
    characters: int

    # Derived metrics
    realtime_factor: float
    chars_per_second: float

    # Cost
    cost_usd: float

    # Output files
    audio_file: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["latency_stats"] = asdict(self.latency_stats)
        return result


@dataclass
class ProviderBenchmark:
    """Aggregate benchmark results for a provider."""
    provider: str
    results: List[BenchmarkResult] = field(default_factory=list)
    total_latency_mean_ms: float = 0
    total_cost_usd: float = 0
    avg_realtime_factor: float = 0
    languages_tested: List[str] = field(default_factory=list)

    def compute_aggregates(self):
        """Compute aggregate statistics."""
        if not self.results:
            return

        self.total_latency_mean_ms = statistics.mean(
            r.latency_stats.mean_ms for r in self.results
        )
        self.total_cost_usd = sum(r.cost_usd for r in self.results)
        self.avg_realtime_factor = statistics.mean(
            r.realtime_factor for r in self.results
        )
        self.languages_tested = list(set(r.language for r in self.results))


class BenchmarkRunner:
    """Run benchmarks across TTS providers."""

    def __init__(
        self,
        providers: Dict[str, TTSProvider],
        output_dir: Path,
        iterations: int = 10,
        warmup_runs: int = 1,
        skip_existing: bool = True,
    ):
        self.providers = providers
        self.output_dir = Path(output_dir)
        self.iterations = iterations
        self.warmup_runs = warmup_runs
        self.skip_existing = skip_existing
        self.results: Dict[str, ProviderBenchmark] = {}
        self._existing_results: Dict = {}

        # Always load existing results for merging (preserves other providers' data)
        self._load_existing_results()

    def _load_existing_results(self):
        """Load existing benchmark results if available."""
        results_path = self.output_dir / "metrics" / "benchmark_results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    self._existing_results = json.load(f)
                print(f"Loaded existing results from {results_path}")
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")
                self._existing_results = {}

    def _provider_has_complete_results(self, provider_key: str, provider: TTSProvider, test_texts: List[dict], languages: List[str]) -> bool:
        """Check if provider already has complete results and audio files."""
        # Check if provider exists in results
        if provider_key not in self._existing_results.get("providers", {}):
            return False

        existing = self._existing_results["providers"][provider_key]
        existing_samples = {r["sample_id"] for r in existing.get("results", [])}

        # Check each test text
        for sample in test_texts:
            if sample["language"] not in languages:
                continue
            if sample["language"] not in provider.config.supported_languages:
                continue

            sample_id = sample["id"]

            # Check if sample exists in results
            if sample_id not in existing_samples:
                return False

            # Check if audio file exists
            audio_dir = self.output_dir / "audio" / provider.name.lower().replace(" ", "_").replace("-", "")
            audio_path = audio_dir / f"{sample_id}.wav"
            if not audio_path.exists():
                return False

        return True

    def load_test_texts(self, config_path: Path) -> List[dict]:
        """Load test texts from config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        samples = config.get("samples", [])
        samples.extend(config.get("long_samples", []))
        return samples

    def benchmark_sample(
        self,
        provider: TTSProvider,
        sample: dict,
    ) -> BenchmarkResult:
        """Benchmark a single text sample."""
        text = sample["text"]
        language = sample["language"]
        sample_id = sample["id"]
        category = sample.get("category", "general")

        # Skip if provider doesn't support language
        if language not in provider.config.supported_languages:
            if language.startswith("zh") and "zh" not in provider.config.supported_languages:
                raise ValueError(f"{provider.name} does not support {language}")

        latencies = []
        ttfbs = []
        last_result: Optional[TTSResult] = None

        # Run iterations
        total_runs = self.iterations + self.warmup_runs
        for i in range(total_runs):
            try:
                result = provider.generate(text, language=language)
                last_result = result

                # Skip warmup runs for statistics
                if i >= self.warmup_runs:
                    latencies.append(result.latency_ms)
                    if result.ttfb_ms is not None:
                        ttfbs.append(result.ttfb_ms)

            except Exception as e:
                print(f"  Error on iteration {i}: {e}")
                continue

        if not latencies:
            raise RuntimeError(f"All iterations failed for {provider.name} on {sample_id}")

        # Calculate statistics
        latency_stats = LatencyStats(
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            p95_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            ttfb_mean_ms=statistics.mean(ttfbs) if ttfbs else None,
        )

        # Save audio file
        audio_dir = self.output_dir / "audio" / provider.name.lower().replace(" ", "_").replace("-", "")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{sample_id}.wav"

        if last_result:
            provider.save_audio(last_result, audio_path)

        # Calculate derived metrics
        realtime_factor = last_result.realtime_factor if last_result else 0
        chars_per_sec = last_result.chars_per_second if last_result else 0

        return BenchmarkResult(
            provider=provider.name,
            sample_id=sample_id,
            text=text,
            language=language,
            category=category,
            latency_stats=latency_stats,
            iterations=len(latencies),
            duration_seconds=last_result.duration_seconds if last_result else 0,
            sample_rate=last_result.sample_rate if last_result else 0,
            characters=len(text),
            realtime_factor=realtime_factor,
            chars_per_second=chars_per_sec,
            cost_usd=CostCalculator.calculate_cost(provider.name, len(text)),
            audio_file=str(audio_path),
        )

    def run_benchmark(
        self,
        test_texts: List[dict],
        languages: Optional[List[str]] = None,
    ) -> Dict[str, ProviderBenchmark]:
        """Run full benchmark across all providers and samples."""
        # Filter by language if specified
        languages = languages or ["en", "zh", "multilingual"]
        filtered_texts = [s for s in test_texts if s["language"] in languages]

        skipped_providers = []
        ran_providers = []

        for provider_name, provider in self.providers.items():
            # Check if we can skip this provider
            if self.skip_existing and self._provider_has_complete_results(
                provider_name, provider, test_texts, languages
            ):
                print(f"\n[SKIP] {provider.name} - already has complete results and audio")
                skipped_providers.append(provider_name)
                continue

            ran_providers.append(provider_name)
            print(f"\n{'='*50}")
            print(f"Benchmarking: {provider.name}")
            print(f"{'='*50}")

            provider_benchmark = ProviderBenchmark(provider=provider.name)

            for sample in filtered_texts:
                sample_id = sample["id"]
                language = sample["language"]

                # Skip if provider doesn't support language
                if language not in provider.config.supported_languages:
                    print(f"  Skipping {sample_id} - {provider.name} doesn't support {language}")
                    continue

                print(f"  Processing: {sample_id} ({language})...")

                try:
                    result = self.benchmark_sample(provider, sample)
                    provider_benchmark.results.append(result)
                    print(f"    Latency: {result.latency_stats.mean_ms:.1f}ms (mean)")
                    print(f"    Realtime factor: {result.realtime_factor:.1f}x")
                except Exception as e:
                    print(f"    ERROR: {e}")

            provider_benchmark.compute_aggregates()
            self.results[provider_name] = provider_benchmark

        # Summary
        if skipped_providers:
            print(f"\n[INFO] Skipped {len(skipped_providers)} providers with existing results: {', '.join(skipped_providers)}")
        if ran_providers:
            print(f"[INFO] Ran benchmarks for {len(ran_providers)} providers: {', '.join(ran_providers)}")

        return self.results

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file, merging with existing results."""
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Start with existing results (to preserve skipped providers)
        output = {
            "timestamp": datetime.now().isoformat(),
            "iterations": self.iterations,
            "providers": {},
        }

        # Copy existing provider results
        if self._existing_results:
            for name, data in self._existing_results.get("providers", {}).items():
                output["providers"][name] = data

        # Update with new results (overwrites if re-run)
        for name, benchmark in self.results.items():
            output["providers"][name] = {
                "total_latency_mean_ms": benchmark.total_latency_mean_ms,
                "total_cost_usd": benchmark.total_cost_usd,
                "avg_realtime_factor": benchmark.avg_realtime_factor,
                "languages_tested": benchmark.languages_tested,
                "results": [r.to_dict() for r in benchmark.results],
            }

        with open(metrics_dir / filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {metrics_dir / filename}")
        return metrics_dir / filename
