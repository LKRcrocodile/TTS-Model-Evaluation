#!/usr/bin/env python3
"""Main script to run TTS evaluation benchmark."""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers import (
    AzureTTSProvider,
    AzureStreamingProvider,
    ElevenLabsTTSProvider,
    ElevenLabsTurboProvider,
    MiniMaxTTSProvider,
    MiniMaxStreamingProvider,
    MiniMaxPCMStreamingProvider,
    QwenTTSProvider,
    QwenStreamingProvider,
    QwenSelfHostedProvider,
    LuxTTSProvider,
)
from src.evaluation import BenchmarkRunner
from src.presentation import DashboardGenerator


def main():
    parser = argparse.ArgumentParser(description="Run TTS Model Evaluation")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["azure", "elevenlabs", "minimax", "qwen3", "luxtts"],
        help="Providers to benchmark (default: all)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "zh", "multilingual"],
        help="Languages to test (default: en zh multilingual)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per sample (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent.parent / "config",
        help="Config directory",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmark and only generate dashboard from existing results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all providers even if they have existing results",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    print("=" * 60)
    print("TTS Model Evaluation")
    print("=" * 60)
    print(f"Providers: {args.providers}")
    print(f"Languages: {args.languages}")
    print(f"Iterations: {args.iterations}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    if not args.skip_benchmark:
        # Initialize providers
        provider_map = {
            "azure": AzureTTSProvider,
            "azure_streaming": AzureStreamingProvider,
            "elevenlabs": ElevenLabsTTSProvider,
            "elevenlabs_turbo": ElevenLabsTurboProvider,
            "minimax": MiniMaxTTSProvider,
            "minimax_streaming": MiniMaxStreamingProvider,
            "minimax_pcm": MiniMaxPCMStreamingProvider,
            "qwen3": QwenTTSProvider,
            "qwen3_streaming": QwenStreamingProvider,
            "qwen3_selfhosted": QwenSelfHostedProvider,
            "luxtts": LuxTTSProvider,
        }

        providers = {}
        for name in args.providers:
            if name not in provider_map:
                print(f"Warning: Unknown provider '{name}', skipping")
                continue

            try:
                provider = provider_map[name]()
                provider.initialize()
                providers[name] = provider
                print(f"Initialized: {provider.name}")
            except Exception as e:
                print(f"Failed to initialize {name}: {e}")
                continue

        if not providers:
            print("No providers initialized. Exiting.")
            sys.exit(1)

        # Run benchmark
        runner = BenchmarkRunner(
            providers=providers,
            output_dir=args.output_dir,
            iterations=args.iterations,
            skip_existing=not args.force,
        )

        # Load test texts
        test_texts = runner.load_test_texts(args.config_dir / "test_texts.yaml")
        print(f"\nLoaded {len(test_texts)} test samples")

        # Run
        runner.run_benchmark(test_texts, languages=args.languages)

        # Save results
        results_path = runner.save_results()
    else:
        results_path = args.output_dir / "metrics" / "benchmark_results.json"
        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            sys.exit(1)

    # Generate dashboard
    print("\nGenerating dashboard...")
    dashboard = DashboardGenerator(results_path, args.output_dir)
    dashboard_path = dashboard.generate()

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results: {args.output_dir / 'metrics' / 'benchmark_results.json'}")
    print(f"Dashboard: {dashboard_path}")
    print(f"\nOpen the dashboard in a browser to view results:")
    print(f"  open {dashboard_path}")


if __name__ == "__main__":
    main()
