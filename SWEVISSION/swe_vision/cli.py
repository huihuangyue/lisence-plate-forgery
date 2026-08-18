"""
CLI entry point for the VLM Tool Call Agent.

Usage:
    python -m swe_vision.cli --image path/to/image.png "Describe what's in this image"
    python -m swe_vision.cli "What is 2^100? Use python to compute it."
    python -m swe_vision.cli --interactive

Environment Variables:
    OPENAI_API_KEY      - Your OpenAI API key
    OPENAI_BASE_URL     - (Optional) Custom API base URL
    OPENAI_MODEL        - (Optional) Model name, default: gpt-4o
"""

import argparse
import asyncio
import sys

from swe_vision.agent import VLMToolCallAgent
from swe_vision.config import DEFAULT_MODEL, MAX_ITERATIONS


async def async_main():
    parser = argparse.ArgumentParser(
        description="VLM Tool Call Agent - Agentic VLM with Docker Jupyter Notebook tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a question with an image
  python -m swe_vision.cli --image image.png "What objects are in this image?"

  # Computation question (no image)
  python -m swe_vision.cli "Compute the first 20 Fibonacci numbers"

  # Multiple images
  python -m swe_vision.cli --image img1.png --image img2.png "Compare these two images"

  # Interactive mode
  python -m swe_vision.cli --interactive

  # Custom model and API
  python -m swe_vision.cli --model gpt-4o --base-url https://api.openai.com/v1 "Hello"
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="The question or instruction for the agent",
    )
    parser.add_argument(
        "--image", "-i",
        action="append",
        default=[],
        help="Path to an image file (can be specified multiple times)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL (or set OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Max agentic loop iterations (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--save-trajectory",
        default=None,
        help="Directory to save trajectory (default: auto-generated under ./trajectories/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (minimal output)",
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reasoning mode (default: True). Use --no-reasoning to disable.",
    )

    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    agent = VLMToolCallAgent(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        save_trajectory=args.save_trajectory,
        reasoning=args.reasoning,
    )

    try:
        if args.interactive:
            await agent.run_interactive(args.image)
        elif args.query:
            answer = await agent.run(
                args.query,
                args.image if args.image else None,
            )
            if not args.verbose:
                print(answer)
        else:
            parser.print_help()
            print("\nError: Please provide a query or use --interactive mode.")
            sys.exit(1)
    finally:
        await agent.cleanup()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
