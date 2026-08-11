from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from datasets import load_dataset
from openai import OpenAI  # 修改：引入 OpenAI SDK


MODELS = (
    (
        "DeepSeek-V4-Pro",
        os.getenv("DEEPSEEK_V4_PRO_MODEL", "deepseek-chat"),
    ),
    (
        "DeepSeek-V4-Flash",
        os.getenv("DEEPSEEK_V4_FLASH_MODEL", "deepseek-chat"),
    ),
)
METRICS = (
    ("GSM8K EM", "Accuracy (%)"),
    ("XSum custom ROUGE-L", "Custom ROUGE-L F1 (%)"),
    ("Prefill speed", "Estimated tokens/s"),
    ("Decode speed", "Observed tokens/s"),
)
COLORS = ("#1A73E8", "#34A853")


def positive_int(value: str) -> int:
    """Parses a positive integer argument.

    Args:
        value: Command-line value.

    Returns:
        Positive integer.

    Raises:
        ArgumentTypeError: If the value is not positive.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments.
    """
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed-trials", type=positive_int, default=3)
    parser.add_argument("--speed-output-tokens", type=positive_int, default=96)
    parser.add_argument("--max-input-chars", type=positive_int, default=12000)
    parser.add_argument(
        "--api-base",
        default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="DeepSeek/OpenAI API base URL",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=output_dir / "deepseek_v4_benchmarks.csv",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=output_dir / "deepseek_v4_benchmarks.png",
    )
    return parser.parse_args()


def get_api_key() -> str:
    """Reads the DeepSeek API key from the environment.

    Returns:
        DeepSeek API key.

    Raises:
        RuntimeError: No API keys found.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("apikey")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or OPENAI_API_KEY in the environment.")
    return api_key


def build_client(
        api_key: str, api_base: str, timeout: float
) -> OpenAI:
    """Builds a simple deepseek client.

    Args:
        api_key: key got from fore function.
        provider: provider name.
        timeout: request timeout in seconds.

    Returns:
        a client.
    """
    return OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=timeout,
    )


def generate_text(
        client: OpenAI,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
) -> str:
    """Generates text with bounded retries.

    Args:
        client: deepseek client.
        model_id: model id.
        prompt: prompt.
        max_new_tokens: Maximum generated token count.

    Returns:
        Generated text.

    Raises:
        RuntimeError: in case requests fail.
    """
    error: Exception | None = None
    for attempt in range(3):
        try:
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            content = result.choices[0].message.content
            return str(content or "")
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Inference failed for {model_id}: {error}") from error


def load_benchmark_data(
        hf_token: str | None, samples: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Loads benchmark samples from Hugging Face.

    Args:
        api_key: hugging face api key but sets to None.
        samples: Number of rows per benchmark.
        seed: Sampling seed.

    Returns:
        GSM8K and XSum sample lists.
    """
    gsm8k = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        token=hf_token,
    )
    xsum = load_dataset(
        "EdinburghNLP/xsum",
        split="test",
        token=hf_token,
    )
    selected_gsm8k = gsm8k.shuffle(seed=seed).select(
        range(min(samples, len(gsm8k)))
    )
    selected_xsum = xsum.shuffle(seed=seed).select(
        range(min(samples, len(xsum)))
    )
    return list(selected_gsm8k), list(selected_xsum)


def extract_number(text: str) -> Decimal | None:
    """Extracts the final numeric answer from text.

    Args:
        text: Model or reference answer.

    Returns:
        Parsed decimal value when available.
    """
    pattern = r"[-+]?\d[\d,]*(?:\.\d+)?"
    final_match = re.search(
        rf"FINAL\s*:\s*\$?\s*({pattern})",
        text,
        flags=re.IGNORECASE,
    )
    candidates = re.findall(pattern, text)
    raw_value = final_match.group(1) if final_match else None
    if raw_value is None and candidates:
        raw_value = candidates[-1]
    if raw_value is None:
        return None
    try:
        return Decimal(raw_value.replace(",", ""))
    except InvalidOperation:
        return None


def evaluate_gsm8k(
        client: OpenAI,
        model_id: str,
        rows: list[dict[str, Any]],
) -> float:
    """Computes GSM8K numeric exact-match accuracy.

    Args:
        client: Hugging Face inference client.
        model_id: Hugging Face model identifier.
        rows: GSM8K examples.

    Returns:
        Accuracy percentage.
    """
    correct = 0
    for row in rows:
        prompt = (
            "Solve the math problem. Show concise reasoning and end with "
            f"FINAL: <number>.\n\nQuestion: {row['question']}\nAnswer:"
        )
        prediction = generate_text(client, model_id, prompt, 384)
        expected = extract_number(str(row["answer"]).rsplit("####", 1)[-1])
        predicted = extract_number(prediction)
        correct += expected is not None and predicted == expected
    return 100.0 * correct / len(rows)


def tokenize_words(text: str) -> list[str]:
    """Tokenizes text for ROUGE-L scoring.

    Args:
        text: Input text.

    Returns:
        Lowercase alphanumeric tokens.
    """
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def lcs_length(left: list[str], right: list[str]) -> int:
    """Computes a token-level longest common subsequence.

    Args:
        left: First token sequence.
        right: Second token sequence.

    Returns:
        Longest common subsequence length.
    """
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Computes token-level ROUGE-L F1.

    Args:
        candidate: Generated summary.
        reference: Reference summary.

    Returns:
        ROUGE-L F1 score from zero to one.
    """
    candidate_tokens = tokenize_words(candidate)
    reference_tokens = tokenize_words(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    common = lcs_length(candidate_tokens, reference_tokens)
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_xsum(
        client: OpenAI,
        model_id: str,
        rows: list[dict[str, Any]],
        max_input_chars: int,
) -> float:
    """Computes mean XSum ROUGE-L F1.

    Args:
        client: Hugging Face inference client.
        model_id: Hugging Face model identifier.
        rows: XSum examples.
        max_input_chars: Maximum source length in characters.

    Returns:
        Mean ROUGE-L F1 percentage.
    """
    scores = []
    for row in rows:
        document = str(row["document"])[:max_input_chars]
        prompt = (
            "Write one faithful, concise sentence summarizing this article. "
            f"Return only the summary.\n\nArticle:\n{document}\n\nSummary:"
        )
        prediction = generate_text(client, model_id, prompt, 160)
        scores.append(rouge_l_f1(prediction, str(row["summary"])))
    return 100.0 * statistics.fmean(scores)


def build_speed_prompt() -> str:
    """Builds a fixed prompt for client-observed speed tests.

    Returns:
        Repeated long-context prompt.
    """
    passage = (
        "Reliable systems separate measurement from interpretation, record "
        "assumptions, and preserve enough evidence for independent review. "
    )
    return (
            "Read the following material and produce a structured synthesis with "
            "eight numbered findings.\n\n" + passage * 160
    )


def measure_speed(
        client: OpenAI,
        model_id: str,
        trials: int,
        output_tokens: int,
) -> tuple[float, float]:
    """Measures client-estimated prefill and observed decode speeds.

    Args:
        client: Hugging Face inference client.
        model_id: Hugging Face model identifier.
        trials: Number of timed trials.
        output_tokens: Maximum output tokens per trial.

    Returns:
        Median prefill and decode tokens per second.

    Raises:
        RuntimeError: If the provider does not stream usable output.
    """
    prompt = build_speed_prompt()
    messages = [{"role": "user", "content": prompt}]

    warmup = client.chat.completions.create(
        messages=messages,
        model=model_id,
        max_tokens=1,
        temperature=0.0,
    )
    input_tokens = warmup.usage.prompt_tokens if warmup.usage else None
    if input_tokens is None or input_tokens <= 0:
        raise RuntimeError("The provider did not report prompt token usage.")
    prefill_speeds = []
    decode_speeds = []
    for _ in range(trials):
        started_at = time.perf_counter()
        stream = client.chat.completions.create(
            messages=messages,
            model=model_id,
            max_tokens=output_tokens,
            temperature=0.0,
            stream=True,
        )
        first_token_at = None
        generated_tokens = 0
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content or ""
            if not text:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            generated_tokens += 1
        finished_at = time.perf_counter()
        if first_token_at is None:
            raise RuntimeError(
                "The selected provider returned an empty stream."
            )
        time_to_first_token = first_token_at - started_at
        decode_seconds = finished_at - first_token_at
        prefill_speeds.append(input_tokens / time_to_first_token)
        if generated_tokens > 1 and decode_seconds > 0.0:
            decode_speeds.append((generated_tokens - 1) / decode_seconds)
        else:
            decode_speeds.append(0.0)
    return statistics.median(prefill_speeds), statistics.median(decode_speeds)


def collect_results(
        client: OpenAI,
        hf_token: str | None,
        provider: str,
        samples: int,
        seed: int,
        speed_trials: int,
        speed_output_tokens: int,
        max_input_chars: int,
) -> list[dict[str, str]]:
    """Runs both models and collects chart-ready metrics.

    Args:
        client: Hugging Face inference client.
        api_key: Hugging Face API key.
        provider: Inference provider name.
        samples: Number of quality examples per dataset.
        seed: Sampling seed.
        speed_trials: Number of timed speed trials.
        speed_output_tokens: Maximum generated tokens per speed trial.
        max_input_chars: Maximum XSum source length.

    Returns:
        Metric rows for CSV output.
    """
    gsm8k_rows, xsum_rows = load_benchmark_data(hf_token, samples, seed)
    measured_at = datetime.now(timezone.utc).isoformat()
    results = []
    for model_name, model_id in MODELS:
        print(f"Evaluating {model_name}...")
        gsm8k_score = evaluate_gsm8k(client, model_id, gsm8k_rows)
        xsum_score = evaluate_xsum(
            client,
            model_id,
            xsum_rows,
            max_input_chars,
        )
        prefill_speed, decode_speed = measure_speed(
            client,
            model_id,
            speed_trials,
            speed_output_tokens,
        )
        values = (
            (
                "GSM8K EM",
                gsm8k_score,
                "percent",
                samples,
                "test",
                "sampled zero-shot numeric EM",
            ),
            (
                "XSum custom ROUGE-L",
                xsum_score,
                "percent",
                samples,
                "test",
                f"sampled zero-shot; max {max_input_chars} source characters",
            ),
            (
                "Prefill speed",
                prefill_speed,
                "estimated tokens/second",
                speed_trials,
                "fixed prompt",
                "provider prompt tokens / client TTFT",
            ),
            (
                "Decode speed",
                decode_speed,
                "observed tokens/second",
                speed_trials,
                "fixed prompt",
                "stream token count / client decode time",
            ),
        )
        for metric, value, unit, sample_size, split, protocol in values:
            results.append(
                {
                    "model": model_name,
                    "model_id": model_id,
                    "metric": metric,
                    "value": f"{value:.4f}",
                    "unit": unit,
                    "sample_size": str(sample_size),
                    "provider": provider,
                    "split": split,
                    "seed": str(seed),
                    "protocol": protocol,
                    "measured_at": measured_at,
                }
            )
    return results


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Writes benchmark rows to CSV.

    Args:
        path: CSV output path.
        rows: Benchmark result rows.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Reads benchmark rows from CSV.

    Args:
        path: CSV input path.

    Returns:
        Parsed CSV rows.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def plot_csv(csv_path: Path, chart_path: Path) -> None:
    """Plots benchmark bars from a generated CSV file.

    Args:
        csv_path: CSV input path.
        chart_path: PNG output path.

    Returns:
        None.
    """
    rows = read_csv(csv_path)
    model_names = [model_name for model_name, _ in MODELS]
    values = {
        (row["metric"], row["model"]): float(row["value"]) for row in rows
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (metric, ylabel) in zip(axes.flat, METRICS, strict=True):
        heights = [values[(metric, model)] for model in model_names]
        bars = axis.bar(model_names, heights, color=COLORS, width=0.62)
        metric_row = next(row for row in rows if row["metric"] == metric)
        sample_size = metric_row["sample_size"]
        size_label = "trials" if "speed" in metric else "n"
        axis.set_title(f"{metric} ({size_label}={sample_size})")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f", padding=3)
        if metric in {"GSM8K EM", "XSum custom ROUGE-L"}:
            axis.set_ylim(0.0, 100.0)
    provider = rows[0]["provider"]
    figure.suptitle(f"DeepSeek V4 comparison via DeepSeek API ({provider})")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Runs evaluation, writes CSV, and plots the chart.

    Returns:
        None.
    """
    args = parse_args()
    api_key = get_api_key()
    hf_token = os.getenv("HF_TOKEN")
    client = build_client(api_key, args.api_base, args.timeout)
    rows = collect_results(
        client,
        hf_token,
        "deepseek-api",
        args.samples,
        args.seed,
        args.speed_trials,
        args.speed_output_tokens,
        args.max_input_chars,
    )
    write_csv(args.csv, rows)
    plot_csv(args.csv, args.chart)
    print(f"CSV: {args.csv}")
    print(f"Chart: {args.chart}")


if __name__ == "__main__":
    main()