"""
Task-specific calibration harness.

Measures real generation speed on YOUR actual task (short, grammar-constrained
XML output for IT ticket triage) rather than generic benchmark tokens. Grammar
constraining adds real per-token overhead (checking which tokens are valid at
each step), which llama-bench / llama-batched-bench won't capture since they
generate unconstrained tokens.

Run this AFTER the baseline benchmarks in 01_setup_and_baseline_bench.md, once
you've confirmed the models load and roughly what throughput to expect.

Requires: pip install llama-cpp-python --break-system-packages
(build with CUDA: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --break-system-packages --force-reinstall --no-cache-dir)
"""

import time
import csv
import json
from pathlib import Path
from llama_cpp import Llama, LlamaGrammar

# ---- CONFIG: fill in your actual model paths ----
MODELS = {
    "qwen2.5-3b": "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "llama-3.2-3b": "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "qwen2.5-7b": "./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}

# A placeholder GBNF grammar matching the rough complexity of your real schema.
# Swap this for your actual grammar file once it's built (Path("schema.gbnf").read_text()).
PLACEHOLDER_GRAMMAR = r"""
root ::= "<ticket>" ws request-type ws resource ws scope ws severity ws count ws "</ticket>"
request-type ::= "<request_type>" ("access_grant" | "access_revoke" | "tool_install" | "password_reset" | "hardware_request") "</request_type>"
resource ::= "<resource>" [a-zA-Z0-9 ]+ "</resource>"
scope ::= "<scope>" ("individual" | "team" | "department" | "org_wide") "</scope>"
severity ::= "<severity>" ("low" | "medium" | "high") "</severity>"
count ::= "<requester_count_estimate>" ("1" | "~5-10" | "~10-50" | "50+") "</requester_count_estimate>"
ws ::= [ \t\n]*
"""

# A handful of representative queries -- mix of short/ambiguous/unambiguous,
# doesn't need to be your full dataset, just representative of real length/difficulty.
SAMPLE_QUERIES = [
    "Grant access to Claude Code.",
    "I forgot my email password, please reset it.",
    "The entire DevOps team needs sudo access on the deployment cluster.",
    "We need new laptops for the new office.",
    "Set up VPN for the new hires.",
]

BATCH_LEVELS = [1, 4, 8, 16]
REPS_PER_CONDITION = 10  # keep small for calibration; scale up for the real run

SYSTEM_PROMPT = (
    "You are an IT ticket triage assistant. Given a short employee request, "
    "output an XML ticket with fields: request_type, resource, scope, "
    "severity, requester_count_estimate, justification. Infer scope and severity from context."
)


def build_prompt(query: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nRequest: {query}\n\nOutput:"


def calibrate_single_sequence(model_name: str, model_path: str, grammar: LlamaGrammar, results: list):
    """Batch=1 baseline: realistic per-query latency for your actual task."""
    print(f"\n=== {model_name} | batch=1 (task-specific) ===")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        seed=42,
        verbose=False,
    )

    for query in SAMPLE_QUERIES:
        prompt = build_prompt(query)
        timings = []
        token_counts = []
        for _ in range(REPS_PER_CONDITION):
            start = time.perf_counter()
            out = llm(
                prompt,
                grammar=grammar,
                temperature=0.0,
                max_tokens=200,
            )
            elapsed = time.perf_counter() - start
            n_tokens = out["usage"]["completion_tokens"]
            timings.append(elapsed)
            token_counts.append(n_tokens)

        avg_time = sum(timings) / len(timings)
        avg_tokens = sum(token_counts) / len(token_counts)
        tok_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

        print(f"  query={query[:40]!r:42s} avg_time={avg_time:.3f}s "
              f"avg_tokens={avg_tokens:.0f} tok/s={tok_per_sec:.1f}")

        results.append({
            "model": model_name,
            "batch": 1,
            "query": query,
            "avg_time_sec": round(avg_time, 4),
            "avg_completion_tokens": round(avg_tokens, 1),
            "tok_per_sec": round(tok_per_sec, 1),
        })

    del llm  # free VRAM before loading next model


def main():
    grammar_text = Path("ticket_schema.gbnf").read_text()
    grammar = LlamaGrammar.from_string(grammar_text)
    results = []

    for model_name, model_path in MODELS.items():
        if not Path(model_path).exists():
            print(f"SKIPPING {model_name}: file not found at {model_path}")
            continue
        calibrate_single_sequence(model_name, model_path, grammar, results)

    # Save results
    out_path = Path("calibration_results.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {out_path.resolve()}")
    print("\nNOTE: This script measures batch=1 only (llama-cpp-python's high-level")
    print("API doesn't expose true parallel-batch decoding easily). For real batch=4/8/16")
    print("throughput numbers, rely on llama-batched-bench from step 1 -- that tool")
    print("uses llama.cpp's low-level batch API correctly. This script's job is just to")
    print("show you the grammar-constraint overhead vs the unconstrained baseline.")


if __name__ == "__main__":
    main()
