#!/usr/bin/env python3
"""
Self-contained benchmark client for the MiniMax-M3-NVFP4 vLLM OpenAI endpoint.

NO local GPU required. NO third-party deps (stdlib only: urllib + threads).
Hits a *live* server over HTTP; safe to run while the container is serving.

Measures, at temperature=0 (greedy, deterministic):
  - decode throughput (tok/s) at batch size 1   <- key interactive metric
  - prefill throughput (tok/s) for a ~512-tok prompt
  - TTFT (time to first token) and TPOT (inter-token latency)
  - throughput under an offered-load sweep: concurrency 1, 4, 16

Uses the /v1/completions endpoint with streaming + ignore_eos so every request
decodes EXACTLY output_len tokens (stable, gibberish-tolerant: we measure speed,
not correctness). Tokens counted from server `usage` when available, else from
the number of streamed SSE chunks.

Output: one JSON file + a human-readable table to stdout.

Example:
  python3 bench_client.py --host http://localhost:8000 \
      --model /models/MiniMax-M3-NVFP4 \
      --out results/baseline.json
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------- #
def _post_stream(url, payload, timeout):
    """POST a streaming completion request. Yields raw SSE 'data:' JSON objects.

    Returns a generator; the first yielded value is the wall-clock time at which
    the request was *sent* so callers can compute TTFT precisely.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        chunk = line[len("data:"):].strip()
        if chunk == "[DONE]":
            break
        try:
            yield json.loads(chunk)
        except json.JSONDecodeError:
            continue


def wait_healthy(host, timeout_s=300, interval_s=3):
    """Block until host/v1/models responds, or raise after timeout_s.

    Guards against firing the benchmark into a server that is (re)loading, which
    would otherwise silently produce all-zero results.
    """
    models_url = host.rstrip("/") + "/v1/models"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:  # noqa: BLE001 - any failure means not ready yet
            last_err = e
        time.sleep(interval_s)
    raise RuntimeError(f"server at {models_url} not healthy after {timeout_s}s "
                       f"(last error: {last_err!r})")


def _build_prompt(approx_tokens, seed):
    """Build a prompt of roughly `approx_tokens` tokens.

    We use random integers separated by spaces so the tokenizer can't merge
    them; ~1 token per number-word is a safe lower bound, so we slightly
    over-generate and rely on the server-reported prompt_tokens for the
    *measured* prefill length.
    """
    rng = random.Random(seed)
    # A 4-digit number tokenizes to ~3 BPE tokens on this tokenizer, so to land
    # near `approx_tokens` we generate ~approx_tokens/3 number-words. The actual
    # measured length is taken from the server-reported prompt_tokens, so this
    # only needs to be approximately right.
    n_words = max(1, approx_tokens // 3)
    words = [str(rng.randint(1000, 9999)) for _ in range(n_words)]
    return "The quick brown fox. " + " ".join(words)


# --------------------------------------------------------------------------- #
# Single-request measurement
# --------------------------------------------------------------------------- #
class ReqResult:
    __slots__ = ("ok", "ttft", "latency", "gen_tokens", "prompt_tokens",
                 "itls", "error")

    def __init__(self):
        self.ok = False
        self.ttft = None          # seconds to first token
        self.latency = None       # total seconds
        self.gen_tokens = 0       # decoded tokens
        self.prompt_tokens = 0    # server-reported prefill length
        self.itls = []            # inter-token latencies (s)
        self.error = None


def run_request(url, model, prompt, output_len, timeout):
    """Run one streaming completion, return a ReqResult with timings."""
    r = ReqResult()
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "min_tokens": output_len,     # force full-length decode
        "temperature": 0.0,           # greedy / deterministic
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,           # don't stop early on EOS
    }
    t0 = time.perf_counter()
    last = t0
    first_seen = False
    try:
        for obj in _post_stream(url, payload, timeout):
            now = time.perf_counter()
            # usage-only final chunk carries token accounting
            usage = obj.get("usage")
            choices = obj.get("choices") or []
            text = ""
            if choices:
                text = choices[0].get("text", "") or ""
            if text:
                if not first_seen:
                    r.ttft = now - t0
                    first_seen = True
                else:
                    r.itls.append(now - last)
                last = now
                r.gen_tokens += 1
            if usage:
                if usage.get("completion_tokens"):
                    r.gen_tokens = usage["completion_tokens"]
                if usage.get("prompt_tokens"):
                    r.prompt_tokens = usage["prompt_tokens"]
        r.latency = time.perf_counter() - t0
        r.ok = first_seen and r.gen_tokens > 0
        if not r.ok and r.error is None:
            r.error = "no tokens produced"
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError,
            TimeoutError, OSError) as e:
        r.error = repr(e)
    return r


# --------------------------------------------------------------------------- #
# Concurrency sweep
# --------------------------------------------------------------------------- #
def run_load(url, model, num_prompts, input_len, output_len, concurrency,
             timeout, seed):
    """Fire `num_prompts` requests through a pool of `concurrency` workers.

    Returns aggregate metrics for this offered-load point.
    """
    prompts = [_build_prompt(input_len, seed + i) for i in range(num_prompts)]
    results = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(run_request, url, model, p, output_len, timeout)
            for p in prompts
        ]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    total_gen = sum(r.gen_tokens for r in ok)
    total_prompt = sum(r.prompt_tokens for r in ok)

    def pct(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        return s[k]

    ttfts = [r.ttft for r in ok if r.ttft is not None]
    # TPOT per request = mean inter-token latency (s) -> report in ms
    tpots_ms = [statistics.mean(r.itls) * 1000.0 for r in ok if r.itls]
    all_itls_ms = [x * 1000.0 for r in ok for x in r.itls]
    per_req_decode = [
        (r.gen_tokens - 1) / (r.latency - r.ttft)
        for r in ok if r.ttft is not None and (r.latency - r.ttft) > 0
        and r.gen_tokens > 1
    ]

    return {
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "input_len_requested": input_len,
        "output_len": output_len,
        "completed": len(ok),
        "failed": failed,
        "errors": list({r.error for r in results if r.error})[:5],
        "wall_time_s": round(wall, 4),
        "prompt_tokens_total": total_prompt,
        "gen_tokens_total": total_gen,
        "mean_prompt_tokens": round(total_prompt / len(ok), 1) if ok else None,
        # System-level throughput (all concurrent streams summed)
        "output_throughput_tok_s": round(total_gen / wall, 2) if wall else None,
        "total_throughput_tok_s":
            round((total_gen + total_prompt) / wall, 2) if wall else None,
        # Request rate
        "request_throughput_req_s": round(len(ok) / wall, 4) if wall else None,
        # Per-request (interactive) decode speed
        "per_request_decode_tok_s_mean":
            round(statistics.mean(per_req_decode), 2) if per_req_decode else None,
        "per_request_decode_tok_s_median":
            round(statistics.median(per_req_decode), 2) if per_req_decode else None,
        # TTFT (ms)
        "ttft_ms_mean": round(statistics.mean(ttfts) * 1000, 2) if ttfts else None,
        "ttft_ms_median": round(statistics.median(ttfts) * 1000, 2) if ttfts else None,
        "ttft_ms_p99": round(pct(ttfts, 99) * 1000, 2) if ttfts else None,
        # TPOT / ITL (ms)
        "tpot_ms_mean": round(statistics.mean(tpots_ms), 3) if tpots_ms else None,
        "tpot_ms_median": round(statistics.median(tpots_ms), 3) if tpots_ms else None,
        "itl_ms_p99": round(pct(all_itls_ms, 99), 3) if all_itls_ms else None,
    }


# --------------------------------------------------------------------------- #
# Prefill measurement (TTFT of a long prompt with 1 output token)
# --------------------------------------------------------------------------- #
def measure_prefill(url, model, input_len, repeats, timeout, seed):
    """Prefill throughput = prompt_tokens / TTFT, max_tokens=1, bs=1.

    TTFT for a single-token request is dominated by the prefill forward pass,
    so prompt_tokens / TTFT is a clean prefill tok/s estimate.
    """
    samples = []
    for i in range(repeats):
        prompt = _build_prompt(input_len, seed + 1000 + i)
        r = run_request(url, model, prompt, output_len=1, timeout=timeout)
        if r.ok and r.ttft and r.prompt_tokens:
            samples.append((r.prompt_tokens, r.ttft))
    if not samples:
        return {"error": "prefill measurement failed", "samples": 0}
    rates = [pt / ttft for pt, ttft in samples]
    ttfts = [ttft for _, ttft in samples]
    ptoks = [pt for pt, _ in samples]
    return {
        "samples": len(samples),
        "mean_prompt_tokens": round(statistics.mean(ptoks), 1),
        "ttft_ms_mean": round(statistics.mean(ttfts) * 1000, 2),
        "ttft_ms_median": round(statistics.median(ttfts) * 1000, 2),
        "prefill_throughput_tok_s_mean": round(statistics.mean(rates), 2),
        "prefill_throughput_tok_s_median": round(statistics.median(rates), 2),
    }


# --------------------------------------------------------------------------- #
# Pretty table
# --------------------------------------------------------------------------- #
def render_table(report):
    lines = []
    A = lines.append
    m = report["meta"]
    A("=" * 78)
    A("  MiniMax-M3-NVFP4 vLLM serving benchmark")
    A("=" * 78)
    A(f"  host            : {m['host']}")
    A(f"  model           : {m['model']}")
    A(f"  timestamp       : {m['timestamp']}")
    A(f"  conditions      : {m['conditions']}")
    A("-" * 78)

    d = report.get("decode_bs1", {})
    A("  DECODE @ batch size 1 (key interactive metric)")
    A(f"    decode throughput : {d.get('per_request_decode_tok_s_median')} tok/s "
      f"(median over {d.get('completed')} reqs, {d.get('output_len')} out-tok each)")
    A(f"    TTFT              : {d.get('ttft_ms_median')} ms (median)")
    A(f"    TPOT              : {d.get('tpot_ms_median')} ms/tok (median inter-token)")
    A("-" * 78)

    p = report.get("prefill", {})
    A("  PREFILL (~512-tok prompt, max_tokens=1)")
    A(f"    prompt tokens     : {p.get('mean_prompt_tokens')}")
    A(f"    TTFT              : {p.get('ttft_ms_median')} ms (median)")
    A(f"    prefill throughput: {p.get('prefill_throughput_tok_s_median')} tok/s (median)")
    A("-" * 78)

    A("  OFFERED-LOAD SWEEP (system output throughput, all streams summed)")
    A(f"    {'conc':>5} {'out tok/s':>11} {'req/s':>8} {'per-req tok/s':>14} "
      f"{'TTFT ms':>9} {'TPOT ms':>9} {'fail':>5}")
    for s in report.get("sweep", []):
        A(f"    {s['concurrency']:>5} "
          f"{str(s.get('output_throughput_tok_s')):>11} "
          f"{str(s.get('request_throughput_req_s')):>8} "
          f"{str(s.get('per_request_decode_tok_s_median')):>14} "
          f"{str(s.get('ttft_ms_median')):>9} "
          f"{str(s.get('tpot_ms_median')):>9} "
          f"{str(s.get('failed')):>5}")
    A("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", default="/models/MiniMax-M3-NVFP4")
    ap.add_argument("--endpoint", default="/v1/completions")
    ap.add_argument("--input-len", type=int, default=512,
                    help="approx prompt tokens for prefill + decode tests")
    ap.add_argument("--output-len", type=int, default=128,
                    help="decode tokens per request for decode/sweep tests")
    ap.add_argument("--decode-reqs", type=int, default=5,
                    help="sequential bs=1 requests for the decode metric")
    ap.add_argument("--prefill-reps", type=int, default=5)
    ap.add_argument("--num-prompts", type=int, default=32,
                    help="requests per concurrency point in the sweep")
    ap.add_argument("--concurrency", default="1,4,16",
                    help="comma-separated offered-load levels")
    ap.add_argument("--sweep-output-len", type=int, default=64,
                    help="shorter output for sweep to keep runtime bounded")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--health-timeout", type=float, default=300.0,
                    help="seconds to wait for the server to be reachable before starting")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup requests (discarded) before measuring")
    ap.add_argument("--out", default="results/baseline.json")
    ap.add_argument("--conditions", default=None,
                    help="free-text serving conditions recorded in the JSON meta "
                         "(set this to match the actual live container config)")
    args = ap.parse_args()

    url = args.host.rstrip("/") + args.endpoint
    conditions = args.conditions or (
        "max-model-len 65536, block-size 128, bf16 KV, TP4, "
        "enforce-eager, gpu-mem-util 0.95")

    print(f"[bench] target {url}  model={args.model}", file=sys.stderr)

    # ---- wait for server health (it may be (re)loading) ----
    print("[bench] waiting for server health ...", file=sys.stderr)
    wait_healthy(args.host, timeout_s=args.health_timeout)
    print("[bench] server healthy.", file=sys.stderr)

    # ---- warmup (server may JIT/compile graphs on first hits) ----
    for i in range(args.warmup):
        p = _build_prompt(args.input_len, args.seed + 9000 + i)
        run_request(url, args.model, p, output_len=8, timeout=args.timeout)
    print(f"[bench] warmup done ({args.warmup} reqs)", file=sys.stderr)

    # ---- decode @ bs1 (sequential, no contention) ----
    print("[bench] decode @ bs1 ...", file=sys.stderr)
    decode = run_load(url, args.model, num_prompts=args.decode_reqs,
                      input_len=args.input_len, output_len=args.output_len,
                      concurrency=1, timeout=args.timeout, seed=args.seed)

    # ---- prefill ----
    print("[bench] prefill ...", file=sys.stderr)
    prefill = measure_prefill(url, args.model, input_len=args.input_len,
                              repeats=args.prefill_reps, timeout=args.timeout,
                              seed=args.seed)

    # ---- concurrency sweep ----
    sweep = []
    for c in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        print(f"[bench] sweep concurrency={c} ...", file=sys.stderr)
        sweep.append(run_load(
            url, args.model, num_prompts=args.num_prompts,
            input_len=args.input_len, output_len=args.sweep_output_len,
            concurrency=c, timeout=args.timeout, seed=args.seed))

    report = {
        "meta": {
            "host": args.host,
            "model": args.model,
            "endpoint": args.endpoint,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "conditions": conditions,
            "params": vars(args),
        },
        "decode_bs1": decode,
        "prefill": prefill,
        "sweep": sweep,
    }

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    table = render_table(report)
    print("\n" + table)
    print(f"\n[bench] JSON written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
