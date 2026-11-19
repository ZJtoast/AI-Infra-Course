"""Summarize the two-request vLLM course experiment without loading the model."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def read_requests(rows):
    arrivals = {}
    outputs = defaultdict(list)
    for row in rows:
        key = int(row["run"]), row["request"]
        stamp = int(row["elapsed_ns"])
        if row["event"] == "arrive":
            if key in arrivals:
                raise ValueError(f"duplicate arrival for {key}")
            arrivals[key] = stamp
        elif row["event"] == "output":
            outputs[key].append((int(row["tokens"]), stamp))
    if arrivals.keys() != outputs.keys():
        raise ValueError("incomplete request: unmatched arrival or output")
    runs = defaultdict(dict)
    for (run, request), arrival in arrivals.items():
        tokens = outputs[run, request]
        expected = {"A": 8, "B": 4}.get(request)
        if expected is None or [n for n, _ in tokens] != list(range(1, expected+1)):
            raise ValueError(f"expected one delivery per token for run {run}, request {request}")
        stamps = [stamp for _, stamp in tokens]
        if stamps[0] < arrival or stamps != sorted(stamps):
            raise ValueError(f"reversed timestamps in run {run}, request {request}")
        runs[run][request] = arrival, stamps
    if not runs or any(set(pair) != {"A", "B"} for pair in runs.values()):
        raise ValueError("expected both finished requests in every run")
    return runs


def summarize(runs, gap_limit_ms, first_limit_ms):
    measurements = defaultdict(list)
    good = 0
    for pair in runs.values():
        a0, a = pair["A"]
        b0, b = pair["B"]
        a_gaps = [right-left for left, right in zip(a, a[1:])]
        values = {
            "A 首词": a[0]-a0,
            "A 七个逐词间隔的平均值": (a[-1]-a[0])/(len(a)-1),
            "A 最长逐词间隔": max(a_gaps),
            "B 首词": b[0]-b0,
            "B 从到达到末词": b[-1]-b0,
            "两请求从首个到达到全部完成": max(a[-1], b[-1])-min(a0, b0),
        }
        for label, ns in values.items():
            measurements[label].append(ns/1e6)
        if max(a_gaps) <= gap_limit_ms*1e6 and b[0]-b0 <= first_limit_ms*1e6:
            good += 1
    for label, ms in measurements.items():
        print(f"{label}: 中位数 {median(ms):.3f} ms，范围 {min(ms):.3f}–{max(ms):.3f} ms")
    print(f"A 最长间隔 ≤ {gap_limit_ms:g} ms 且 B 首词 ≤ {first_limit_ms:g} ms："
          f"{good}/{len(runs)} 轮同时达标")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="the --csv output of one vllm_system_trace.py run")
    parser.add_argument("--a-gap-ms", type=float, default=25)
    parser.add_argument("--b-first-ms", type=float, default=40)
    args = parser.parse_args()
    with args.events.open(newline="", encoding="utf-8-sig") as file:
        summarize(read_requests(csv.DictReader(file)), args.a_gap_ms, args.b_first_ms)
