"""Trace a two-request vLLM engine run. See the Systems lab in lesson 4."""
import argparse
import csv
import os
from time import perf_counter_ns

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm import EngineArgs, LLMEngine, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="local Llama-compatible checkpoint")
    parser.add_argument("--budget", type=int, choices=(128, 192, 256, 320, 321, 512), default=128)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--csv", default="requests.csv")
    args = parser.parse_args()
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model, dtype="bfloat16", enforce_eager=True, tensor_parallel_size=1,
        max_model_len=512, max_num_seqs=4, max_num_batched_tokens=args.budget,
        enable_chunked_prefill=True, enable_prefix_caching=False, async_scheduling=False,
        gpu_memory_utilization=0.35, disable_log_stats=True,
    ))

    def submit(req, length, outputs):
        prompt = {"prompt_token_ids": [1000 + i % 71 for i in range(length)]}
        params = SamplingParams(temperature=0, max_tokens=outputs, ignore_eos=True)
        engine.add_request(req, prompt, params)

    submit("warm", 320, 4)
    while engine.has_unfinished_requests():
        engine.step()
    torch.cuda.synchronize()
    # Read the real scheduler decision for each step in this in-process V1 engine.
    scheduler = engine.engine_core.engine_core.scheduler
    schedule = scheduler.schedule
    scheduled = {}
    def observe_schedule(*args, **kwargs):
        with torch.cuda.nvtx.range("course/scheduler"):
            result = schedule(*args, **kwargs)
        scheduled.clear()
        for key, count in result.num_scheduled_tokens.items():
            scheduled[key.split("-")[0]] = count
        return result
    scheduler.schedule = observe_schedule
    entries = []
    for run in range(1 if args.capture else 5):
        step, b_added = 0, False
        seen = {"A": 0, "B": 0}
        if args.capture:
            torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("course/requests"):
            origin = perf_counter_ns()
            def log(event, req, count):
                entries.append([run+1, step, event, req, count, perf_counter_ns()-origin])
            log("arrive", "A", 32)
            submit("A", 32, 8)
            while engine.has_unfinished_requests():
                step += 1
                scheduled.clear()
                with torch.cuda.nvtx.range(f"course/step/{step}"):
                    output = engine.step()
                    for req, count in sorted(scheduled.items()):
                        log("scheduled", req, count)
                    for result in output:
                        req = result.request_id
                        total = len(result.outputs[0].token_ids)
                        if total > seen[req]:
                            seen[req] = total
                            log("output", req, total)
                            torch.cuda.nvtx.mark(f"course/output/{req}/{total}")
                if seen["A"] and not b_added:
                    log("arrive", "B", 320)
                    submit("B", 320, 4)
                    b_added = True
            assert seen == {"A": 8, "B": 4}, seen
        if args.capture:
            torch.cuda.cudart().cudaProfilerStop()
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "step", "event", "request", "tokens", "elapsed_ns"])
        writer.writerows(entries)


if __name__ == "__main__":
    main()
