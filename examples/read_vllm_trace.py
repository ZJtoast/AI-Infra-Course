"""Inspect a vLLM engine step in a single-GPU course Nsight Systems SQLite export."""
import argparse
from bisect import bisect_right
from collections import defaultdict
from math import ceil
import sqlite3
from pathlib import Path


def device_activity(db, start, end):
    # Some traces contain no copies or memory fills. Nsight then omits the table.
    available = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    queries = ["""SELECT k.start,k.end,s.value,k.correlationId,k.globalPid FROM
                  CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName"""]
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in available:
        queries.append("""SELECT start,end,CASE copyKind WHEN 1 THEN 'H2D' WHEN 2 THEN 'D2H'
                          WHEN 8 THEN 'D2D' ELSE 'memcpy' END,correlationId,globalPid
                          FROM CUPTI_ACTIVITY_KIND_MEMCPY""")
    if "CUPTI_ACTIVITY_KIND_MEMSET" in available:
        queries.append("SELECT start,end,'memset',correlationId,globalPid FROM CUPTI_ACTIVITY_KIND_MEMSET")
    query = " UNION ALL ".join(q+" WHERE end>? AND start<?" for q in queries)+" ORDER BY 1,2"
    return list(db.execute(query, (start, end)*len(queries)))


def gaps_in(window, activities):
    start, end = window
    cursor = start
    gaps = []
    for left, right, label, correlation, pid in activities:
        if right <= start or left >= end:
            continue
        if left > cursor:
            gaps.append((cursor, left, label, correlation, pid))
        cursor = max(cursor, min(right, end))
    if cursor < end:
        gaps.append((cursor, end, "end of window", None, None))
    return gaps


def duration(gaps):
    return sum(right-left for left, right, *_ in gaps)


def step_window(db, step):
    name = f"course/step/{step}"
    windows = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text=?", (name,)).fetchall()
    if len(windows) != 1:
        raise ValueError(f"expected one {name} NVTX range, found {len(windows)}")
    return windows[0]


def inspect(db, step):
    start, end = step_window(db, step)
    activities = device_activity(db, start, end)
    gaps = gaps_in((start, end), activities)
    idle = duration(gaps)
    print(f"course/step/{step} window={(end-start)/1000:.3f} us; "
          f"kernel/copy/fill union={(end-start-idle)/1000:.3f} us; gaps={idle/1000:.3f} us")
    for left, right, label, correlation, pid in sorted(gaps, key=lambda g: g[1]-g[0], reverse=True)[:3]:
        print(f"  +{(left-start)/1000:.3f} .. +{(right-start)/1000:.3f} us "
              f"({(right-left)/1000:.3f} us): {label}")
        if correlation is None:
            continue
        api = db.execute("""
            SELECT s.value,r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id=r.nameId
            WHERE r.correlationId=? AND (r.globalTid >> 24)=(? >> 24)
            ORDER BY abs(r.start-?) LIMIT 1
        """, (correlation, pid, right)).fetchone()
        if api:
            print(f"    CPU: {api[0]}, correlation={correlation}, starts at +{(api[1]-start)/1000:.3f} us")
    print("  Host CUDA event waits and overlapping device work (ms):")
    waits = db.execute("""
        SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        WHERE s.value LIKE 'cudaEventSynchronize%' AND r.start>=? AND r.end<=? ORDER BY r.start
    """, (start, end))
    for left, right in waits:
        overlap = right-left-duration(gaps_in((left, right), activities))
        print(f"    wait={(right-left)/1e6:.3f}, device active={overlap/1e6:.3f}")
    print("  Kernel families wholly contained in the step (count, sum in ms):")
    groups = db.execute("""
        SELECT s.value, COUNT(*), SUM(k.end-k.start) AS ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY ns DESC LIMIT 5
    """, (start, end))
    for family, count, ns in groups:
        print(f"    {family}: {count}, {ns/1e6:.3f} ms")


def inspect_gap_calls(db, step):
    """Show the CPU CUDA calls intersecting the largest uncovered device interval."""
    start, end = step_window(db, step)
    gaps = gaps_in((start, end), device_activity(db, start, end))
    if not gaps:
        print("  Device activities cover the whole step.")
        return
    left, right, target, _, _ = max(gaps, key=lambda g: g[1]-g[0])
    print(f"  Largest device gap before {target}: {(left-start)/1000:.3f} .."
          f" {(right-start)/1000:.3f} us")
    calls = db.execute("""
        SELECT s.value,r.start,r.end,r.globalTid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        WHERE r.start<? AND r.end>? ORDER BY r.start,r.end
    """, (right, left))
    for name, enter, leave, tid in calls:
        overlap = min(right, leave) - max(left, enter)
        print(f"    {name}, thread {tid & 0xffffff}: {(enter-start)/1000:.3f} .."
              f" {(leave-start)/1000:.3f} us; overlaps gap by {overlap/1000:.3f} us")


def inspect_submissions(db, step):
    """Relate launches to the preceding work on their own device stream."""
    start, end = step_window(db, step)
    # Keep earlier kernels, including those launched outside the CPU step.
    # Each completed predecessor has an end timestamp on the same stream.
    query = """SELECT k.start,k.end,k.globalPid,k.deviceId,k.contextId,k.streamId,
                      k.correlationId,s.value
               FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
               WHERE k.start<? ORDER BY k.start,k.end"""
    runtime = defaultdict(list)
    for pid, corr, enter, leave in db.execute("""
            SELECT globalTid >> 24,correlationId,start,end
            FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start<?""", (end,)):
        runtime[pid, corr].append((enter, leave))
    ends = defaultdict(list)
    launches = []
    missing = ambiguous = total = 0
    for left, right, pid, device, context, stream, corr, name in db.execute(query, (end,)):
        times = ends[pid, device, context, stream]
        if start <= left and right <= end:
            total += 1
            calls = runtime[pid >> 24, corr]
            if len(calls) == 1:
                enter, leave = calls[0]
                # Count preceding kernel executions ending after this API returned.
                pending = len(times) - bisect_right(times, leave)
                launches.append((left-leave, pending, corr, enter-start, leave-start,
                                 left-start, right-start, name))
            elif calls:
                ambiguous += 1
            else:
                missing += 1
        # This exercise uses an ordinary serial stream from eager inference.
        if times and times[-1] > left:
            raise ValueError("overlapping kernels on one stream; inspect graph nodes separately")
        times.append(right)
    print(f"  Kernels wholly in step: {total}; unique runtime matches: {len(launches)};"
          f" missing: {missing}; ambiguous: {ambiguous}")
    positive = sorted(x[0] for x in launches if x[0] > 0)
    if positive:
        quantiles = [positive[ceil(q*len(positive))-1]/1000 for q in (.5, .95)]
        print(f"  Positive return-to-start intervals: {len(positive)};"
              f" nearest-rank P50={quantiles[0]:.3f}, P95={quantiles[1]:.3f} us")
    if launches:
        highest = max(launches, key=lambda row: row[0])
        lag, pending, corr, enter, leave, left, right, name = highest
        print(f"  Largest signed return-to-start: {lag/1000:.3f} us, correlation {corr};"
              f" preceding same-stream kernels unfinished at API return: {pending}")
        print(f"    API {enter/1000:.3f} .. {leave/1000:.3f} us;"
              f" GPU {left/1000:.3f} .. {right/1000:.3f} us; {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="file from nsys export --type sqlite")
    parser.add_argument("step", type=int)
    parser.add_argument("--submissions", action="store_true",
                        help="inspect API return, device start and same-stream predecessors")
    parser.add_argument("--gap-calls", action="store_true",
                        help="show CPU CUDA calls intersecting the largest device gap")
    args = parser.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        inspect(db, args.step)
        if args.submissions:
            inspect_submissions(db, args.step)
        if args.gap_calls:
            inspect_gap_calls(db, args.step)
