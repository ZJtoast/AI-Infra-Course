"""Inspect a vLLM engine step in a single-GPU course Nsight Systems SQLite export."""
import argparse
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


def inspect(db, step):
    name = f"course/step/{step}"
    windows = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text=?", (name,)).fetchall()
    if len(windows) != 1:
        raise ValueError(f"expected one {name} NVTX range, found {len(windows)}")
    start, end = windows[0]
    activities = device_activity(db, start, end)
    gaps = gaps_in((start, end), activities)
    idle = duration(gaps)
    print(f"{name} window={(end-start)/1000:.3f} us; "
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="file from nsys export --type sqlite")
    parser.add_argument("step", type=int)
    args = parser.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        inspect(db, args.step)
