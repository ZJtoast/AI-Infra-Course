"""Read one vLLM engine step from an Nsight Systems SQLite export (one GPU)."""
import argparse
import sqlite3
from pathlib import Path


def inspect(db, step):
    name = f"course/step/{step}"
    windows = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text=?", (name,)).fetchall()
    if len(windows) != 1:
        raise ValueError(f"expected one {name} NVTX range, found {len(windows)}")
    start, end = windows[0]
    activities = db.execute("""
        SELECT k.start,k.end,s.value,k.correlationId,k.globalPid
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        WHERE k.end>? AND k.start<?
        UNION ALL
        SELECT start,end,CASE copyKind WHEN 1 THEN 'H2D' WHEN 2 THEN 'D2H'
                      WHEN 8 THEN 'D2D' ELSE 'memcpy' END,NULL,globalPid
        FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE end>? AND start<?
        ORDER BY 1,2
    """, (start, end, start, end))
    cursor = start
    gaps = []
    for left, right, label, correlation, pid in activities:
        if left > cursor:
            gaps.append((cursor, left, label, correlation, pid))
        cursor = max(cursor, min(right, end))
    if cursor < end:
        gaps.append((cursor, end, "end of step", None, None))
    idle = sum(right-left for left, right, *_ in gaps)
    print(f"{name} window={(end-start)/1000:.3f} us; "
          f"kernel/copy union={(end-start-idle)/1000:.3f} us; gaps={idle/1000:.3f} us")
    for left, right, label, correlation, pid in sorted(gaps, key=lambda g: g[1]-g[0], reverse=True)[:3]:
        print(f"  +{(left-start)/1000:.3f} .. +{(right-start)/1000:.3f} us "
              f"({(right-left)/1000:.3f} us): {label}")
        if correlation is None:
            continue
        # Nsight Systems packs the process in the high bits of globalTid.
        api = db.execute("""
            SELECT s.value,r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id=r.nameId
            WHERE r.correlationId=? AND (r.globalTid >> 24)=(? >> 24)
            ORDER BY abs(r.start-?) LIMIT 1
        """, (correlation, pid, right)).fetchone()
        if api:
            print(f"    CPU: {api[0]}, correlation={correlation}, starts at +{(api[1]-start)/1000:.3f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="file from nsys export --type sqlite")
    parser.add_argument("step", type=int)
    args = parser.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        inspect(db, args.step)
