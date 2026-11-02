"""Signal history, data/signals.json: when each record's signals appeared and
changed, the training history for milestone models. Updated by run.py every
run after all stages are applied; append-only (values are never rewritten,
only new changes are added).

Schema 1:
  {"schema": 1,
   "started_utc": first run with this log (signals first seen then may be older),
   "runs": [[run_utc, {source: status}], ...]   one entry per run; the map lists
            only the sources that were not "ok" ("error", "blocked", "skipped",
            "stale"), so an empty map means every source was checked,
   "records": {record id: {signal: {"first": utc, "last": utc, "value": v,
                                    "changes": [[utc, old, new], ...]}}}}
A signal is updated only in a run whose source (SOURCES) was "ok": `last` is
the last such run that saw it; a run that checked the source and found the
record without the signal records a change to null ("checked and absent").
A failed source leaves its signals untouched; the run's entry in "runs" says
it was not checked. "changes" is left out while empty. Record ids are those of
films.json; a record later folded into another keeps its own history.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = 1


def _pick(d: dict, *keys) -> dict:
    return {k: d.get(k, "") for k in keys}


# signal -> (meta.json source whose status decides whether it was checked, value)
SOURCES = {
    "rating": ("kmrb_rating", lambda s: _pick(s, "date", "grade", "rating_no")),
    "intake": ("ors_intake", lambda s: _pick(s, "date", "status", "progress")),
    # `open`: the filings in today's ORS lists and their status, so a filing's
    # 신청완료 -> 진행중 (received) -> gone (decided) steps are logged
    "ors_ad": ("ors_ad", lambda s: {**_pick(s, "first_date", "last_date", "count"),
                                    "open": s.get("open", {})}),
    "promo": ("kmrb_promo", lambda s: {**_pick(s, "first_date", "last_date"),
                                       "kinds": s.get("kinds", [])}),
    "kobis": ("kobis", lambda s: _pick(s, "open_dt", "status")),
    "kobis_detail": ("kobis_web", lambda s: _pick(s, "status")),
    "release_history": ("kobis_web", lambda s: {"open_dt": s.get("open_dt", ""),
                                                "rereleases": s.get("rereleases", [])}),
    "schedule": ("kobis_schedule", lambda s: _pick(s, "date", "month")),
    "megabox": ("megabox", lambda s: _pick(s, "status", "release_date", "exclusive")),
    "lotte": ("lotte", lambda s: _pick(s, "status", "release_date", "release_month")),
    "cgv": ("cgv", lambda s: _pick(s, "status", "release_date", "release_month")),
    "program": ("megabox_program", lambda s: _pick(s, "chain", "program", "month", "date")),
    # presence is the signal (tickets on sale); shares and counts change every run
    "booking": ("kobis_booking", lambda s: _pick(s, "open_dt", "rerelease_dt")),
    "boxoffice": ("kobis_boxoffice", lambda s: {"first_date": (s.get("series") or [[s.get("date", "")]])[0][0]}),
}


def load(path: Path) -> dict:
    if not path.exists():
        return {"schema": SCHEMA, "started_utc": "", "runs": [], "records": {}}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"signals.json schema {doc.get('schema')!r}")
    return doc


def update(doc: dict, films: list[dict], sources: dict, run_utc: str) -> dict:
    """Add this run (in place). `sources`: meta.json's {name: {"status": ...}}.
    Returns counts for meta.json."""
    status = {name: (sources.get(src) or {}).get("status", "skipped")
              for name, (src, _) in SOURCES.items()}
    doc["started_utc"] = doc.get("started_utc") or run_utc
    failed = {src: (sources.get(src) or {}).get("status", "skipped")
              for src in sorted({s for s, _ in SOURCES.values()})}
    doc["runs"].append([run_utc, {k: v for k, v in failed.items() if v != "ok"}])
    records = doc["records"]
    counts = {"new_signals": 0, "changes": 0, "absent": 0}
    for f in films:
        log = records.get(f["id"], {})
        for name, (_, value_of) in SOURCES.items():
            if status[name] != "ok":
                continue
            stage = f.get("stages", {}).get(name)
            entry = log.get(name)
            if isinstance(stage, dict) and stage:
                value = value_of(stage)
                if entry is None:
                    log[name] = {"first": run_utc, "last": run_utc, "value": value}
                    counts["new_signals"] += 1
                    continue
                if entry["value"] != value:
                    entry.setdefault("changes", []).append([run_utc, entry["value"], value])
                    entry["value"] = value
                    counts["changes"] += 1
                entry["last"] = run_utc
            elif entry is not None and entry["value"] is not None:
                entry.setdefault("changes", []).append([run_utc, entry["value"], None])
                entry["value"] = None
                counts["absent"] += 1
        if log:
            records[f["id"]] = log
    counts["records"] = len(records)
    counts["runs"] = len(doc["runs"])
    return counts


def write(path: Path, doc: dict) -> None:
    """Compact JSON, one record per line (small diffs between runs)."""
    def dump(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    lines = ["{" + f'"schema":{SCHEMA},"started_utc":{dump(doc["started_utc"])},"runs":[']
    lines += [dump(r) + ("," if i < len(doc["runs"]) - 1 else "") for i, r in enumerate(doc["runs"])]
    lines.append('],"records":{')
    ids = sorted(doc["records"])
    lines += [f"{dump(i)}:{dump(doc['records'][i])}" + ("," if n < len(ids) - 1 else "")
              for n, i in enumerate(ids)]
    lines.append("}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
