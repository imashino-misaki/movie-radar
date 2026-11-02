"""Persistent applicant history (data/applicants.json) and the trusted-applicant rule.

Why: ORS intake rows have no adult flag. An intake row is published only when
its applicant is *trusted*:
    >= 1 non-adult KMRB rating and 0 adult KMRB ratings in the last 12 months
    (rolling, by KMRB rating date, Asia/Seoul).
Masked individual applicants ("개인") are never trusted and are not stored.
Everything else (new/unknown applicants, mixed applicants) stays unpublished
until its own KMRB rating arrives - rating rows carry the adult flag.

The store is built once by a 12-month backfill (`scripts/collect.py
--backfill-applicants 12`) and then updated by every run's 3-month fetch.
Each applicant keeps the KMRB receipt ids it was seen with, split into
general ("g") and adult ("a"), mapped to the rating date; counts are derived
from those id sets, so seeing the same rating on many runs never double counts.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .privacy import MASK, mask_applicant

SCHEMA_VERSION = 1
TRUST_WINDOW_MONTHS = 12


def _months_back(day: dt.date, months: int) -> dt.date:
    from .kmrb import months_back
    return months_back(day, months)


def load(path: Path) -> dict:
    if not path.exists():
        return {"schema": SCHEMA_VERSION, "backfill": None, "applicants": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _receipt_key(row: dict) -> str:
    return row.get("rcv_no") or f'no:{row.get("rating_no", "")}'


def update(store: dict, general_rows: list[dict], adult_rows: list[dict]) -> dict:
    """Add KMRB rows to the store (idempotent per receipt id). Returns the store."""
    apps = store.setdefault("applicants", {})
    for rows, bucket, other in ((general_rows, "g", "a"), (adult_rows, "a", "g")):
        for r in rows:
            name = mask_applicant(r["applicant"])
            if not name or name == MASK:
                continue
            entry = apps.setdefault(name, {"g": {}, "a": {}})
            key = _receipt_key(r)
            entry[other].pop(key, None)
            entry[bucket][key] = r.get("date", "")
    return store


def summarize(store: dict, today: dt.date) -> dict[str, dict]:
    """Per applicant: rolling-window counts and first/last rating dates."""
    cutoff = _months_back(today, TRUST_WINDOW_MONTHS).isoformat()
    out = {}
    for name, entry in store.get("applicants", {}).items():
        all_dates = sorted(d for b in ("g", "a") for d in entry.get(b, {}).values() if d)
        out[name] = {
            "general_count": sum(1 for d in entry.get("g", {}).values() if d >= cutoff),
            "adult_count": sum(1 for d in entry.get("a", {}).values() if d >= cutoff),
            "first_rating": all_dates[0] if all_dates else "",
            "last_rating": all_dates[-1] if all_dates else "",
        }
    return out


def trusted(summary: dict[str, dict]) -> set[str]:
    return {name for name, s in summary.items()
            if name != MASK and s["general_count"] >= 1 and s["adult_count"] == 0}


def to_json(store: dict, summary: dict[str, dict]) -> dict:
    """Serialisable form: summary fields first, receipt-id sets kept for dedupe."""
    apps = {}
    for name in sorted(store.get("applicants", {})):
        entry = store["applicants"][name]
        apps[name] = {**summary[name],
                      "g": dict(sorted(entry.get("g", {}).items())),
                      "a": dict(sorted(entry.get("a", {}).items()))}
    return {"schema": SCHEMA_VERSION, "trust_window_months": TRUST_WINDOW_MONTHS,
            "backfill": store.get("backfill"), "applicants": apps}


def from_json(data: dict) -> dict:
    return {"schema": data.get("schema", SCHEMA_VERSION), "backfill": data.get("backfill"),
            "applicants": {n: {"g": dict(e.get("g", {})), "a": dict(e.get("a", {}))}
                           for n, e in data.get("applicants", {}).items()}}
