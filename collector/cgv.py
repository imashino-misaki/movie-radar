"""CGV movie lists (now showing + upcoming), from cgv.json.

The collector does not fetch CGV itself. cgv.json (CGV's movie list) is read
from the repository's cgv-data branch when present: the workflow copies it to
data/cgv_upload.json (git-ignored) before the collector runs, and this module
reads it. No file means the source is skipped, not failed.

The file is applied with its own observation date (the UTC date of
`fetched_at`), so first_seen / runs / ended reflect when CGV listed a film,
not when the collector ran. A file older than STALE_CYCLES of the update
cycles it reports is still applied, and the source is marked "stale" in
meta.json.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .http import FetchError
from .merge import TitleIndex, _sort, ended_stage, fold_into, set_bookable, track_runs, upgrade_title
from .tags import tag_film

UPLOAD_NAME = "cgv_upload.json"
SCHEMA = 1
# Two cycles: a current file is at most one cycle old; older than two means
# at least one whole cycle's update is missing.
STALE_CYCLES = 2
STAGE_KEYS = ("movie_no", "status", "release_date", "release_month", "first_release_date",
              "rerelease", "grade", "poster", "booking_rate", "booking_rank", "bookable",
              "formats", "labels", "badge")


def cgv_id(movie_no: str) -> str:
    return "c" + movie_no


def stale_after_minutes(doc: dict) -> float:
    """Derived from the interval_minutes and jitter fields of cgv.json. Both are
    required: the collector has no defaults of its own to fall back on."""
    interval, jitter = doc.get("interval_minutes"), doc.get("jitter")
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (interval, jitter))
    if not numbers or interval <= 0 or not 0 <= jitter < 1:
        raise FetchError(f"cgv upload: interval_minutes {interval!r} / jitter {jitter!r} "
                         "missing or out of range")
    return STALE_CYCLES * interval * (1 + jitter)


def parse_time(value: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("fetched_at without a time zone")
    return t.astimezone(dt.timezone.utc)


def read_upload(path: Path, now: dt.datetime) -> dict | None:
    """{"rows", "seen", "stats"}; None when there is no upload. Raises
    FetchError on a file that is not a valid upload."""
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise ValueError(f"schema {doc.get('schema') if isinstance(doc, dict) else None}")
        fetched = parse_time(doc["fetched_at"])
        movies = doc["movies"]
        if not isinstance(movies, list) or not movies:
            raise ValueError("no movies")  # an empty list would end every CGV stage
        rows = []
        for m in movies:
            row = {k: m.get(k) for k in STAGE_KEYS}
            row["title"] = " ".join(str(m.get("title") or "").split())
            row["movie_no"] = str(m.get("movie_no") or "")
            if not row["movie_no"] or not row["title"] or row["status"] not in ("now", "upcoming"):
                raise ValueError(f"bad movie row {row['movie_no']!r}")
            for k in ("release_date", "release_month", "first_release_date", "grade", "poster",
                      "badge"):
                row[k] = str(row[k] or "")
            row["formats"] = list(row["formats"] or [])
            row["labels"] = list(row["labels"] or [])
            row["rerelease"] = bool(row["rerelease"])
            row["bookable"] = bool(row["bookable"])
            rows.append(row)
    except (ValueError, KeyError, TypeError) as exc:
        raise FetchError(f"cgv upload: {type(exc).__name__}: {exc}") from None
    seen_ids: set[str] = set()
    rows = [r for r in rows if not (r["movie_no"] in seen_ids or seen_ids.add(r["movie_no"]))]
    age = (now.astimezone(dt.timezone.utc) - fetched).total_seconds() / 60
    limit = stale_after_minutes(doc)
    return {"rows": rows, "seen": fetched.date().isoformat(), "stats": {
        "rows": len(rows),
        "now": sum(r["status"] == "now" for r in rows),
        "upcoming": sum(r["status"] == "upcoming" for r in rows),
        "fetched_at": fetched.isoformat().replace("+00:00", "Z"),
        "age_minutes": round(age),
        "stale_after_minutes": round(limit),
        "stale": age > limit,
    }}


def apply_cgv(films: list[dict], rows: list[dict], seen: str) -> tuple[list[dict], dict]:
    """Attach the CGV lists (pure, no I/O). `seen`: the upload's UTC date.

    A row joins a record whose normalised title matches (as for Megabox and
    Lotte; CGV gives no KOBIS code), Megabox- and Lotte-only records included.
    Unmatched rows become CGV-only records (id "c" + CGV movie number), folded
    into another record once one matches. Records that held a CGV stage but
    are not in this upload are kept with status "ended"; `runs` keeps every
    listing period. CGV-only records are published without the
    trusted-applicant rule, as Megabox- and Lotte-only ones are.
    """
    stats = {"cgv_matched": 0, "cgv_only": 0, "cgv_ended": 0}
    by_id = {f["id"]: f for f in films}
    index = TitleIndex(films, skip=lambda f: f["id"].startswith("c"),
                       wanted=[r["title"] for r in rows])

    def pick(cands: list[dict], release: str) -> dict:
        exact = [f for f in cands if release and f["stages"].get("kobis", {}).get("open_dt") == release]
        return max(exact or cands, key=lambda f: (max(f["first_seen"].values(), default=""), f["id"]))

    assigned: set[str] = set()
    for r in sorted(rows, key=lambda r: r["movie_no"]):
        stage = {k: r[k] for k in STAGE_KEYS}
        cid = cgv_id(r["movie_no"])
        first = seen
        # a re-release joins on its original opening date, a first release on its date
        cands = [f for f in index.find(r["title"]) if f["id"] in by_id]
        if cands:
            rec = pick(cands, r["first_release_date"] or r["release_date"])
            upgrade_title(rec, r["title"])
            old = by_id.pop(cid, None)
            if old:  # a CGV-only record found its record: fold it in
                first = min(first, old["first_seen"].get("cgv", first))
                fold_into(rec, old)
            stats["cgv_matched"] += 1
        else:
            rec = by_id.get(cid) or {"id": cid, "rcv_no": "", "title": r["title"],
                                     "original_title": "", "applicant": "", "stages": {},
                                     "first_seen": {}, "tags": []}
            rec["tags"] = tag_film(rec["title"], "", "", "")
            by_id[cid] = rec
            stats["cgv_only"] += 1
        rec["first_seen"]["cgv"] = min(first, rec["first_seen"].get("cgv", first))
        old_stage = rec["stages"].get("cgv")
        stage["runs"] = track_runs(old_stage, seen, stage["release_date"])
        set_bookable(stage, old_stage, seen)
        rec["stages"]["cgv"] = stage
        rec["last_seen"] = max(rec.get("last_seen", ""), seen)
        assigned.add(rec["id"])

    for rec in by_id.values():
        st = rec["stages"].get("cgv")
        if st and rec["id"] not in assigned and st.get("status") != "ended":
            rec["stages"]["cgv"] = ended_stage(st)
            stats["cgv_ended"] += 1
    return _sort(by_id.values()), stats
