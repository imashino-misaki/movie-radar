"""KMRB trailer and poster ratings (예고편 / 영화광고선전물): the earliest public
trace of a film being marketed, and the only official trace of a same-version
re-release (which needs no new film rating, but its trailer and posters do).

List: the same KMRB grade list as kmrb.py, category=AD, media codes
17503,17504 (국내/국외 영화예고편) and 17506 (영화광고선전물), custom date
window of WINDOW_DAYS (checked 2026-10-31: 7 days = 7 trailer pages + 20 poster
pages of 20 rows). The AD list has no adult flag and is full of adult-video
promo material, so a row is used only if its applicant is trusted (same rule
as ORS intake rows; see applicants.py). Withheld rows are counted, never stored.

Re-release check: for each new trusted title, one KMRB title search over the
whole film-rating history. A film rating of the same title dated more than
PRIOR_DAYS before the first promo filing marks a re-release candidate, unless
  * the record's KOBIS first-release date (open_dt) is later than that cutoff:
    KOBIS keeps a film's first opening date on re-release, so a recent or
    future open_dt means a first release that was rated long ago, or
  * the title also has a recent film rating that is not the same film as the
    old one (see same_film): two different films sharing a title.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import re

from . import kmrb
from .http import BlockedError, FetchError
from .privacy import MASK, mask_applicant
from .text import TitleIndex, is_truncated, norm_key
from .text import same_title as _same_title

AD_POST_URL = kmrb.LIST_URL + "?category=AD&mi=1092"
TRAILER_CODES = "17503,17504"
POSTER_CODES = "17506"
WINDOW_DAYS = 7      # runs are twice a day; a week of overlap tolerates missed runs
# One-time catch-up window, read once when promo.json has no "backfill" mark:
# the 3 months the KMRB film-rating list covers (kmrb.py). The first runs read
# only WINDOW_DAYS, so filings decided before them were never seen (천년여우's
# poster, decided 2026-10-16, before the first run on 2026-10-31). Groups
# older than WINDOW_DAYS only join existing records: no promo-only records
# for material of films nobody tracks.
BACKFILL_DAYS = 92
PRIOR_DAYS = 365     # an older film rating marks a re-release
HISTORY_CAP = 30     # title-history searches per run


def _trunc(title: str) -> bool:
    return is_truncated(title)


def title_key(title: str) -> str:
    return norm_key(title.rstrip().rstrip(".…"))


def fetch_window(client, medi: str, start: dt.date, end: dt.date) -> list[dict]:
    form = {"pageIndex": kmrb.PAGE_SIZE, "mediCode": medi, "advancedOpen": "N",
            "searchDate": "", "searchDateStart": start.isoformat(),
            "searchDateEnd": end.isoformat(), "adult_yn": "N", "useTitle": ""}
    rows: list[dict] = []
    page = 1
    while True:
        resp = client.post(AD_POST_URL, data={**form, "currPage": page},
                           headers={"Referer": kmrb.LIST_URL})
        got, pages, _ = kmrb.parse_page(resp.text)
        if not got and pages:
            raise kmrb.ParseError(f"KMRB AD {medi}: page {page}/{pages} had no rows")
        rows += got
        if page >= pages:
            break
        page += 1
    return rows


# The list drops all rows when the search term contains markup characters
# (checked 2026-10-31: a title with "<...>" returns an empty page without the
# list markup), so only the text before the first such character is searched.
_UNSAFE = re.compile(r"[<>\"'&]")
# rows read per title search: at least the 60 rows (3 pages of 20) chosen when
# the list was read 20 rows a page; with kmrb.PAGE_SIZE rows a page
HISTORY_ROWS = 60


def search_term(title: str) -> str:
    return _UNSAFE.split(title.rstrip().rstrip(".…"), maxsplit=1)[0].strip()


same_title = _same_title


# Same-film test for an old and a recent rating of the same title. A re-release
# keeps the cut, so the runtime barely moves (checked 2026-10-31: 트레인스포팅
# 90:57 -> 93:49, +3%; 스트레인저 무황인담 103:00 -> 102:42), while two films
# that share a title differ by far more (레이오버 호텔 2021 81:58 vs 2026
# 102:07, +25%).
RUNTIME_TOLERANCE = 0.10
_RUNTIME = re.compile(r"(\d+)\s*분(?:\s*(\d+)\s*초)?")


def runtime_s(text: str) -> int:
    m = _RUNTIME.search(text or "")
    return int(m.group(1)) * 60 + int(m.group(2) or 0) if m else 0


def same_film(a: dict, b: dict) -> bool:
    """Same production country and (same original title or near-equal runtime)."""
    if not a.get("country") or a.get("country") != b.get("country"):
        return False
    oa, ob = a.get("original_title", ""), b.get("original_title", "")
    if oa and ob and same_title(oa, ob):
        return True
    ra, rb = runtime_s(a.get("runtime", "")), runtime_s(b.get("runtime", ""))
    return bool(ra and rb and abs(ra - rb) <= RUNTIME_TOLERANCE * max(ra, rb))


def _cutoff(first_date: str) -> str:
    return (dt.date.fromisoformat(first_date) - dt.timedelta(days=PRIOR_DAYS)).isoformat()


def prior_ratings(ratings: list[dict], first_date: str) -> list[str]:
    """Dates of film ratings older than PRIOR_DAYS before `first_date`; when the
    title also has a rating inside that period, only old ratings of the same
    film count."""
    cutoff = _cutoff(first_date)
    recent = [r for r in ratings if r["date"] >= cutoff]
    old = [r for r in ratings if r["date"] < cutoff]
    if recent:
        old = [o for o in old if any(same_film(o, r) for r in recent)]
    return sorted({o["date"] for o in old})


def first_release_after(rec: dict, first_date: str) -> bool:
    """KOBIS says the film first opens (or opened) after the re-release cutoff."""
    open_dt = rec["stages"].get("kobis", {}).get("open_dt", "")
    return bool(open_dt) and open_dt >= _cutoff(first_date)


def search_history(client, title: str, today: dt.date) -> list[dict]:
    """Film ratings (all time) of films with the same title: date, grade,
    rating number, original title, runtime, country."""
    form = {"pageIndex": kmrb.PAGE_SIZE, "mediCode": kmrb.MEDIA_CODES, "advancedOpen": "N",
            "searchDate": "", "searchDateStart": "1990-01-01",
            "searchDateEnd": today.isoformat(), "adult_yn": "N", "useTitle": search_term(title)}
    found: dict[str, dict] = {}
    for page in range(1, math.ceil(HISTORY_ROWS / kmrb.PAGE_SIZE) + 1):
        resp = client.post(kmrb.POST_URL, data={**form, "currPage": page},
                           headers={"Referer": kmrb.LIST_URL})
        rows, pages, _ = kmrb.parse_page(resp.text)
        for r in rows:
            if r["date"] and same_title(r["title"], title):
                found[r["rating_no"] or r["date"]] = {
                    k: r[k] for k in ("date", "grade", "rating_no", "original_title", "runtime",
                                      "country")}
        if page >= pages or not rows:
            break
    return sorted(found.values(), key=lambda r: (r["date"], r["runtime"]))


def group_rows(rows: list[dict], trusted: set[str]) -> tuple[dict, int]:
    """Group trusted rows by (title key, masked applicant). Truncated titles
    ("크리스마스에 기적을...") join a full title of the same applicant that starts
    with the same text. Returns ({group_key: group}, withheld_count)."""
    withheld = 0
    kept = []
    for r in rows:
        applicant = mask_applicant(r["applicant"])
        if applicant == MASK or applicant not in trusted:
            withheld += 1
            continue
        kept.append({**r, "applicant": applicant})
    full = {(title_key(r["title"]), r["applicant"]) for r in kept if not _trunc(r["title"])}
    groups: dict[str, dict] = {}
    for r in sorted(kept, key=lambda r: _trunc(r["title"])):  # full titles first
        tk = title_key(r["title"])
        if _trunc(r["title"]):
            tk = next((k for k, a in sorted(full) if a == r["applicant"] and k.startswith(tk)), tk)
        gk = f"{tk}|{r['applicant']}"
        g = groups.setdefault(gk, {"title_key": tk, "title": r["title"], "applicant": r["applicant"],
                                   "original_title": "", "dates": [], "kinds": set(), "count": 0})
        if _trunc(g["title"]) and not _trunc(r["title"]):
            g["title"] = r["title"]
        if r.get("original_title") and (not g["original_title"] or _trunc(g["original_title"])):
            g["original_title"] = r["original_title"]
        g["dates"].append(r["date"])
        g["kinds"].add("trailer" if "예고편" in r.get("kind", "") else "poster")
        g["count"] += 1
    return groups, withheld


def fetch(client, cache: dict, today: dt.date, trusted: set[str], stamp: str,
          window_days: int = WINDOW_DAYS) -> dict:
    """Fetch the window, group, run history searches for new titles. Updates
    `cache` (group key -> {"ratings": [...], "checked": date}) in place; entries
    for groups no longer in the window are dropped."""
    client.get(kmrb.LIST_URL)  # session cookie
    start = today - dt.timedelta(days=window_days)
    trailers = fetch_window(client, TRAILER_CODES, start, today)
    posters = fetch_window(client, POSTER_CODES, start, today)
    if not trailers and not posters:
        # an empty week would prune the whole cache: treat it as a failed fetch
        raise FetchError("KMRB AD: no trailer or poster ratings in the window")
    groups, withheld = group_rows(trailers + posters, trusted)
    searched = errors = 0
    for gk, g in sorted(groups.items()):
        if "ratings" in cache.get(gk, {}) or searched >= HISTORY_CAP:
            continue
        searched += 1
        try:
            cache[gk] = {"ratings": search_history(client, g["title"], today), "checked": stamp}
        except BlockedError:
            raise
        except Exception:  # noqa: BLE001 - one bad title search must not fail the source
            errors += 1  # not cached: retried next run
    for gk in [k for k in cache if k not in groups]:
        del cache[gk]  # out of the window: its prior ratings are already on the film record
    return {"groups": groups, "stats": {
        "trailer_rows": len(trailers), "poster_rows": len(posters),
        "withheld_untrusted": withheld, "groups": len(groups), "history_searched": searched,
        "history_errors": errors,
    }}


def promo_id(group_key: str) -> str:
    return "p" + hashlib.sha1(group_key.encode()).hexdigest()[:12]


def merge_stage(a: dict | None, b: dict | None) -> dict:
    """Union of two promo stages (used when two records are folded together)."""
    if not a or not b:
        return dict(a or b or {})
    dates = sorted(set(a.get("dates_seen", [])) | set(b.get("dates_seen", [])))
    prior = sorted(set(a.get("prior_ratings", [])) | set(b.get("prior_ratings", [])))
    return {"first_date": dates[0] if dates else "", "last_date": dates[-1] if dates else "",
            "kinds": sorted(set(a.get("kinds", [])) | set(b.get("kinds", []))),
            "dates_seen": dates, "prior_ratings": prior, "rerelease": bool(prior)}


def _latest_activity(rec: dict) -> str:
    st = rec["stages"]
    return max([st.get("rating", {}).get("date", ""), st.get("intake", {}).get("date", ""),
                st.get("kobis", {}).get("open_dt", ""), st.get("megabox", {}).get("release_date", ""),
                st.get("promo", {}).get("last_date", "")] or [""])


def pick_record(cands: list[dict], applicant: str, first_date: str) -> dict | None:
    """The record a promo group joins. Same applicant first. A record of another
    applicant (the film's producer often holds the rating, its distributor files
    the trailer) is joined only if it shows activity within PRIOR_DAYS of the
    filing, so an old, unrelated film that shares the title is left alone."""
    same = [f for f in cands if f.get("applicant") == applicant]
    pool = same or [f for f in cands if _latest_activity(f) >= _cutoff(first_date)]
    if not pool:
        return None
    return max(pool, key=lambda f: (max(f["first_seen"].values(), default=""), f["id"]))


def apply_promo(films: list[dict], groups: dict, cache: dict, stamp: str,
                new_since: str = "") -> tuple[list[dict], dict]:
    """Attach stage "promo" (pure, no I/O). A group joins an existing record with
    the same normalised title (see pick_record); otherwise it becomes a
    promo-only record, unless its last filing is older than `new_since` (a
    backfill: see BACKFILL_DAYS). Dates accumulate across runs. Tags are set by
    finalize() for every record, so they survive runs without promo data."""
    from .merge import _sort, fold_into, upgrade_title

    stats = {"promo_matched": 0, "promo_only": 0, "promo_rerelease": 0}
    if new_since:
        stats["promo_old_unmatched"] = 0
    by_id = {f["id"]: f for f in films}
    index = TitleIndex(films, skip=lambda f: f["id"][:1] in ("m", "l"),
                       wanted=[g["title"] for g in groups.values()])

    for gk, g in sorted(groups.items()):
        pid = promo_id(gk)
        first_new = min(g["dates"])
        cands = [f for f in index.find(g["title"]) if f["id"] != pid and f["id"] in by_id]
        rec = pick_record(cands, g["applicant"], first_new)
        if rec is not None:
            upgrade_title(rec, g["title"], g["original_title"])
            old = by_id.pop(pid, None)
            if old:  # a promo-only record found its film record: carry everything over
                fold_into(rec, old)
            stats["promo_matched"] += 1
        elif pid not in by_id and new_since and max(g["dates"]) < new_since:
            stats["promo_old_unmatched"] += 1  # backfilled material of an untracked film
            continue
        else:
            rec = by_id.get(pid) or {"id": pid, "rcv_no": "", "title": g["title"],
                                     "original_title": g["original_title"],
                                     "applicant": g["applicant"], "stages": {}, "first_seen": {},
                                     "tags": []}
            by_id[pid] = rec
            stats["promo_only"] += 1
        prev = rec["stages"].get("promo") or {}
        dates = sorted(set(prev.get("dates_seen", [])) | set(g["dates"]))
        kinds = sorted(set(prev.get("kinds", [])) | g["kinds"])
        first = dates[0]
        entry = cache.get(gk, {})
        prior = (sorted(set(prior_ratings(entry["ratings"], first)) | set(prev.get("prior_ratings", [])))
                 if "ratings" in entry else prev.get("prior_ratings", []))
        rerelease = bool(prior) and not first_release_after(rec, first)
        rec["stages"]["promo"] = {"first_date": first, "last_date": dates[-1], "kinds": kinds,
                                  "dates_seen": dates, "prior_ratings": prior,
                                  "rerelease": rerelease}
        rec["first_seen"]["promo"] = min(stamp, rec["first_seen"].get("promo", stamp))
        rec["last_seen"] = stamp
        stats["promo_rerelease"] += rerelease
    return _sort(by_id.values()), stats


def chain_release_date(rec: dict) -> str:
    """The earliest release date a cinema chain lists for the record."""
    dates = [rec["stages"].get(c, {}).get("release_date", "") for c in ("megabox", "lotte", "cgv")]
    return min((d for d in dates if d), default="")


def kobis_open_date(rec: dict) -> str:
    """The film's first opening date per KOBIS (KOBIS keeps it on re-releases)."""
    st = rec["stages"]
    return (st.get("release_history", {}).get("open_dt", "") or st.get("kobis", {}).get("open_dt", "")
            or st.get("lotte", {}).get("kobis_open_dt", "") or st.get("booking", {}).get("open_dt", ""))


def chain_rerelease(rec: dict) -> bool:
    """A chain lists the film for a release after its first opening.

    With a KOBIS first opening date: the chain's release date is later (for
    first releases the two are equal: 48 of 48 records with both on
    2026-11-01). Without one: the chain's release date is more than PRIOR_DAYS
    after the film's past KMRB rating (late first releases 112-974 days after
    their rating were seen, so this fallback misses short-gap re-releases).
    KOBIS's 실시간 예매율 table also labels re-released films with a 재개봉일,
    and CGV's list labels its re-releases (rrlsYn)."""
    if rec["stages"].get("booking", {}).get("rerelease_dt"):
        return True  # KOBIS labels the film on sale with a 재개봉일
    if rec["stages"].get("cgv", {}).get("rerelease"):
        return True  # CGV lists the film as a re-release (rrlsYn)
    release = chain_release_date(rec)
    if not release:
        return False
    opened = kobis_open_date(rec)
    if opened:
        return opened < release
    past = rec["stages"].get("past_rating", {}).get("date", "")
    return bool(past) and past < _cutoff(release)


def _keyword_rerelease(rec: dict) -> bool:
    from .tags import tag_film

    return "rerelease" in tag_film(rec.get("title", ""), rec.get("original_title", ""),
                                   rec.get("applicant", ""),
                                   rec["stages"].get("rating", {}).get("country", ""))


def finalize(films: list[dict]) -> None:
    """Recompute the re-release flag and tag of every record (in place). Runs
    every collection, also when a fetch failed: the tags are rebuilt from
    scratch by merge(), and KOBIS can add a recent first-release date after a
    promo stage was written. A record is tagged "rerelease" for its promo
    stage, a chain listing (chain_rerelease) or a title keyword."""
    for rec in films:
        st = rec["stages"].get("promo")
        if st:
            st["rerelease"] = (bool(st.get("prior_ratings"))
                               and not first_release_after(rec, st["first_date"]))
        tags = set(rec.get("tags", [])) - {"rerelease"}
        if (st and st["rerelease"]) or chain_rerelease(rec) or _keyword_rerelease(rec):
            tags.add("rerelease")
        rec["tags"] = sorted(tags)
    _share_group_tags(films)


# tags that describe the film, so every per-format record of it shares them;
# chain_* (who applied for that format's rating) and format stay per record
GROUP_TAGS = {"rerelease", "anime", "concert"}


def group_view(lead: dict, by_id: dict) -> tuple[dict, list[dict]]:
    """A film rated per screening format (merge.group_variants) as one record:
    the lead's stages, with the stages only its per-format records have
    (a rating, a KOBIS entry, a chain listing). Returns (view, members)."""
    members = [lead] + [by_id[m["id"]] for m in lead.get("formats", [])
                        if m["id"] != lead["id"] and m["id"] in by_id]
    stages: dict = {}
    for m in reversed(members):  # the lead's own stages win
        stages.update(m["stages"])
    return {**lead, "stages": stages}, members


def _share_group_tags(films: list[dict]) -> None:
    """Re-release and genre tags are decided per film, not per format: a
    member's tag applies to the whole group, and the re-release rule also runs
    on the group's combined stages (a chain listing on one record, the KOBIS
    first opening date on another)."""
    by_id = {f["id"]: f for f in films}
    for lead in films:
        if not lead.get("formats"):
            continue
        view, members = group_view(lead, by_id)
        shared = set().union(*(set(m["tags"]) & GROUP_TAGS for m in members))
        if chain_rerelease(view):
            shared.add("rerelease")
        for m in members:
            m["tags"] = sorted(set(m["tags"]) | shared)
