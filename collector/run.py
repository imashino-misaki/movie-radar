"""Collector entry point: fetch the sources, merge, write data/.

Each source (ORS intake, KMRB ratings, KMRB trailer/poster ratings, KOBIS, Megabox list,
ORS trailer/advertising filings, Megabox MEGA ONLY, ...) is fetched independently. A source
that fails - HTTP 403/429, a block notice in the body, KOBIS faultInfo, or a
page missing its expected markup - is recorded in meta.json as "blocked" or
"error" and contributes nothing this run: its last good data stays as it is
and nothing is deleted because of it. The run exits non-zero (and writes
nothing) only when every attempted source failed. KOBIS without an API key is
"skipped", not failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

from . import (applicants, boxoffice, cgv, companies, distributors, kobis, kmrb, kobis_web, lists,
               lookup, lotte, megabox, ors, posters, promo, release, theatrical)
from . import ors_ad, programs, signals, site
from .http import BlockedError, PoliteClient
from .merge import (apply_kobis, apply_lotte, apply_megabox, group_variants, merge,
                    restricted_withheld)
from .privacy import MASK, mask_applicant

log = logging.getLogger("collector")

KST = dt.timezone(dt.timedelta(hours=9))
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCHEMA_VERSION = 1


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_films(path: Path) -> list[dict]:
    data = load_json(path, [])
    return data["films"] if isinstance(data, dict) else data


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_source(name: str, fn, key: str = ""):
    """Run one source fetch. Returns (result or None, status dict)."""
    try:
        return fn(), {"status": "ok"}
    except BlockedError as exc:
        status = {"status": "blocked", "reason": kobis.redact(str(exc), key)[:200]}
    except Exception as exc:  # noqa: BLE001 - one source failing must not stop the others
        status = {"status": "error",
                  "reason": kobis.redact(f"{type(exc).__name__}: {exc}", key)[:200]}
    log.warning("source %s failed: %s", name, status["reason"])
    return None, status


class AllSourcesFailed(RuntimeError):
    pass


def collect(client, data_dir: Path, now: dt.datetime, kobis_key: str | None = None) -> dict:
    """Fetch, merge and write. Raises AllSourcesFailed (writing nothing) if every
    attempted source failed."""
    today_kst = now.astimezone(KST).date()
    # calendar-day stamps (first_seen, last_seen, runs, bookable_from, cache "checked"
    # days) are the Korean day: the runs at 15:00-24:00 UTC (collect.yml, every 2 hours)
    # fall on the next KST day. Days stored before 2026-11-03 are UTC days, never later
    # than the KST day of a later run, so the switch only extends periods. Run times
    # (last_run_utc, last_ok_utc, signals) stay UTC.
    today = today_kst.isoformat()
    key = kobis.get_key() if kobis_key is None else kobis_key

    def timed(name: str, fn, key: str = ""):
        """run_source, plus the source's wall time and request count in its status."""
        t0, n0 = time.monotonic(), getattr(client, "request_count", 0)
        res, status = run_source(name, fn, key)
        status["seconds"] = round(time.monotonic() - t0, 1)
        status["requests"] = getattr(client, "request_count", 0) - n0
        log.info("source %s: %s, %ss, %s requests", name, status["status"],
                 status["seconds"], status["requests"])
        return res, status

    prev_meta = load_json(data_dir / "meta.json", {})
    prev_sources = prev_meta.get("sources", {})
    run_utc = now.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()
    # the run time with the KST offset: first_seen.ors_ad, whose first 10 characters
    # are read as the day (ors_ad.apply, site.day)
    run_kst = now.astimezone(KST).replace(microsecond=0).isoformat()

    intake, s_ors = timed("ors_intake", lambda: ors.fetch_intake(client))
    # trailer / advertising filings in progress (join records, or create them: ors_ad.py)
    adres, s_ors_ad = timed("ors_ad", lambda: ors_ad.fetch(client))
    ratings, s_kmrb = timed("kmrb_rating", lambda: kmrb.fetch_ratings(client, today_kst))

    apps_path = data_dir / "applicants.json"
    store = applicants.from_json(applicants.load(apps_path))
    if ratings:
        applicants.update(store, ratings["general"], ratings["adult"])
    summary = applicants.summarize(store, today_kst)
    trusted = applicants.trusted(summary)

    # representatives of every rated applicant, for the manual lists only
    if ratings:
        reps_res, s_reps = timed("kmrb_reps", lambda: kmrb.fetch_reps(
            client, today_kst, ratings["stats"]["all_declared"]))
    else:
        reps_res, s_reps = None, {"status": "skipped", "reason": "no rating list"}
    manual = lists.load()

    # Lotte first: the KOBIS codes it lists get KOBIS info (first opening date)
    lres_list, s_lotte = timed("lotte", lambda: lotte.fetch_list(client))
    lotte_codes = tuple(r["kobis_cd"] for r in (lres_list or {}).get("rows", []))
    # films KOBIS lists as 개봉예정 / 개봉준비, also those without an opening date
    ures, s_upcoming = timed("kobis_upcoming", lambda: kobis_web.fetch_upcoming(client))
    # the 개봉스케줄 calendar: planned dates (or months) and posters
    sres, s_schedule = timed("kobis_schedule",
                                  lambda: kobis_web.fetch_schedule(client, today_kst))
    # 실시간 예매율: advance-ticket share and audience counts of films on sale
    bres, s_booking = timed("kobis_booking", lambda: kobis_web.fetch_booking(client))
    undated = tuple(dict.fromkeys(
        r["movie_cd"] for r in (ures or {}).get("rows", []) + (sres or {}).get("rows", [])
        if kobis.ADULT_GENRE not in r["genres"]))

    kobis_path = data_dir / "kobis.json"
    kobis_data = load_json(kobis_path, {"schema": SCHEMA_VERSION, "movies": {}})
    cache = {cd: dict(e) for cd, e in kobis_data.get("movies", {}).items()}

    # applicants and distributors KMRB has no record of: the KOBIS company register
    comp_path = data_dir / "companies.json"
    comp_cache = {n: dict(e) for n, e in load_json(comp_path, {}).get("companies", {}).items()}
    comp_new = {n: dict(e) for n, e in comp_cache.items()}
    known = {n for n, e in summary.items() if e["general_count"] or e["adult_count"]}
    names = [mask_applicant(r["applicant"]) for r in (intake or {}).get("rows", [])]
    names += [d for e in cache.values() for d in e.get("distributors", [])]
    if not intake:  # keep the applicants of the last good intake list
        names += list(comp_cache)
    cres, s_companies = timed("kobis_companies", lambda: companies.check(
        client, names, known, comp_new, today))
    if cres:
        comp_cache = comp_new
    company_trusted = companies.trusted_names(comp_cache)
    trusted = set(trusted) | company_trusted

    # manual block / allow lists: a block beats every rule, an allow trusts
    rep_map: dict[str, set[str]] = {}
    for found in ((intake or {}).get("reps", {}), (reps_res or {}).get("reps", {})):
        for name, people in found.items():
            rep_map.setdefault(mask_applicant(name), set()).update(people)
    for name, entry in comp_cache.items():
        if entry.get("rep"):
            rep_map.setdefault(name, set()).add(entry["rep"])
    rated = (ratings["general"] + ratings["adult"]) if ratings else []
    observed = (set(rep_map) | set(names) | set(trusted)
                | {mask_applicant(r["applicant"]) for r in rated}) - {MASK, ""}
    decisions = {n: manual.decide(n, rep_map.get(n, ())) for n in observed}
    blocked = frozenset({companies.company_key(n) for n, d in decisions.items() if d == "block"}
                        | manual.block_companies)
    allowed = {n for n, d in decisions.items() if d == "allow"}
    trusted = {n for n in trusted if companies.company_key(n) not in blocked} | allowed
    allowed_keys = {companies.company_key(n) for n in allowed} | set(manual.allow_companies)
    # applicants with adult KMRB ratings: their ORS filings never create a record
    ad_withheld = ({companies.company_key(n) for n, s in summary.items() if s["adult_count"]}
                   - allowed_keys)

    def is_blocked(applicant: str) -> bool:
        return companies.company_key(mask_applicant(applicant)) in blocked

    general = [r for r in ratings["general"] if not is_blocked(r["applicant"])] if ratings else []
    ratings_blocked = [r for r in ratings["general"] if is_blocked(r["applicant"])] if ratings else []
    # a blocked company's general ratings are withheld like adult ones
    adult = (ratings["adult"] if ratings else []) + ratings_blocked
    withheld_audits = {r["rating_no"] for r in adult + restricted_withheld(general, trusted)
                       if r.get("rating_no")}
    intake_rows = [r for r in (intake or {}).get("rows", []) if not is_blocked(r["applicant"])]
    manual_stats = {"active": bool(manual), "blocked_names": sum(d == "block" for d in decisions.values()),
                    "allowed_names": len(allowed), "ratings_withheld": len(ratings_blocked),
                    "intake_withheld": len((intake or {}).get("rows", [])) - len(intake_rows)}

    if not key:
        kres, s_kobis = None, {"status": "skipped", "reason": "no API key configured"}
    else:
        kc = kobis.KobisClient(client, key, cap=kobis.CALL_CAP)
        kres, s_kobis = timed(
            "kobis", lambda: kobis.fetch(kc, cache, today_kst, trusted, withheld_audits,
                                         stamp=today, extra_codes=lotte_codes,
                                         undated_codes=undated, blocked=blocked), key)
        s_kobis["calls"] = kc.calls
        s_kobis["call_cap"] = kc.cap

    # daily box office: each day fetched once into a rolling history
    bo_path = data_dir / "boxoffice.json"
    bo_history = {d: list(rows) for d, rows in load_json(bo_path, {}).get("days", {}).items()}
    if not key:
        bores, s_boxoffice = None, {"status": "skipped", "reason": "no API key configured"}
    else:
        bo_new = dict(bo_history)
        bkc = kobis.KobisClient(client, key, cap=boxoffice.MAX_CALLS, base=boxoffice.BASE)
        bores, s_boxoffice = timed("kobis_boxoffice",
                                   lambda: boxoffice.fetch(bkc, bo_new, today_kst), key)
        s_boxoffice["calls"] = bkc.calls
        if bores:
            bo_history = bo_new

    promo_path = data_dir / "promo.json"
    promo_doc = load_json(promo_path, {})
    promo_cache = dict(promo_doc.get("history", {}))
    # one longer catch-up window, once (promo.BACKFILL_DAYS)
    promo_backfill = promo_doc.get("backfill") or None
    promo_days = promo.WINDOW_DAYS if promo_backfill else promo.BACKFILL_DAYS
    pres, s_promo = timed("kmrb_promo", lambda: promo.fetch(client, promo_cache, today_kst,
                                                                 trusted, today, promo_days))
    if pres and not promo_backfill:
        promo_backfill = {"date": today_kst.isoformat(), "days": promo_days}

    mres, s_megabox = timed("megabox", lambda: megabox.fetch_list(client))
    # Megabox MEGA ONLY tab and monthly lineups (programs.py)
    pgres, s_program = timed("megabox_program", lambda: programs.fetch_megabox(client))
    # CGV: cgv.json (CGV's movie list), read from the cgv-data branch when
    # present; the workflow copies it to data/cgv_upload.json (no request)
    cgv_path = data_dir / cgv.UPLOAD_NAME
    if cgv_path.exists():
        cgv_res, s_cgv = timed("cgv", lambda: cgv.read_upload(cgv_path, now))
        if cgv_res and cgv_res["stats"]["stale"]:
            s_cgv = {**s_cgv, "status": "stale", "reason": (
                f'upload {cgv_res["stats"]["age_minutes"]} min old '
                f'(stale after {cgv_res["stats"]["stale_after_minutes"]})')}
    else:
        cgv_res, s_cgv = None, {"status": "skipped", "reason": "no upload"}

    # every source whose result is applied on its own: ORS filings create records, a
    # fresh CGV upload updates the CGV stages (a stale one is "stale", not ok)
    attempted = [s for s in (s_ors, s_ors_ad, s_kmrb, s_promo, s_kobis, s_megabox, s_lotte, s_cgv)
                 if s["status"] != "skipped"]
    if all(s["status"] != "ok" for s in attempted):
        raise AllSourcesFailed("; ".join(f'{s["status"]}: {s.get("reason", "")}' for s in attempted))

    films_path = data_dir / "films.json"
    existing = load_films(films_path)
    films, stats = merge(existing, intake_rows, general, adult,
                         today, trusted)
    kstats = {}
    if kres:
        films, kstats = apply_kobis(films, cache, kres["kept_ids"], trusted, withheld_audits,
                                    today, allow_new=ratings is not None, blocked=blocked)
    pstats = {}
    if pres:  # before Megabox, so a Megabox listing can join a promo-only record
        new_since = (today_kst - dt.timedelta(days=promo.WINDOW_DAYS)).isoformat()
        films, pstats = promo.apply_promo(films, pres["groups"], promo_cache, today, new_since)
    # records created from ORS filings join the film's record once it exists
    # (every run, before the chains: a listing then joins the film's record)
    films, settle_stats = ors_ad.settle(films, ad_withheld)
    mstats = {}
    if mres:
        films, mstats = apply_megabox(films, mres["rows"], today)
    ltstats = {}
    if lres_list:  # after Megabox: a Lotte row joins a Megabox-only record by title
        films, ltstats = apply_lotte(films, lres_list["rows"], today, cache)
    cstats = {}
    if cgv_res:  # after Lotte: a CGV row joins any chain's record by title
        films, cstats = cgv.apply_cgv(films, cgv_res["rows"], cgv_res["seen"])
    # after every chain: ORS filings join records or create them (ors_ad.py);
    # programmes only join records
    adstats = {}
    if adres:
        ad_rows = [r for r in adres["rows"] if not is_blocked(r["applicant"])]
        films, adstats = ors_ad.apply(films, ad_rows, run_kst, ad_withheld)
    pgstats = programs.apply(films, pgres, (lres_list or {}).get("rows"), bool(cgv_res), today)
    # past KMRB ratings for records the 3-month rating list does not cover
    lookup_path = data_dir / "lookup.json"
    lookup_cache = dict(load_json(lookup_path, {}).get("titles", {}))
    lookup_new = {k: dict(v) for k, v in lookup_cache.items()}
    lres, s_lookup = timed("kmrb_lookup", lambda: lookup.fetch(
        client, films, lookup_new, promo_cache, today_kst, today))
    if lres:
        lookup_cache = lookup_new
    lstats = lookup.apply_past_ratings(films, lookup_cache)
    # first opening and re-release dates from the KOBIS detail page of chain-listed films
    web_path = data_dir / "kobis_detail.json"
    legacy_web_path = data_dir / "rerelease.json"  # its name before the page gave more fields
    web_cache = {cd: dict(e) for cd, e in load_json(
        web_path if web_path.exists() else legacy_web_path, {}).get("movies", {}).items()}
    # rating number -> KOBIS code from release_lags (KOBIS lists the rating
    # number on the film's page): links rated records the KOBIS list missed
    lags_path = data_dir / "release_lags.json"
    try:
        lags = release.load_lags(lags_path)
        rating_codes = release.rating_codes(lags)
    except Exception as exc:  # noqa: BLE001 - retried (and reported) by the release step below
        log.warning("release_lags.json unreadable: %s", exc)
        lags, rating_codes = None, {}
    wres, s_web = timed("kobis_web", lambda: kobis_web.fetch(client, films, web_cache, today,
                                                             rating_codes))
    wstats = kobis_web.apply(films, web_cache, rating_codes)
    schstats = kobis_web.apply_schedule(films, sres["rows"]) if sres else {}
    bstats = kobis_web.apply_booking(films, bres) if bres else {}
    bostats = boxoffice.apply(films, bo_history)
    # a blocked company's records go, whatever source brought them in
    kept_films = [f for f in films if not (
        companies.company_key(f.get("applicant", "")) in blocked
        or kobis.blocked_distributor(f["stages"].get("kobis", {}), blocked))]
    manual_stats["records_removed"] = len(films) - len(kept_films)
    films = kept_films
    gstats = group_variants(films)
    promo.finalize(films)  # every run: merge() rebuilt the tags
    dormant_n = kobis_web.mark_dormant(films, today_kst)
    # opening dates: confirmed by a source, else estimated (collector/release.py)
    try:
        lags = lags if lags is not None else release.load_lags(lags_path)
        # a row's follow-up advances only on a day both KOBIS opening sources were read
        checked = s_kobis["status"] == "ok" and s_web["status"] == "ok"
        release_meta = release.apply(films, lags, today_kst, openings_checked=checked)
    except Exception as exc:  # noqa: BLE001 - a model failure must not stop the collection
        log.warning("release model failed: %s", exc)
        release_meta, lags = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:200]}, None
    # theatrical / candidate / non-theatrical class (collector/theatrical.py)
    try:
        theatrical_meta = theatrical.apply(
            films, lags if lags is not None else load_json(lags_path, {}), today_kst,
            allowed=allowed_keys)
    except Exception as exc:  # noqa: BLE001 - a classification failure must not stop the collection
        log.warning("theatrical classification failed: %s", exc)
        theatrical_meta = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:200]}
    poster_path = data_dir / "posters.json"
    poster_cache = {u: dict(e) for u, e in load_json(poster_path, {}).get("sources", {}).items()}
    prev_posters = {f["id"]: f["poster"] for f in existing if f.get("poster")}
    pores, s_posters = timed("posters", lambda: posters.fetch(
        client, films, poster_cache, data_dir / "posters", today))
    if not pores:  # keep the posters records had (their files were not touched)
        for f in films:
            if f["id"] in prev_posters:
                f["poster"] = prev_posters[f["id"]]

    sources = {}
    for name, st, extra in (
        ("ors_intake", s_ors, {"tabs": intake["stats"]} if intake else {}),
        ("kmrb_rating", s_kmrb, {"window": "3months", **ratings["stats"]} if ratings else {}),
        ("kmrb_promo", s_promo, {"window_days": promo_days,
                                 **(pres or {}).get("stats", {}), **pstats}),
        ("ors_ad", s_ors_ad, {"tabs": (adres or {}).get("stats", {}), **adstats, **settle_stats}),
        ("megabox_program", s_program, {**(pgres or {}).get("stats", {}), **pgstats}),
        ("kobis", s_kobis, {k: v for k, v in (kres or {}).items() if k != "kept_ids"} | kstats),
        ("megabox", s_megabox, {**(mres or {}).get("stats", {}), **mstats}),
        ("lotte", s_lotte, {**(lres_list or {}).get("stats", {}), **ltstats}),
        ("cgv", s_cgv, {**(cgv_res or {}).get("stats", {}), **cstats}),
        ("kmrb_lookup", s_lookup, {**(lres or {}), **lstats}),
        ("kobis_web", s_web, {**(wres or {}), **wstats, "dormant": dormant_n}),
        ("posters", s_posters, pores or {}),
        ("kobis_schedule", s_schedule, {"months": (sres or {}).get("stats", {}), **schstats}),
        ("kobis_companies", s_companies, {**(cres or {}), "trusted": len(company_trusted)}),
        ("kmrb_reps", s_reps, (reps_res or {}).get("stats", {})),
        ("kobis_booking", s_booking, {**(bres or {}).get("stats", {}), **bstats}),
        ("kobis_upcoming", s_upcoming, {**(ures or {}).get("stats", {}),
                                        "not_adult_genre": len(undated)}),
        ("kobis_boxoffice", s_boxoffice, {"window_days": boxoffice.WINDOW_DAYS,
                                          **(bores or {}), **bostats}),
    ):
        entry = {**st, **extra}
        last_ok = run_utc if st["status"] == "ok" else prev_sources.get(name, {}).get("last_ok_utc")
        if last_ok:
            entry["last_ok_utc"] = last_ok
        sources[name] = entry

    # signal history: after every stage is applied (collector/signals.py)
    sig_path = data_dir / "signals.json"
    try:
        sig_doc = signals.load(sig_path)
        sig_stats = signals.update(sig_doc, films, sources, run_utc)
    except Exception as exc:  # noqa: BLE001 - a broken log must not stop the collection
        log.warning("signal history failed: %s", exc)
        sig_doc, sig_stats = None, {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:200]}

    tag_counts = Counter(t for f in films for t in f["tags"])
    company_index = distributors.build(films, today_kst)
    meta = {
        "schema": SCHEMA_VERSION,
        "last_run_utc": run_utc,
        "manual_lists": manual_stats,  # counts only: the lists are not published
        "counts": {
            "films": len(films),
            "with_intake": sum("intake" in f["stages"] for f in films),
            "with_rating": sum("rating" in f["stages"] for f in films),
            "with_kobis": sum("kobis" in f["stages"] for f in films),
            "with_megabox": sum("megabox" in f["stages"] for f in films),
            "with_lotte": sum("lotte" in f["stages"] for f in films),
            "with_cgv": sum("cgv" in f["stages"] for f in films),
            "with_promo": sum("promo" in f["stages"] for f in films),
            "with_ors_ad": sum("ors_ad" in f["stages"] for f in films),
            "with_program": sum("program" in f["stages"] for f in films),
            "with_past_rating": sum("past_rating" in f["stages"] for f in films),
            "rerelease_candidates": sum(f["stages"].get("promo", {}).get("rerelease", False)
                                        for f in films),
            "chain_rereleases": sum(promo.chain_rerelease(f) for f in films),
            "seen_this_run": sum(f["last_seen"] == today for f in films),
            **gstats,
            "distributors": len(company_index["companies"]),
            "tags": dict(sorted(tag_counts.items())),
            **stats,
            "applicants_known": len(summary),
            "applicants_trusted": len(trusted),
            "applicants_adult_only": sum(1 for v in summary.values()
                                         if v["adult_count"] and not v["general_count"]),
            "applicants_mixed": sum(1 for v in summary.values()
                                    if v["adult_count"] and v["general_count"]),
        },
        "applicant_store": {"backfill": store.get("backfill"),
                            "trust_window_months": applicants.TRUST_WINDOW_MONTHS},
        "sources": sources,
        "requests": getattr(client, "request_count", None),
        "release_model": release_meta,
        "theatrical": theatrical_meta,
        "signals": sig_stats,
    }
    if ratings:
        write_json(apps_path, applicants.to_json(store, summary))
    if kres:
        write_json(kobis_path, {"schema": SCHEMA_VERSION, "movies": dict(sorted(cache.items()))})
    if pres:
        write_json(promo_path, {"schema": SCHEMA_VERSION, "backfill": promo_backfill,
                                "history": dict(sorted(promo_cache.items()))})
    if lres:
        write_json(lookup_path, {"schema": SCHEMA_VERSION, "titles": dict(sorted(lookup_cache.items()))})
    if pores:
        write_json(poster_path, {"schema": SCHEMA_VERSION, "sources": dict(sorted(poster_cache.items()))})
    if cres:
        write_json(comp_path, {"schema": SCHEMA_VERSION, "companies": dict(sorted(comp_cache.items()))})
    if bores:
        write_json(bo_path, {"schema": SCHEMA_VERSION, "days": dict(sorted(bo_history.items()))})
    if wres:
        write_json(web_path, {"schema": SCHEMA_VERSION, "movies": dict(sorted(web_cache.items()))})
        legacy_web_path.unlink(missing_ok=True)
    write_json(data_dir / "distributors.json", {"schema": SCHEMA_VERSION, **company_index})
    if lags is not None:
        write_json(lags_path, lags)
    write_json(films_path, {"schema": SCHEMA_VERSION, "films": films})
    write_json(data_dir / "meta.json", meta)
    if sig_doc is not None:
        signals.write(sig_path, sig_doc)
    return meta


def backfill_applicants(client, data_dir: Path, now: dt.datetime, months: int) -> dict:
    """One-time: fill data/applicants.json from a longer KMRB window. Writes only that file."""
    today_kst = now.astimezone(KST).date()
    ratings = kmrb.fetch_ratings(client, today_kst, months)
    apps_path = data_dir / "applicants.json"
    store = applicants.from_json(applicants.load(apps_path))
    applicants.update(store, ratings["general"], ratings["adult"])
    store["backfill"] = {"date": today_kst.isoformat(), "months": months,
                         "start": kmrb.months_back(today_kst, months).isoformat(),
                         **ratings["stats"]}
    summary = applicants.summarize(store, today_kst)
    write_json(apps_path, applicants.to_json(store, summary))
    return {"applicants": len(summary), "trusted": len(applicants.trusted(summary)),
            **ratings["stats"], "requests": getattr(client, "request_count", None)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect upcoming Korean film releases.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--backfill-applicants", type=int, choices=sorted(kmrb.SEARCH_DATE),
                    help="one-time: rebuild data/applicants.json from this many months of KMRB ratings")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs full request URLs (which would include the KOBIS key): keep it quiet
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    key = kobis.get_key()
    client = PoliteClient()
    try:
        now = dt.datetime.now(dt.timezone.utc)
        if args.backfill_applicants:
            print(f"backfill ok: {backfill_applicants(client, args.data_dir, now, args.backfill_applicants)}")
            return 0
        meta = collect(client, args.data_dir, now, key)
    except AllSourcesFailed as exc:
        print(f"ALL SOURCES FAILED, nothing written: {kobis.redact(str(exc), key)}", file=sys.stderr)
        return 3 if "blocked" in str(exc) else 1
    except Exception as exc:  # noqa: BLE001 - unexpected bug: leave data untouched
        print(f"FAILED, nothing written: {kobis.redact(repr(exc), key)}", file=sys.stderr)
        return 1
    finally:
        client.close()
    # the website files (collector/site.py) are built from the data just written; a failure here
    # leaves the previous site files in place and must not stop the data from being committed
    try:
        today = now.astimezone(KST).date().isoformat()
        site_msg = f"site: {len(site.write(args.data_dir, today)['written'])} files rewritten"
    except Exception as exc:  # noqa: BLE001
        log.exception("site build failed")
        site_msg = f"site: FAILED {type(exc).__name__}"
    c = meta["counts"]
    status = " ".join(f"{k}={v['status']}" for k, v in meta["sources"].items())
    print(f"ok: {status} films={c['films']} intake_published={c['intake_published']} "
          f"intake_withheld={c['intake_withheld']} rating_published={c['rating_published']} "
          f"rating_withheld_restricted={c['rating_withheld_restricted']} "
          f"with_kobis={c['with_kobis']} with_megabox={c['with_megabox']} "
          f"with_lotte={c['with_lotte']} chain_rereleases={c['chain_rereleases']} "
          f"with_promo={c['with_promo']} rerelease={c['rerelease_candidates']} requests={meta['requests']}")
    print(site_msg)
    return 0
