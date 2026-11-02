"""Past KMRB film ratings for records without a rating stage.

The rating source covers only the last 3 months, so a film rated earlier (a
film still in cinemas, a KOBIS- or Megabox-only title, a same-version
re-release known only from its trailer) has no KMRB grade in the data. For
each such record, one KMRB title search over the whole film-rating history
(the same search as promo.search_history) finds earlier ratings of the same
title. The result is attached as stage "past_rating" (date, grade, rating
number) only when something ties the rating to this film (see pick): a
same-title rating alone may belong to another film.

Searches are cached in data/lookup.json by title key; titles already searched
for the trailer/poster source are reused from its cache. A leading "[...]"
series label ("[오페라] ...", "[2026 NT Live] ...") is left out of the search
and the comparison (see lookup_title); the record keeps its title.
"""
from __future__ import annotations

import datetime as dt

from .http import BlockedError
from .promo import HISTORY_CAP, search_history, title_key
from .text import grade_key, strip_label

UNDETERMINED = "등급미정"  # chain listings' grade before KMRB rates the film


def lookup_title(title: str) -> str:
    """The title searched for and compared with KMRB titles."""
    return strip_label(title)


def lookup_key(title: str) -> str:
    return title_key(lookup_title(title))


def needs_lookup(rec: dict) -> bool:
    """No rating in the window and no application in progress (an application
    means a new rating is coming; an old same-title rating is another film)."""
    st = rec["stages"]
    return "rating" not in st and "intake" not in st and bool(rec.get("title"))


def fetch(client, films: list[dict], cache: dict, promo_cache: dict, today: dt.date,
          stamp: str) -> dict:
    """Search titles of records without a rating stage (at most HISTORY_CAP per
    run). Updates `cache` (title key -> {"ratings": [...], "checked": date}) in
    place; entries no record needs any more are dropped."""
    from . import kmrb

    wanted = {lookup_key(f["title"]): lookup_title(f["title"]) for f in films if needs_lookup(f)}
    reused = searched = errors = 0
    # promo searches made before grades were stored lack "grade": not reused
    promo_by_title = {k.split("|", 1)[0]: v for k, v in promo_cache.items()
                      if "ratings" in v and all("grade" in r for r in v["ratings"])}
    cookie = False
    for key, title in sorted(wanted.items()):
        if key in cache:
            continue
        if key in promo_by_title:
            cache[key] = {"ratings": promo_by_title[key]["ratings"], "checked": stamp}
            reused += 1
            continue
        if searched >= HISTORY_CAP:
            continue
        if not cookie:
            client.get(kmrb.LIST_URL)  # session cookie
            cookie = True
        searched += 1
        try:
            cache[key] = {"ratings": search_history(client, title, today), "checked": stamp}
        except BlockedError:
            raise
        except Exception:  # noqa: BLE001 - one bad title search must not fail the source
            errors += 1  # not cached: retried next run
    for key in [k for k in cache if k not in wanted]:
        del cache[key]
    return {"wanted": len(wanted), "searched": searched, "reused_promo": reused,
            "errors": errors, "pending": sum(k not in cache for k in wanted)}


def pick(ratings: list[dict], rec: dict) -> dict | None:
    """The rating of this film among same-title ratings (search_history keeps
    only same-title ones), or None when nothing ties a rating to this film:
      1. the rating number KOBIS lists for the film;
      2. else, if KOBIS or Megabox give a grade: the latest rating with that grade;
      3. else, for a title known from trailer/poster ratings: its latest rating
         within the re-release period (the film's own rating), or for a
         re-release candidate its latest earlier rating.
    """
    from .promo import _cutoff

    if not ratings:
        return None
    st = rec["stages"]
    audits = set(st.get("kobis", {}).get("audit_nos", []))
    by_no = [r for r in ratings if r.get("rating_no") and r["rating_no"] in audits]
    if by_no:
        return max(by_no, key=lambda r: r["date"])
    # grades compared under their current names: KMRB and KOBIS keep the old
    # name on old ratings, the chains list the current one
    known = {grade_key(g) for g in st.get("kobis", {}).get("grades", [])}
    for chain in ("megabox", "lotte", "cgv"):
        if st.get(chain, {}).get("grade"):
            known.add(grade_key(st[chain]["grade"]))
    known -= {UNDETERMINED, ""}  # "no grade yet" ties no rating to the film
    if known:
        same = [r for r in ratings if grade_key(r.get("grade", "")) in known]
        return max(same, key=lambda r: r["date"]) if same else None
    promo = st.get("promo")
    if promo:
        cutoff = _cutoff(promo["first_date"])
        recent = [r for r in ratings if r["date"] >= cutoff]
        if recent:
            return max(recent, key=lambda r: r["date"])
        prior = [r for r in ratings if r["date"] in set(promo.get("prior_ratings", []))]
        if promo.get("rerelease") and prior:
            return max(prior, key=lambda r: r["date"])
    return None


def apply_past_ratings(films: list[dict], cache: dict) -> dict:
    """Attach stage "past_rating" (in place) to records without a rating stage."""
    stats = {"past_rating": 0}
    for rec in films:
        if not needs_lookup(rec):
            rec["stages"].pop("past_rating", None)  # the film got a rating in the window
            continue
        entry = cache.get(lookup_key(rec["title"]))
        if not entry:
            continue
        r = pick(entry["ratings"], rec)
        if r is None:
            rec["stages"].pop("past_rating", None)
            continue
        rec["stages"]["past_rating"] = {"date": r["date"], "grade": r.get("grade", ""),
                                        "rating_no": r.get("rating_no", "")}
        stats["past_rating"] += 1
    return stats
