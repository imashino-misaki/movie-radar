"""Will the film open in cinemas? A `theatrical` class on every record.

Classes, first match wins:
  confirmed  theatrical evidence in the record's stages (`evidence` lists it):
             `kobis_open` (KOBIS 개봉일, API or detail page), `kobis_rerelease`
             (재개봉일), `megabox` / `lotte` / `cgv` (a chain listing, also a past
             one), `booking` (tickets on sale), `schedule` (the KOBIS calendar),
             `boxoffice` (the daily top 10), `program` (a chain's exclusive
             programme lists it: programs.py). A token release (below) is
             reported as `token` instead.
  token      opened, but a screening measure shows a token release: the
             `real_open` flag of a stage or of the data/release_lags.json row
             with the film's KOBIS code (no more paying viewers than
             screenings in the year, lags_build.real_open), else a
             `screenings` count under TOKEN_MIN_SCREENINGS (KOFIC's rule).
  unlikely   the applicant almost never opens its films in cinemas: at least
             MIN_RATINGS ratings at least HISTORY_MIN_AGE_DAYS old in
             data/release_lags.json, of which at most MAX_OPEN_SHARE opened.
  ad_only    a record created from ORS filings alone (ors_ad.ad_only: no
             rating, intake, KOBIS, chain or programme stage yet) none of
             whose applicants (the record's, or a filing's) has a theatrical
             history (MIN_REAL_SHARE) or is on the manual allow list;
             `ad_only` gives the record applicant's reason: `no_history`,
             `no_real_opening`, `token_share`, `unlikely` or `masked`. Such a
             record with a theatrical or allowed applicant is a candidate.
  candidate  everything else.
`p_open` copies `release.p_open` (the release model's probability that the film
opens in cinemas at all) and `p_open_real` copies `release.p_open_real` (a real
release, not a token one) when the record has them.

`intent` (only when not empty) lists marketing filings: `ors_ad` (an ORS
trailer / advertising filing in progress, ors_ad.py) and `promo` (a decided
KMRB trailer / poster rating, promo.py). They strengthen a candidate but never
confirm it and never lift `unlikely`: on 2026-11-01, at least 10 of the 24
received advertising-material filings in the ORS list were adult-video
posters (from three applicants), material for video and online release,
not cinemas.
"""
from __future__ import annotations

import datetime as dt
import re

from .companies import company_key
from .privacy import MASK

CLASSES = ("confirmed", "candidate", "unlikely", "token", "ad_only")

# KOFIC (영화진흥위원회) statistics count a film as released in a year when it
# had at least 40 screenings (상영횟수) that year.
TOKEN_MIN_SCREENINGS = 40

# Applicant rule, measured 2026-11-01 on data/release_lags.json: test ratings
# 2023-01..2025-06 (2,960 ratings, 1,396 openings), each judged on its
# applicant's ratings at least 180 days older. ">= 20 ratings and opening share
# <= 5%" moved 518 ratings (18%) to non-theatrical and lost 0 openings;
# ">= 10 and <= 5%" lost 2 (0.14%). Re-measured the same day with this module
# (applicants compared by company_key, openings known on the rating date):
# 3,079 test ratings, 1,482 openings; ">= 20" moved 522 (17.0%) and lost 0,
# ">= 10" moved 526 and lost 2 (0.13%). Only ratings at least 180 days old
# count, so recent ratings that may still open are not counted as failures.
MIN_RATINGS = 20
MAX_OPEN_SHARE = 0.05
HISTORY_MIN_AGE_DAYS = 180

# Theatrical history of an applicant (routing of records created from ORS
# filings): at least one real opening (lags_build.real_open) among its
# ratings at least HISTORY_MIN_AGE_DAYS old, real openings at least
# MIN_REAL_SHARE of its openings, and not `unlikely`. Measured 2026-11-02 on
# data/release_lags.json, each rating judged on its applicant's ratings at
# least 180 days older (openings known on the rating day): after applicants
# without a real opening, 6% of the ratings opened for real (33 of 541,
# ratings 2022-01..2025-06), after applicants with no history 55% (295 of
# 535). Among applicants with a real opening, a two-rate split of the next
# rating's real opening by the applicant's real share, searched over
# 0.05..0.95 in steps of 0.05, was best at 0.30 on ratings 2019-07..2025-06
# (5,078 ratings: 28% real below, 60% at or above); on 2022-01..2025-06 it
# was best at 0.15 (0.30: 12% vs 56%). Above 0.30 the rate does not rise with
# the share (2022-01..2025-06: 0.30-0.90 62%, 243 of 393; 0.90 and more 55%,
# 1,514 of 2,744), so a higher cut moves openings out without gaining
# precision. Adult-video importers are not left to this rule: their filings
# never create a record (ors_ad.can_create).
MIN_REAL_SHARE = 0.30

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH = re.compile(r"^\d{4}-\d{2}$")


def _day(s) -> str:
    return s if isinstance(s, str) and _DAY.match(s) else ""


def applicant_key(name: str) -> str:
    """Key of an applicant in the history ("" for masked or empty names)."""
    name = (name or "").strip()
    if not name or name == MASK:
        return ""
    return company_key(name)


def applicant_history(lags: dict, today: dt.date,
                      min_age_days: int = HISTORY_MIN_AGE_DAYS) -> dict[str, dict]:
    """{applicant key: {"ratings": n, "opened": k}} over the release_lags rows
    rated at least `min_age_days` before `today` (k: rows with an opening)."""
    cut = (today - dt.timedelta(days=min_age_days)).isoformat()
    out: dict[str, dict] = {}
    for row in (lags or {}).get("rows", []):
        key = applicant_key(row.get("applicant", ""))
        rated = _day(row.get("rating_date"))
        if not key or not rated or rated > cut:
            continue
        h = out.setdefault(key, {"ratings": 0, "opened": 0})
        h["ratings"] += 1
        h["opened"] += bool(_day(row.get("open_dt")))
    return out


def unlikely(hist: dict | None) -> bool:
    """The applicant-history rule (see MIN_RATINGS)."""
    return bool(hist) and hist["ratings"] >= MIN_RATINGS and \
        hist["opened"] <= MAX_OPEN_SHARE * hist["ratings"]


def real_history(lags: dict, today: dt.date,
                 min_age_days: int = HISTORY_MIN_AGE_DAYS) -> dict[str, dict]:
    """{applicant key: {"opened": k, "real": j}} over the same rows as
    applicant_history (j: openings with `real_open` true)."""
    cut = (today - dt.timedelta(days=min_age_days)).isoformat()
    out: dict[str, dict] = {}
    for row in (lags or {}).get("rows", []):
        key = applicant_key(row.get("applicant", ""))
        rated = _day(row.get("rating_date"))
        if not key or not rated or rated > cut:
            continue
        h = out.setdefault(key, {"opened": 0, "real": 0})
        h["opened"] += bool(_day(row.get("open_dt")))
        h["real"] += row.get("real_open") is True
    return out


def ad_reason(name: str, history: dict[str, dict], real: dict[str, dict]) -> str:
    """"" for an applicant with a theatrical history (see MIN_REAL_SHARE),
    else why not."""
    key = applicant_key(name)
    if not key:
        return "masked"
    h, r = history.get(key), real.get(key)
    if not h:
        return "no_history"
    if unlikely(h):
        return "unlikely"
    if not r or not r["real"]:
        return "no_real_opening"
    if r["real"] < MIN_REAL_SHARE * r["opened"]:
        return "token_share"
    return ""


def ad_route(rec: dict, history: dict[str, dict], real: dict[str, dict],
             allowed=frozenset()) -> str:
    """"" when a record created from ORS filings alone may be a candidate (its
    applicant or a filing's applicant is allowed or has a theatrical
    history), else the record applicant's reason."""
    names = [rec.get("applicant", "")]
    names += [n for n in rec.get("stages", {}).get("ors_ad", {}).get("applicants", []) if n not in names]
    if any(applicant_key(n) and applicant_key(n) in allowed for n in names):
        return ""
    reasons = [ad_reason(n, history, real) for n in names]
    return "" if "" in reasons else reasons[0]


def evidence(rec: dict) -> list[str]:
    """Short codes of the theatrical evidence in the record's stages."""
    st = rec.get("stages", {})
    hist, kobis = st.get("release_history", {}), st.get("kobis", {})
    out = []
    if _day(hist.get("open_dt")) or _day(kobis.get("open_dt")):
        out.append("kobis_open")
    if any(_day(d) for d in hist.get("rereleases", [])):
        out.append("kobis_rerelease")
    out += [c for c in ("megabox", "lotte", "cgv") if st.get(c)]
    if st.get("booking"):
        out.append("booking")
    sch = st.get("schedule", {})
    if _day(sch.get("date")) or (isinstance(sch.get("month"), str) and _MONTH.match(sch["month"])):
        out.append("schedule")
    if st.get("boxoffice"):
        out.append("boxoffice")
    if st.get("program"):
        out.append("program")
    return out


INTENT_STAGES = ("ors_ad", "promo")


def intent(rec: dict) -> list[str]:
    """Marketing filings in the record's stages (see `intent` above)."""
    st = rec.get("stages", {})
    return [k for k in INTENT_STAGES if st.get(k)]


def screening_index(lags: dict) -> dict[str, dict]:
    """{KOBIS code: measure} of release_lags rows carrying `screenings` or `real_open`."""
    out: dict[str, dict] = {}
    for row in (lags or {}).get("rows", []):
        m = _measure(row)
        if m and row.get("kobis_cd"):
            out[row["kobis_cd"]] = m
    return out


def _measure(d: dict) -> dict:
    m = {}
    if isinstance(d.get("screenings"), int) and not isinstance(d.get("screenings"), bool):
        m["screenings"] = d["screenings"]
    if isinstance(d.get("real_open"), bool):
        m["real_open"] = d["real_open"]
    return m


def kobis_codes(rec: dict) -> list[str]:
    st = rec.get("stages", {})
    codes = [st.get("release_history", {}).get("kobis_cd"), st.get("kobis", {}).get("movie_cd"),
             st.get("lotte", {}).get("kobis_cd"), st.get("kobis_detail", {}).get("kobis_cd")]
    return list(dict.fromkeys(c for c in codes if c))


def is_token(rec: dict, today: dt.date, screenings: dict[str, dict] | None = None) -> bool:
    """Opened (a KOBIS opening date on or before today) and a screening measure
    says the opening was a token one."""
    st = rec.get("stages", {})
    hist, kobis = st.get("release_history", {}), st.get("kobis", {})
    dates = [_day(hist.get("open_dt")), _day(kobis.get("open_dt"))]
    dates += [_day(d) for d in hist.get("rereleases", [])]
    if not any(d and d <= today.isoformat() for d in dates):
        return False
    measure = {}
    for stage in st.values():
        if isinstance(stage, dict) and _measure(stage):
            measure = _measure(stage)
            break
    if not measure:
        for cd in kobis_codes(rec):
            if cd in (screenings or {}):
                measure = screenings[cd]
                break
    if "real_open" in measure:
        return not measure["real_open"]
    return "screenings" in measure and measure["screenings"] < TOKEN_MIN_SCREENINGS


def classify(rec: dict, history: dict[str, dict], today: dt.date,
             screenings: dict[str, dict] | None = None, real: dict[str, dict] | None = None,
             allowed=frozenset()) -> dict:
    """The `theatrical` value of one record. `real`: real_history; `allowed`:
    company keys on the manual allow list."""
    from .ors_ad import ad_only
    rel = rec.get("release")
    p = rel.get("p_open") if isinstance(rel, dict) else None
    out: dict = {"class": "candidate", "evidence": evidence(rec)}
    if out["evidence"]:
        out["class"] = "token" if is_token(rec, today, screenings) else "confirmed"
    else:
        h = history.get(applicant_key(rec.get("applicant", "")))
        if h:
            out["applicant_history"] = dict(h)
            if unlikely(h):
                out["class"] = "unlikely"
        if ad_only(rec):
            why = ad_route(rec, history, real or {}, allowed)
            if why:
                out["class"], out["ad_only"] = "ad_only", why
            elif out["class"] == "unlikely":  # allowed, or a theatrical applicant filed too
                out["class"] = "candidate"
    out["p_open"] = p if isinstance(p, (int, float)) and not isinstance(p, bool) else None
    pr = rel.get("p_open_real") if isinstance(rel, dict) else None
    out["p_open_real"] = pr if isinstance(pr, (int, float)) and not isinstance(pr, bool) else None
    if intent(rec):
        out["intent"] = intent(rec)
    return out


def apply(films: list[dict], lags: dict | None, today: dt.date, allowed=frozenset()) -> dict:
    """Set `theatrical` on every record in place; returns the meta entry.
    `allowed`: company keys on the manual allow list."""
    history = applicant_history(lags or {}, today)
    real = real_history(lags or {}, today)
    screenings = screening_index(lags or {})
    counts = dict.fromkeys(CLASSES, 0)
    for rec in films:
        rec["theatrical"] = classify(rec, history, today, screenings, real, allowed)
        counts[rec["theatrical"]["class"]] += 1
    return {"counts": counts,
            "candidates_with_intent": sum(rec["theatrical"]["class"] == "candidate"
                                          and bool(rec["theatrical"].get("intent")) for rec in films),
            "applicants_unlikely": sum(unlikely(h) for h in history.values()),
            "rule": {"min_ratings": MIN_RATINGS, "max_open_share": MAX_OPEN_SHARE,
                     "min_age_days": HISTORY_MIN_AGE_DAYS,
                     "token_min_screenings": TOKEN_MIN_SCREENINGS,
                     "min_real_share": MIN_REAL_SHARE},
            "screening_measures": len(screenings)}
