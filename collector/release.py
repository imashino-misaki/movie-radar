"""Opening date of every record: confirmed by a source, or estimated.

Confirmed dates (`release.basis` != "model"), in this order:
  day   KOBIS (release_history / kobis `open_dt`, re-release dates, the
        opening data/release_lags.json gives for the record's own rating) >
        KOBIS booking (`rerelease_dt`, `open_dt`) > chains (Megabox, Lotte
        `release_date`) > the KOBIS schedule's planned date;
  month the KOBIS schedule's month, else Lotte's `release_month`.
A date before the film's current cycle (the earliest of its rating, filing
and trailer/poster dates) belongs to an earlier release and is skipped (KOBIS
keeps the first opening date on a re-release). Every source date is kept in
`release.sources`. Dormant records get no `release`.

Without a day, the opening is estimated with a mixture ("cure") survival
model: a rated film opens in cinemas with probability pi, and when it does,
log(1 + T) ~ Normal(m, s), T = days from rating to opening. Two targets get
their own fit and backtest: any theatrical opening (`p_open`, the dates)
and a real one (`p_open_real`; a token release, `real_open` false in
data/release_lags.json, counts as none and an opening not labelled yet is
censored the day before it). Training rows are data/release_lags.json, one
per rating, right-censored at their follow-up (today for records still
collected, else the file's `generated` date). Rows in a rating-day segment
(row_segments: short films, format versions of open films, VOD applicants)
stay out of the fit and, while no source dates the film, get their
segment's opening share conditioned on the time elapsed (SEGMENT_STATES).

Announced days and months are evidence with a measured reliability
(commitment_rates): q = P(opens), r = P(in the announced month | opens),
P(T = t) = q {r g_month(t) + (1 - r) g_outside(t)}. Every record gets a
signal state and a display tier from that state's measured hit rates
(tier_table, display).

  logit pi = x.gamma + a_g        m = x.beta + b_g
x: binary film features known at the rating (FEATURES, picked by the
backtest). g: the applicant (known at the rating; individuals form one
group), also once KOBIS names the film's distributor: a distributor-grouped
fit is backtested only as if the distributor were known at the rating
(optimistic: KOBIS registers it late, often near the opening, and mostly for
films that open), so it is reported as a diagnostic and never used for an
estimate. The company effects a_g, b_g are random effects whose
prior mean is a regression on company features known at prediction time
(GROUP_FEATURES: ratings per year, shares of its ratings with a trailer or
poster before the rating, short, restricted, foreign, format), fitted with
each company weighted by its reliability 1 / (tau^2 + v_g): a company with
hundreds of ratings counts no more than any other well-measured company, so
high-volume applicants whose films never open do not set the prior of other
companies. A company's effect is its own estimate shrunk toward that prior
by tau^2 / (tau^2 + v_g) (empirical Bayes; v_g its sampling variance); a new
company gets the prior from its film's features. Censored rows enter every
step through the EM expectations (P(opens) q and the expected log lag beyond
the censoring point), so slow or never-opening companies are not mistaken
for fast ones. Row weights halve every h days of age.

Backtest: rows are scored only once matured (followed up for at least the
95th percentile of real-opening lags, see maturity_days). Nested time
split: h, then features (forward, in FEATURES order), the calendar, the
company-effect variant and milestones are chosen on a tuning fold (fits on
what was known on its first day), keeping the simpler candidate unless the
more complex one is better by one standard error; the chosen model is
refitted every REFIT_DAYS from the first day of the later, untouched test
fold (each time on what was known that day; a row is predicted by the
latest refit on or before its rating) and reported there only. A film's
ratings share one fold; each film counts once. Reported in meta: p50 error
(mean and median), quantile coverage over all matured openers, P(opens
within the maturity horizon) Brier / log score and reliability by decile
(overall and for new or small companies, whose own data weigh less than the
prior, vs established ones).

Calendar: within each calendar week the lag model's mass is spread over the
weekdays by the weekday shares of past real openings (recency-weighted), so
quantile dates land on the days films open; kept when it improves the test
log score of the opening day.

Milestones: a rated film whose trailer / poster rating (KOBIS listing,
chain listing, presale) came after its rating is re-estimated from that
date with a model fitted on rows anchored the same way (its own pi and lag)
when that milestone's backtest beats the rating model conditioned on the
elapsed time.

Anchor: the rating; else the filing; else the first trailer/poster rating;
else the first ORS trailer / advertising filing (its receipt date, else the
day it was first seen: a filing not received yet has no date). An unrated
film adds its delay to the rating (filing-to-rating days of rows with both
dates; first trailer/poster to film rating days of current records, also
for an ORS filing, which comes about a day before its trailer/poster
rating), conditioned on no rating so far, to the rating-to-opening model.
Without observed delays there is no estimate; a trailer/poster only
re-release needs no new rating, so it has none either.

Estimates are conditional on what has not happened: given no opening e days
after the anchor, P(open within d) = pi (F(e+d) - F(e)) / (1 - pi F(e)) and
P(ever opens) = pi (1 - F(e)) / (1 - pi F(e)); quantiles are of the lag
given an opening after e.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import operator
from pathlib import Path
from statistics import NormalDist

from .companies import company_key
from .privacy import MASK
from .text import norm_key, split_format

N01 = NormalDist()
SCHEMA = 1
# candidate half-lives (days) for the recency weights; None = no weighting.
# The backtest picks one (the simplest within one standard error of the best).
H_GRID = (30, 60, 90, 180, 365, 730, None)
# film features known at the rating, tried in this order by the backtest
FEATURES = ("short", "format", "promo", "restricted", "foreign", "rerelease", "rule_adult")
# company features (prior mean of the company effects)
GROUP_FEATURES = ("ratings_per_year", "promo", "short", "restricted", "foreign", "format")
RESTRICTED = ("청소년관람불가", "제한상영가")
# runtime under this many minutes = short. Measured 2026-11-01 on the KMRB
# film ratings 2019-2026: every 단편영화 (1,309) runs under 40 minutes (max
# 39.8), and 8% of 극영화 ratings do too; KMRB's 종별 is not in the rating
# list the collector reads, the runtime is.
SHORT_MINUTES = 40
# a trailer / poster rating more than this many days before the film rating
# belongs to an earlier release (promo.PRIOR_DAYS, the collector's rule)
PROMO_LOOKBACK_DAYS = 365
# the backtest's 90-day target and the `by` horizon
HORIZON_DAYS = 90
# horizons (days from today) of the published and scored P(open within d)
WITHIN_DAYS = (7, 30, 90, 180, 365)
# rows count as matured once followed up for this quantile of the observed
# real-opening lags (so at most this share of eventual openers is unseen)
MATURITY_QUANTILE = 0.95
# share of the matured rows held out as the (latest) test set
TEST_SHARE = 0.25
# rolling-origin evaluation: the model is refitted every run, so the backtest
# refits it every REFIT_DAYS days of a fold, each fit on what was known that
# day, and predicts a row with the latest fit on or before its rating day.
# Measured 2026-11-01 on the test folds (horizon Brier / censored log score):
# any opening 0.1463 / 3.531 every 182 days, 0.1464 / 3.520 every 91, 0.1461 /
# 3.522 every 45; real opening 0.1500 / 3.265 every 182, 0.1484 / 3.253
# every 91. Refitting more often than every 91 days gained nothing
REFIT_DAYS = 91
# reliability band reported separately: predictions at or above this
BAND = 0.9
# milestones after the rating that can re-anchor an estimate: row field of
# its date in data/release_lags.json
MILESTONES = {"promo": "promo_after", "kobis": "kobis_seen", "chain": "chain_seen",
              "booking": "booking_seen"}
# EM stops when the log likelihood per (weighted) row moves less than this
# in a step. Measured 2026-11-01 on the fit at 2023-12-18: 1e-5 takes 28
# steps against 58 for 1e-6 with the same test log score, p50 error and
# Brier to 4 digits (1e-4: 13 steps, log score off in the 4th digit).
_TOL = 1e-5
_MAX_ITER = 500
_EPS = 1e-9
# numerical guard: keeps s > 0 if every opened row has the same lag
_SIGMA_MIN = 1e-3
# numerical guard on logits (|logit| 12 = probability 6e-6 from 0 or 1)
_LOGIT_MAX = 12.0
# alternations of the company-prior regression per EM step (warm-started,
# so the EM iterations finish its convergence; a speed setting only)
_META_STEPS = 3
# at most this many halvings of a Newton step of the logit coefficients
_HALVINGS = 20


# ---------------------------------------------------------------- dates

def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s[:10])


def _month_end(d: dt.date) -> dt.date:
    nxt = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return nxt - dt.timedelta(days=1)


def _valid_day(s) -> str:
    s = (s or "")[:10]
    try:
        dt.date.fromisoformat(s)
    except ValueError:
        return ""
    return s


def _valid_month(s) -> str:
    s = (s or "")[:7]
    try:
        dt.date.fromisoformat(s + "-01")
    except ValueError:
        return ""
    return s


# ---------------------------------------------------------------- confirmed dates

def cycle_start(rec: dict) -> str:
    """Earliest of rating, filing and first trailer/poster date ("" if none)."""
    st = rec["stages"]
    dates = [_valid_day(st.get("rating", {}).get("date")),
             _valid_day(st.get("intake", {}).get("date")),
             _valid_day(st.get("promo", {}).get("first_date"))]
    dates = [d for d in dates if d]
    return min(dates) if dates else ""


def last_run_start(srcs: dict) -> str:
    """The start of the latest run of a record with no current-cycle stage
    (an old film back for a re-release needs no new rating): its latest KOBIS
    opening, or a chain / program listing day after it (a re-release the
    chains list before KOBIS records it). "" without a KOBIS opening."""
    kobis = [d for d in [srcs.get("kobis", ""), srcs.get("kobis_api", ""),
                         *srcs.get("kobis_rereleases", []), srcs.get("release_lags", "")] if d]
    if not kobis:
        return ""
    listed = [srcs[k] for k in ("megabox", "lotte", "cgv", "program") if srcs.get(k)]
    return max(kobis + listed)


def rating_row(rec: dict, rows: dict | None) -> dict | None:
    """The release_lags row of the record's own rating ({rating_no: row})."""
    no = rec["stages"].get("rating", {}).get("rating_no")
    return (rows or {}).get(no) if no else None


def sources(rec: dict, row: dict | None = None) -> dict:
    """Every opening date the sources give (empty ones left out). `row`: the
    release_lags row of the record's own rating (rating_row), if any."""
    st = rec["stages"]
    hist, kobis = st.get("release_history", {}), st.get("kobis", {})
    out = {}
    k_hist, k_api = _valid_day(hist.get("open_dt")), _valid_day(kobis.get("open_dt"))
    if k_hist or k_api:
        out["kobis"] = k_hist or k_api
    if k_api and k_hist and k_api != k_hist:
        out["kobis_api"] = k_api
    rr = [d for d in (_valid_day(x) for x in hist.get("rereleases", [])) if d]
    if rr:
        out["kobis_rereleases"] = rr
    # the KOBIS opening (first release or re-release on or after the rating)
    # of the film KOBIS ties to this rating number, as release_lags records
    # it: the record's own stages lack it when KOBIS listed the film only
    # before the collector first saw it (the Open API list keeps films
    # opened in the last kobis.RECENT_DAYS days) or does not list a
    # screening-format rating's number at all
    lag_open = _valid_day((row or {}).get("open_dt")) if (row or {}).get("kobis_cd") else ""
    if lag_open and lag_open not in (k_hist, k_api, *rr):
        out["release_lags"] = lag_open
    bk = st.get("booking", {})
    for key, field in (("booking_rerelease", "rerelease_dt"), ("booking", "open_dt")):
        if _valid_day(bk.get(field)):
            out[key] = _valid_day(bk.get(field))
    for chain in ("megabox", "lotte", "cgv"):
        if _valid_day(st.get(chain, {}).get("release_date")):
            out[chain] = _valid_day(st[chain]["release_date"])
    sch = st.get("schedule", {})
    if _valid_day(sch.get("date")):
        out["schedule"] = _valid_day(sch["date"])
    elif _valid_month(sch.get("month")):
        out["schedule"] = _valid_month(sch["month"])
    for chain in ("lotte", "cgv"):
        if _valid_month(st.get(chain, {}).get("release_month")):
            out[f"{chain}_month"] = _valid_month(st[chain]["release_month"])
    # a chain's program line-up (`stages.program`: chain, program, month,
    # date, exclusive): its day, else its month
    prog = st.get("program", {})
    if isinstance(prog, dict):
        if _valid_day(prog.get("date")):
            out["program"] = _valid_day(prog["date"])
        elif _valid_month(prog.get("month")):
            out["program_month"] = _valid_month(prog["month"])
    return out


def confirmed(rec: dict, srcs: dict | None = None) -> tuple[str, str, str] | None:
    """(date, precision, basis) of the current cycle's opening per the sources."""
    srcs = sources(rec) if srcs is None else srcs
    start = cycle_start(rec) or last_run_start(srcs)

    def ok(d: str) -> bool:  # a month counts while its last day is in the cycle
        if len(d) == 7:
            return not start or _month_end(_d(d + "-01")).isoformat() >= start
        return not start or d >= start

    kobis = sorted(d for d in [srcs.get("kobis", ""), srcs.get("kobis_api", ""),
                               *srcs.get("kobis_rereleases", []), srcs.get("release_lags", "")]
                   if d and ok(d))
    if kobis:
        return kobis[0], "day", "kobis"
    for key, basis in (("booking_rerelease", "booking"), ("booking", "booking"),
                       ("megabox", "chain"), ("lotte", "chain"), ("cgv", "chain"),
                       ("program", "program")):
        if srcs.get(key) and ok(srcs[key]):
            return srcs[key], "day", basis
    sch = srcs.get("schedule", "")
    if sch and ok(sch):
        return sch, ("day" if len(sch) == 10 else "month"), "schedule"
    for key, basis in (("lotte_month", "chain"), ("cgv_month", "chain"),
                       ("program_month", "program")):
        if srcs.get(key) and ok(srcs[key]):
            return srcs[key], "month", basis
    return None


def distributor(rec: dict) -> str:
    """KOBIS 배급사 (detail page, else Open API), else "" ."""
    st = rec["stages"]
    for c in st.get("kobis_detail", {}).get("companies", []):
        if c.get("role") == "배급사" and c.get("name") and c["name"] != MASK:
            return c["name"]
    for name in st.get("kobis", {}).get("distributors", []):
        if name and name != MASK:
            return name
    return ""


def group_key(dist: str, applicant: str) -> str:
    name = dist or (applicant if applicant != MASK else "")
    return company_key(name) if name else ""



# ---------------------------------------------------------------- features

def runtime_minutes(s) -> float | None:
    """KMRB runtime ("115분36초", "90분 22초", "30초") in minutes, or None."""
    s = (s or "").replace(" ", "")
    mins = secs = None
    if "분" in s:
        head, _, s = s.partition("분")
        mins = int(head) if head.isdigit() else None
        if mins is None:
            return None
    if s.endswith("초") and s[:-1].isdigit():
        secs = int(s[:-1])
    if mins is None and secs is None:
        return None
    return (mins or 0) + (secs or 0) / 60


def _short(runtime) -> int:
    v = runtime_minutes(runtime)
    return int(v is not None and v < SHORT_MINUTES)


def _promo_before(promo_first: str, rating_date: str) -> int:
    """A trailer / poster rating of this cycle on or before the film rating."""
    if not promo_first or not rating_date or promo_first > rating_date:
        return 0
    return int((_d(rating_date) - _d(promo_first)).days <= PROMO_LOOKBACK_DAYS)


def row_features(row: dict) -> tuple:
    """FEATURES of a release_lags.json row (values known at its rating)."""
    rd = _valid_day(row.get("rating_date"))
    v = {"short": _short(row.get("runtime")),
         "format": int(bool(row.get("format"))),
         "promo": _promo_before(_valid_day(row.get("promo_first")), rd),
         "restricted": int(row.get("grade", "") in RESTRICTED),
         "foreign": int(row.get("kind") == "국외영화"),
         "rerelease": int(bool(row.get("rerelease"))),
         "rule_adult": int(row.get("adult_check") == "rule")}
    return tuple(v[f] for f in FEATURES)


def promo_dates(rec: dict) -> list[str]:
    """Trailer / poster rating dates of a record (sorted)."""
    p = rec["stages"].get("promo", {})
    ds = {d for d in (_valid_day(x) for x in [p.get("first_date"), *p.get("dates_seen", [])]) if d}
    return sorted(ds)


def rec_features(rec: dict, anchor_kind: str = "rating") -> tuple:
    """FEATURES of a current record as they stand at its rating (for a film
    not rated yet: a trailer / poster filed now precedes its rating)."""
    st = rec["stages"]
    rating = st.get("rating", {})
    rd = _valid_day(rating.get("date"))
    pd = promo_dates(rec)
    # an ORS filing anchor is a trailer / poster filed before the rating
    promo = (_promo_before(pd[0], rd) if pd and rd
             else int((bool(pd) or anchor_kind == "ors_ad") and anchor_kind != "rating"))
    foreign = rating.get("kind") == "국외영화" or (
        not rating and st.get("intake", {}).get("tab") == "MV_FOR")
    v = {"short": _short(rating.get("runtime")),
         "format": int(bool(split_format(rec.get("title", ""))[1])),
         "promo": promo, "restricted": int(rating.get("grade", "") in RESTRICTED),
         "foreign": int(foreign), "rerelease": int("rerelease" in rec.get("tags", [])),
         "rule_adult": 0}
    return tuple(v[f] for f in FEATURES)


# ---------------------------------------------------------------- release_lags.json

def load_lags(path: Path) -> dict:
    """The release_lags file; rows that are not objects are dropped (every
    consumer reads rows as dicts)."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            data["rows"] = [r for r in data["rows"] if isinstance(r, dict)]
            return data
    return {"schema": SCHEMA, "generated": "", "coverage": {"rating_from": "", "rating_to": ""},
            "rows": []}


def rating_codes(lags: dict) -> dict[str, str]:
    """{rating_no: KOBIS code} of the release_lags rows that have one."""
    return {r["rating_no"]: r["kobis_cd"] for r in lags.get("rows") or []
            if isinstance(r, dict) and r.get("rating_no") and r.get("kobis_cd")}


def _milestone_dates(rec: dict, today_s: str) -> dict:
    """First day the collector saw the record listed by KOBIS, by a chain,
    and in the presale (booking) list."""
    fs = rec.get("first_seen", {})
    chain = [fs[c] for c in ("megabox", "lotte", "cgv") if _valid_day(fs.get(c))]
    out = {"kobis_seen": _valid_day(fs.get("kobis")), "chain_seen": min(chain) if chain else ""}
    out["booking_seen"] = today_s if "booking" in rec["stages"] else ""
    return out


def accumulate(lags: dict, films: list[dict], today: dt.date) -> dict:
    """Add a row for every current rated record and fill openings (KOBIS
    opening days only) and milestone dates of rows that had none (in place).
    Rows are never dropped."""
    rows = lags.setdefault("rows", [])
    by_no = {r.get("rating_no"): r for r in rows if r.get("rating_no")}
    today_s = today.isoformat()
    # milestones the collector sees from its first run on: dates on or before
    # the first recording day may be later than the real first listing
    ms_from = lags.setdefault("milestones_from", {})
    if films and "kobis_seen" not in ms_from:
        for key, stages in (("kobis_seen", ("kobis",)), ("chain_seen", ("megabox", "lotte", "cgv"))):
            seen = [f["first_seen"][s] for f in films for s in stages
                    if _valid_day(f.get("first_seen", {}).get(s))]
            ms_from[key] = min(seen) if seen else today_s
        ms_from["booking_seen"] = today_s
    added = opened = 0
    for rec in films:
        st = rec["stages"]
        rating = st.get("rating", {})
        no, rdate = rating.get("rating_no", ""), _valid_day(rating.get("date"))
        if not no or not rdate:
            continue
        conf = confirmed(rec)
        # an opening is recorded only when KOBIS has it: a passed chain,
        # program, presale or schedule date is a listing, not an observed
        # opening (Megabox lists event screenings KOBIS does not record)
        open_dt = (conf[0] if conf and conf[2] == "kobis" and rdate <= conf[0] <= today_s
                   else "")
        pd = [d for d in promo_dates(rec) if d <= (open_dt or today_s)]
        row = {"rating_no": no, "rating_date": rdate,
               "receipt_date": _valid_day(st.get("intake", {}).get("date")),
               "title": rec.get("title", ""), "applicant": rec.get("applicant", ""),
               "distributor": distributor(rec), "kind": rating.get("kind", ""),
               "kobis_cd": st.get("kobis", {}).get("movie_cd", "")
               or st.get("release_history", {}).get("kobis_cd", ""),
               "open_dt": open_dt, "rerelease": "rerelease" in rec.get("tags", []),
               "grade": rating.get("grade", ""), "runtime": rating.get("runtime", ""),
               "format": split_format(rec.get("title", ""))[1],
               "promo_first": next((d for d in pd if _promo_before(d, rdate) or d >= rdate), ""),
               "promo_after": next((d for d in pd if d >= rdate), ""),
               # the first planned opening (a later day, or a month) a source
               # gave and the day it was seen: slips can be measured later
               "planned_first": conf[0] if conf and not open_dt and conf[0] >= today_s[:len(conf[0])]
               else "",
               "planned_seen": today_s if conf and not open_dt and conf[0] >= today_s[:len(conf[0])]
               else "",
               **_milestone_dates(rec, today_s)}
        old = by_no.get(no)
        if old is None:
            rows.append(row)
            by_no[no] = row
            added += 1
            opened += bool(open_dt)
            continue
        if not old.get("open_dt") and open_dt:
            old["open_dt"] = open_dt
            opened += 1
        for k in ("receipt_date", "distributor", "kobis_cd", "grade", "runtime", "promo_first",
                  "promo_after", "kobis_seen", "chain_seen", "booking_seen"):
            if not old.get(k) and row[k]:
                old[k] = row[k]
        if not old.get("planned_first") and row["planned_first"]:
            old["planned_first"], old["planned_seen"] = row["planned_first"], row["planned_seen"]
        if row["rerelease"]:
            old["rerelease"] = True
    rows.sort(key=lambda r: (r.get("rating_date", ""), r.get("rating_no", "")))
    dates = [r["rating_date"] for r in rows if _valid_day(r.get("rating_date"))]
    cov = lags.setdefault("coverage", {})
    if dates:
        cov["rating_from"] = min([d for d in (cov.get("rating_from"), dates[0]) if d])
        cov["rating_to"] = max([d for d in (cov.get("rating_to"), dates[-1]) if d])
    lags["schema"] = SCHEMA
    lags["generated"] = lags.get("generated") or today_s
    lags["updated"] = today_s
    return {"rows_added": added, "openings_added": opened}


class Obs:
    """One training row: rating day (ordinal), real-opening lag (days) or
    None, last day its opening is known (ordinal), company (the applicant,
    known at the rating), FEATURES values, the lag of an opening a source
    has set for a later day (or None), the lags of milestones after the
    rating ({kind: days}), whether its opening was a token release, the
    distributor-or-applicant company (known once KOBIS lists the film) and
    a film key (KOBIS code, else title: ratings of one film share it)."""
    __slots__ = ("r", "t", "end", "g", "x", "sched", "ms", "token", "gd", "film", "seg")

    def __init__(self, r, t, end, g="", x=None, sched=None, ms=None, token=False, gd=None,
                 film="", seg=""):
        self.r, self.t, self.end, self.g = r, t, end, g
        self.x = tuple(x) if x is not None else (0,) * len(FEATURES)
        self.sched, self.ms, self.token = sched, ms or {}, token
        self.gd = g if gd is None else gd
        self.film = film
        self.seg = seg

    def regroup(self, g: str) -> "Obs":
        return Obs(self.r, self.t, self.end, g, self.x, self.sched, self.ms, self.token, self.gd,
                   self.film, self.seg)

    @property
    def opened(self) -> bool:
        return self.t is not None and self.r + self.t <= self.end


def observations(lags: dict, tracked: set[str], today: dt.date,
                 scheduled: dict | None = None, target: str = "real") -> list[Obs]:
    """Rows as observations. Rows of records still collected are followed up
    to today; others only to the date the file's openings were last checked
    in full (`generated`). `target` "any": every opening counts; "real": an
    opening counts when `real_open` is true, a token release (`real_open`
    false) counts as none and an opening not labelled yet censors the row
    the day before it. `scheduled`: rating_no -> a later day a source has
    set for the opening. Each observation carries its rating-day segment
    (row_segments)."""
    scheduled = scheduled or {}
    segs = row_segments(lags.get("rows", []))
    checked = _valid_day(lags.get("generated")) or today.isoformat()
    checked_o, today_o = _d(checked).toordinal(), today.toordinal()
    ms_from = lags.get("milestones_from", {})
    out = []
    for row in lags.get("rows", []):
        rd = _valid_day(row.get("rating_date"))
        if not rd:
            continue
        r = _d(rd).toordinal()
        end = today_o if row.get("rating_no") in tracked else max(checked_o, r)
        od = _valid_day(row.get("open_dt"))
        t, token = None, False
        if od:
            lag = _d(od).toordinal() - r
            if lag < 0:
                continue
            real = row.get("real_open")
            if target == "any" or real is True:
                t = lag
                token = real is False
            elif real is False:
                token = True
            else:
                end = min(end, r + max(lag - 1, 0))
        ms = {}
        for kind, field in MILESTONES.items():
            md = _valid_day(row.get(field))
            if md and md > rd and (kind == "promo" or md > ms_from.get(field, "")):
                ms[kind] = _d(md).toordinal() - r
        sd = _valid_day(scheduled.get(row.get("rating_no"), "")) if t is None else ""
        out.append(Obs(r, t, end, group_key("", row.get("applicant", "")), row_features(row),
                       _d(sd).toordinal() - r if sd and sd >= rd else None, ms, token,
                       group_key(row.get("distributor", ""), row.get("applicant", "")),
                       _row_film(row), segs.get(row.get("rating_no"), "")))
    return out


# ---------------------------------------------------------------- segments

# Rating-day segments outside the theatrical model, first match wins. Measured
# 2026-11-01 on the ratings 2019-01..2025-06 (8,622, followed up 16+ months):
# short (단편영화 or runtime under SHORT_MINUTES) 1,904 rows, 0.1% opened;
# vod_applicant (theatrical.unlikely on the applicant's ratings at least
# HISTORY_MIN_AGE_DAYS older, openings known on the rating day) 681 rows, none
# opened; feature-length candidates 6,559 rows, 73.3% opened (65.4% real).
# The share is counted within the maturity horizon but used as the eventual
# P(opens): measured 2026-11-01 on data/release_lags.json, 3 of the 2,352
# short rows ever opened (after 15, 15 and 133 days; horizon 298 days) and
# none of the 176 vod_applicant rows, so no segment row opened after it.
SEGMENTS = ("short", "format_of_open", "vod_applicant")


def _seg_short(category: str, runtime) -> bool:
    return category == "단편영화" or bool(_short(runtime))


def row_segments(rows: list[dict]) -> dict[str, str]:
    """{rating_no: segment} for rows outside the theatrical candidates, each
    decided with what was known on its rating day: `short`; `format_of_open`
    (a screening-format rating of a film that had opened before it:
    match "format", first opening before the rating, no opening of its own);
    `vod_applicant` (theatrical.unlikely over the applicant's ratings at least
    HISTORY_MIN_AGE_DAYS older than this one, counting the openings on or
    before this rating's date)."""
    from bisect import bisect_right, insort

    from .theatrical import HISTORY_MIN_AGE_DAYS, applicant_key, unlikely
    out: dict[str, str] = {}
    by_app: dict[str, list] = {}
    for r in rows:
        no, rd = r.get("rating_no"), _valid_day(r.get("rating_date"))
        if not no or not rd:
            continue
        first = _valid_day(r.get("first_open_dt"))
        if _seg_short(r.get("category", ""), r.get("runtime")):
            out[no] = "short"
        elif r.get("match") == "format" and not _valid_day(r.get("open_dt")) and first and first < rd:
            out[no] = "format_of_open"
        if k := applicant_key(r.get("applicant", "")):
            by_app.setdefault(k, []).append((rd, _valid_day(r.get("open_dt")), no))
    gap = dt.timedelta(days=HISTORY_MIN_AGE_DAYS)
    for items in by_app.values():
        items.sort()
        opens: list[str] = []
        j = 0
        for rd, _, no in items:
            cut = (_d(rd) - gap).isoformat()
            while j < len(items) and items[j][0] <= cut:
                if items[j][1]:
                    insort(opens, items[j][1])
                j += 1
            if no not in out and unlikely({"ratings": j, "opened": bisect_right(opens, rd)}):
                out[no] = "vod_applicant"
    return out


def rec_segment(rec: dict, row: dict | None, history: dict) -> str:
    """Segment of a current record (today's applicant history,
    theatrical.applicant_history); `row`: its release_lags row, if any."""
    from .theatrical import applicant_key, unlikely
    rating = rec["stages"].get("rating", {})
    row = row or {}
    if rating and _seg_short(row.get("category", ""), rating.get("runtime") or row.get("runtime")):
        return "short"
    if unlikely(history.get(applicant_key(rec.get("applicant", "")))):
        return "vod_applicant"
    return ""


# ---------------------------------------------------------------- the model

def _upper(a: float) -> float:
    """P(Z > a)."""
    return 0.5 * math.erfc(a / math.sqrt(2))


def _mills(a: float) -> float:
    """phi(a) / P(Z > a)."""
    if a > 30:
        return a + 1 / a
    return math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) / _upper(a)


def _sigmoid(x: float) -> float:
    x = max(-_LOGIT_MAX, min(_LOGIT_MAX, x))
    return 1 / (1 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(1 - _EPS, max(_EPS, p))
    return math.log(p / (1 - p))


def _dot(a, b) -> float:
    return sum(map(operator.mul, a, b))


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                fac = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= fac * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _wls(rows: list[tuple[float, list[float], float]], dim: int) -> list[float] | None:
    """Weighted least squares: rows of (weight, x, y)."""
    xtx = [[0.0] * dim for _ in range(dim)]
    xty = [0.0] * dim
    for u, x, y in rows:
        if u <= 0:
            continue
        for i in range(dim):
            xty[i] += u * x[i] * y
            for j in range(i, dim):
                xtx[i][j] += u * x[i] * x[j]
    for i in range(dim):
        for j in range(i):
            xtx[i][j] = xtx[j][i]
    return _solve(xtx, xty)


def _ml_tau2(stats: list[tuple[float, float]]) -> float:
    """Maximum likelihood between-group variance from (group mean residual,
    its sampling variance) pairs: means ~ N(0, tau^2 + v). 0 when the score
    at 0 is not positive; else the root of the score, by bisection (the score
    is negative once tau^2 exceeds every squared mean)."""
    stats = [(m, v) for m, v in stats if v > 0]

    def score(t):
        return sum(m * m / (t + v) ** 2 - 1 / (t + v) for m, v in stats)

    if len(stats) < 2 or score(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, max(m * m for m, _ in stats)
    for _ in range(100):
        mid = (lo + hi) / 2
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2


def active_columns(zs: list[list[float]]) -> list[int]:
    """Columns of the group features that vary over the groups and are not
    collinear with earlier ones (column 0, the intercept, always)."""
    if not zs:
        return [0]
    active = [0]
    for j in range(1, len(zs[0])):
        if max(z[j] for z in zs) > min(z[j] for z in zs) and _wls(
                [(1.0, [z[i] for i in active + [j]], 0.0) for z in zs], len(active) + 1):
            active.append(j)
    return active


def meta_regression(est: list[tuple[list[float], float, float]], tau2: float = 0.0,
                    active: list[int] | None = None, steps: int = 100):
    """Random-effects regression of group estimates on group features:
    theta_g ~ N(z_g . coef, tau^2 + v_g) from (z_g, theta_g, v_g). Each
    group is weighted by 1 / (tau^2 + v_g): a group measured precisely counts
    once, however many rows it has. coef by weighted least squares and
    tau^2 by maximum likelihood (ML score of the residuals), alternated.
    Features outside `active` (default: active_columns) get coefficient 0.
    At most `steps` alternations (a fit's EM warm-starts it every step).
    Returns (coef, tau2); coef None when the features are collinear."""
    est = [e for e in est if e[2] > 0 and math.isfinite(e[1])]
    if not est:
        return None, 0.0
    dim = len(est[0][0])
    if active is None:
        active = active_columns([z for z, _, _ in est])
    # column sums (per active feature pair) are computed once per call and
    # reweighted each alternation: the same sums as a weighted least squares
    # over the groups, without a Python loop over groups per step
    ths = [th for _, th, _ in est]
    vs = [v for _, _, v in est]
    cols = [[z[i] for z, _, _ in est] for i in active]
    k = len(active)
    prods = {(i, j): list(map(operator.mul, cols[i], cols[j])) for i in range(k) for j in range(i, k)}
    coef = None
    for _ in range(steps):
        u = [1 / (tau2 + v) for v in vs]
        xtx = [[0.0] * k for _ in range(k)]
        for (i, j), pr in prods.items():
            xtx[i][j] = xtx[j][i] = sum(map(operator.mul, u, pr))
        uth = list(map(operator.mul, u, ths))
        sub = _solve(xtx, [sum(map(operator.mul, uth, c)) for c in cols])
        if sub is None:
            return coef, tau2
        coef = [0.0] * dim
        for i, c in zip(active, sub):
            coef[i] = c
        old = tau2
        # Fisher scoring step of the ML score in tau^2 (warm-started)
        fitted = [0.0] * len(est)
        for c, col in zip(sub, cols):
            fitted = [fv + c * zv for fv, zv in zip(fitted, col)]
        num = den = 0.0
        for th, v, fv, w in zip(ths, vs, fitted, u):
            num += w * w * ((th - fv) ** 2 - tau2 - v)
            den += w * w
        tau2 = max(0.0, tau2 + num / den)
        if abs(tau2 - old) < 1e-8:
            break
    return coef, tau2


def group_features(data: list) -> dict:
    """{g: z} with z = [1, log ratings per year, shares of GROUP_FEATURES
    among its rows] from the rows of one fit (features known at each rating,
    no outcomes). data: (ordinal rating day, g, FEATURES values)."""
    acc: dict = {}
    idx = [FEATURES.index(f) for f in GROUP_FEATURES[1:]]
    for r, g, x in data:
        e = acc.setdefault(g, [0, r, r, [0] * len(idx)])
        e[0] += 1
        e[1], e[2] = min(e[1], r), max(e[2], r)
        for j, i in enumerate(idx):
            e[3][j] += x[i]
    return {g: _z(n, (hi - lo + 1) / 365.25, [s / n for s in sums])
            for g, (n, lo, hi, sums) in acc.items()}


def _z(n: int, years: float, shares: list[float]) -> list[float]:
    return [1.0, math.log(n / max(years, 1.0)), *shares]


def z_new(x: tuple) -> list[float]:
    """Company features of a company with no rows: one rating, this film's."""
    return _z(1, 1.0, [float(x[FEATURES.index(f)]) for f in GROUP_FEATURES[1:]])


class Fit:
    """A fitted model; `dist(g, x)` gives (m, s, pi, known) for one film."""

    def __init__(self):
        self.feats: tuple = ()
        self.sel: list = []
        self.groups = True
        self.beta: list = []     # location coefficients of the film features
        self.gamma: list = []    # logit coefficients of the film features
        self.sigma = 1.0         # within-company SD of the log lag
        self.eta = None          # company prior: location coefficients
        self.delta = None        # company prior: logit coefficients
        self.tau2_b = self.tau2_a = 0.0
        self.b: dict = {}        # company location effect (posterior)
        self.a: dict = {}        # company logit effect (posterior)
        self.wb: dict = {}       # weight of the company's own data, location
        self.wa: dict = {}       # weight of the company's own data, logit
        self.z: dict = {}        # company features
        self.n_rows: dict = {}   # effective rows per company
        self.n = self.opened = 0
        self.iterations = 0
        self.cal = None          # weekday shares (calendar), or None
        self.milestones: dict = {}
        self.active: list = [0]
        self.count: dict = {}    # rows per company
        self.k_rows = math.inf   # rows at which a company's own record weighs as much as the prior
        self.recal = None        # (a, b): P(opens) recalibrated as sigmoid(a + b logit pi), or None

    def xs(self, x) -> list:
        return [x[i] for i in self.sel]

    def own_weight(self, g: str) -> float:
        """1.0 for an established company (at least k_rows ratings in the
        fit: its own record outweighs the prior), else 0.0."""
        return float(self.count.get(g, 0) >= self.k_rows)

    def dist(self, g: str, x=None) -> tuple[float, float, float, bool]:
        x = x if x is not None else (0,) * len(FEATURES)
        xs = self.xs(x)
        if not self.groups:
            g = "*"
        known = g in self.a
        if known:
            lb, la = self.b[g], self.a[g]
            vb = self.tau2_b * (1 - self.wb.get(g, 0.0))
            va = self.tau2_a * (1 - self.wa.get(g, 0.0))
        else:
            z = z_new(x)
            lb = _dot(self.eta, z) if self.eta else 0.0
            la = _dot(self.delta, z) if self.delta else 0.0
            vb, va = self.tau2_b, self.tau2_a
        m = _dot(self.beta, xs) + lb
        # mean of the logistic over the effect's uncertainty (probit approximation)
        pi = _sigmoid((_dot(self.gamma, xs) + la) / math.sqrt(1 + math.pi * va / 8))
        return m, math.sqrt(self.sigma ** 2 + vb), recal_pi(min(1 - _EPS, max(_EPS, pi)), self.recal), \
            known and self.groups


def fit(obs: list[Obs], ref: int, half_life: float | None = None, groups: bool = True,
        feats: tuple = (), scheduled: bool = False, censored_groups: bool = True,
        init: "Fit | None" = None) -> Fit | None:
    """Fit on what was known on day `ref` (ordinal): rows rated before ref;
    openings after ref, or after a row's last check, censored. With
    `scheduled`, an opening a source has set for a later day counts as
    observed (only for the fit on today's data: the backtest cannot know
    what was scheduled in the past). Without `groups` every row is one
    company. `censored_groups` False: company location effects from opened
    rows only (the previous model; kept for the backtest comparison).
    `init`: a fit whose estimates start the EM (a warm start for the
    backtest's refits: the same fixed point in fewer steps)."""
    sel = [FEATURES.index(f) for f in feats]
    data = []  # (w, y or None, censor point c, g, xs)
    for o in obs:
        if o.r >= ref:
            continue
        end = min(ref, o.end)
        w = 0.5 ** ((ref - o.r) / half_life) if half_life else 1.0
        g = o.g if groups else "*"
        xs = tuple(o.x[i] for i in sel)
        if o.t is not None and o.r + o.t <= end:
            data.append((w, math.log1p(o.t), 0.0, g, xs))
        elif scheduled and o.sched is not None:
            data.append((w, math.log1p(o.sched), 0.0, g, xs))
        else:
            data.append((w, None, math.log1p(max(0, end - o.r)), g, xs))
    opened = [x for x in data if x[1] is not None]
    if len(opened) < 2:
        return None
    f = Fit()
    f.feats, f.sel, f.groups = tuple(feats), sel, groups
    f.n, f.opened = len(data), len(opened)
    if groups:
        f.z = group_features([(o.r, o.g, o.x) for o in obs if o.r < ref])
        f.active = active_columns(list(f.z.values()))
    p = len(sel)
    ys = [x[1] for x in opened]
    m0 = sum(ys) / len(ys)
    f.sigma = max(math.sqrt(sum((y - m0) ** 2 for y in ys) / len(ys)), _SIGMA_MIN)
    l0 = _logit(len(opened) / len(data))
    f.beta, f.gamma = [0.0] * p, [0.0] * p
    gs = sorted({x[3] for x in data})
    f.b = {g: m0 for g in gs}
    f.a = {g: l0 for g in gs}
    if init is not None and init.groups == groups:
        _warm_start(f, init, gs)
    # Kish effective rows per company
    kish: dict = {}
    for w, _, _, g, _ in data:
        e = kish.setdefault(g, [0.0, 0.0])
        e[0] += w
        e[1] += w * w
    f.n_rows = {g: s * s / s2 for g, (s, s2) in kish.items()}
    for x in data:
        f.count[x[3]] = f.count.get(x[3], 0) + 1
    wsum = sum(x[0] for x in data)
    cells = {x[4] for x in data}
    prev_ll = None
    for it in range(_MAX_ITER):
        f.iterations = it + 1
        # E step: P(opens) q, E[y], E[y^2] of each row given what is known
        rows = []
        ll = 0.0
        cm = {xs: _dot(f.beta, xs) for xs in cells}
        cl = {xs: _dot(f.gamma, xs) for xs in cells}
        for w, y, c, g, xs in data:
            m = cm[xs] + f.b[g]
            pi = _sigmoid(cl[xs] + f.a[g])
            if y is not None:
                rows.append((w, 1.0, y, y * y, g, xs))
                ll += w * (math.log(pi) - math.log(f.sigma) - 0.5 * ((y - m) / f.sigma) ** 2)
                continue
            a = (c - m) / f.sigma
            surv = _upper(a)
            p_c = 1 - pi + pi * surv
            ll += w * math.log(max(p_c, 1e-300))
            q = pi * surv / p_c if p_c > 0 else 0.0
            lam = _mills(a)
            s = f.sigma
            rows.append((w, q, m + s * lam, s * s * (1 + a * lam) + 2 * m * s * lam + m * m, g, xs))
        # stop when the weighted log likelihood per row stops moving
        if prev_ll is not None and abs(ll - prev_ll) < _TOL * wsum:
            break
        prev_ll = ll
        _m_step_pi(f, rows)
        _m_step_location(f, rows, data, censored_groups)
    if groups and f.tau2_a > 0:
        # a binomial share from n rows has variance about p(1-p)/n, i.e.
        # 1/(n p(1-p)) on the logit scale: equal to tau^2 at n = k_rows
        pbar = sum(_sigmoid(v) for v in f.a.values()) / len(f.a)
        f.k_rows = 1 / (f.tau2_a * pbar * (1 - pbar))
    elif not groups:
        f.k_rows = 0.0
    return f


def _warm_start(f: Fit, init: Fit, gs: list) -> None:
    """Start f's EM at init's estimates (coefficients by feature name, new
    ones 0; a company init lacks gets init's prior mean)."""
    bi = dict(zip(init.feats, init.beta))
    gi = dict(zip(init.feats, init.gamma))
    f.beta = [bi.get(k, 0.0) for k in f.feats]
    f.gamma = [gi.get(k, 0.0) for k in f.feats]
    f.sigma = init.sigma
    f.eta, f.delta, f.tau2_a, f.tau2_b = init.eta, init.delta, init.tau2_a, init.tau2_b
    for g in gs:
        if g in init.b:
            f.b[g], f.a[g] = init.b[g], init.a[g]
        elif f.groups and init.eta and init.delta and g in f.z:
            f.b[g], f.a[g] = _dot(init.eta, f.z[g]), _dot(init.delta, f.z[g])


def _m_step_pi(f: Fit, rows: list) -> None:
    """Logit coefficients of the film features (Newton steps with the
    company effects as offsets), then the company effects."""
    agg: dict = {}  # (g, xs) -> [sum w, sum w q]
    for w, q, _, _, g, xs in rows:
        e = agg.setdefault((g, xs), [0.0, 0.0])
        e[0] += w
        e[1] += w * q
    # Newton steps on (c, gamma) with a common intercept c added to every
    # company effect: the level and the slopes move together (alternating
    # them converges slowly when a feature is common: measured 2026-11-01 on
    # the fit at 2023-12-06, 17 EM steps instead of 46, same estimates)
    p = len(f.gamma)
    for _ in range(3 if p else 0):
        # score and information summed per feature cell first
        cell: dict = {}
        cl = {}
        for (g, xs), (sw, swq) in agg.items():
            if xs not in cl:
                cl[xs] = _dot(f.gamma, xs)
                cell[xs] = [0.0, 0.0]
            pr = _sigmoid(cl[xs] + f.a[g])
            e = cell[xs]
            e[0] += swq - sw * pr
            e[1] += sw * pr * (1 - pr)
        grad, hess = [0.0] * (p + 1), [[0.0] * (p + 1) for _ in range(p + 1)]
        for xs, (r, v) in cell.items():
            xe = (1, *xs)
            for i in range(p + 1):
                grad[i] += r * xe[i]
                for j in range(p + 1):
                    hess[i][j] += v * xe[i] * xe[j]
        step = _solve(hess, grad)
        if step is None:
            break
        # step halving: a step must not lower the expected log likelihood
        # (a feature whose rows all open has its optimum at the logit bound,
        # where full Newton steps would jump from one bound to the other)
        old = (f.a, f.gamma)
        base = _pi_objective(f, agg)
        for _ in range(_HALVINGS):
            f.a = {g: v + step[0] for g, v in old[0].items()}
            f.gamma = [max(-_LOGIT_MAX, min(_LOGIT_MAX, gm + st)) for gm, st in zip(old[1], step[1:])]
            if _pi_objective(f, agg) >= base:
                break
            step = [st / 2 for st in step]
        else:
            f.a, f.gamma = old
            break
    if not f.groups and p:
        return  # the one company's effect is the common intercept, set above
    score: dict = {}
    cl = {xs: _dot(f.gamma, xs) for _, xs in agg}
    for (g, xs), (sw, swq) in agg.items():
        pr = _sigmoid(cl[xs] + f.a[g])
        e = score.setdefault(g, [0.0, 0.0])
        e[0] += swq - sw * pr
        e[1] += sw * pr * (1 - pr)
    # one Newton step per company: its own estimate theta and variance v
    est = {g: (max(-_LOGIT_MAX, min(_LOGIT_MAX, f.a[g] + s / i)) if i > 0 else f.a[g],
               1 / i if i > 0 else math.inf) for g, (s, i) in score.items()}
    if not f.groups:
        f.a = {g: th for g, (th, _) in est.items()}
        return
    f.delta, f.tau2_a = meta_regression([(f.z[g], th, v) for g, (th, v) in est.items()
                                         if math.isfinite(v)], f.tau2_a, f.active, _META_STEPS)
    f.a, f.wa = _posterior(f, est, f.delta, f.tau2_a)


def _pi_objective(f: Fit, agg: dict) -> float:
    """Expected complete-data log likelihood of the P(opens) part."""
    tot = 0.0
    cl: dict = {}
    for (g, xs), (sw, swq) in agg.items():
        if xs not in cl:
            cl[xs] = _dot(f.gamma, xs)
        x = max(-_LOGIT_MAX, min(_LOGIT_MAX, cl[xs] + f.a[g]))
        # log sigmoid(x) and log(1 - sigmoid(x)), stable for either sign
        lp = -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))
        tot += swq * lp + (sw - swq) * (lp - x)
    return tot


def _posterior(f: Fit, est: dict, coef, tau2: float) -> tuple[dict, dict]:
    """Company effects shrunk toward their prior mean z . coef."""
    eff, wt = {}, {}
    for g, (th, v) in est.items():
        prior = _dot(coef, f.z[g]) if coef else th
        k = tau2 / (tau2 + v) if math.isfinite(v) and tau2 + v > 0 else 0.0
        eff[g], wt[g] = prior + k * (th - prior), k
    return eff, wt


def _m_step_location(f: Fit, rows: list, data: list, censored_groups: bool) -> None:
    """Location coefficients, company location effects, within-company SD."""
    p = len(f.beta)
    if p:  # normal equations need only per-cell sums
        cell: dict = {}
        for w, q, ey, _, g, xs, *_ in rows:
            e = cell.setdefault(xs, [0.0, 0.0])
            e[0] += w * q
            e[1] += w * q * (ey - f.b[g])
        # with a common intercept c, added to every company effect: the
        # slopes and the level move together instead of in alternation
        # (alternating them converges slowly when a feature is common)
        new = _wls([(su, [1.0, *xs], suy / su) for xs, (su, suy) in cell.items() if su > 0], p + 1)
        if new is not None:
            f.beta = new[1:]
            f.b = {g: v + new[0] for g, v in f.b.items()}
    cm: dict = {}
    for row in rows:
        xs = row[5]
        if xs not in cm:
            cm[xs] = _dot(f.beta, xs)
    # g -> [sum u, sum u*w, sum u*residual], u = w q: a censored row counts
    # by its probability of opening, so the effective size is the expected
    # number of openers (sum q when unweighted; Kish size when all opened)
    acc: dict = {}
    for (w, q, ey, _, g, xs, *_), d in zip(rows, data):
        if not censored_groups and d[1] is None:
            continue
        u = w * q
        e = acc.setdefault(g, [0.0, 0.0, 0.0])
        e[0] += u
        e[1] += u * w
        e[2] += u * (ey - cm[xs])
    s2 = f.sigma ** 2
    est = {g: (sr / su, s2 / (su * su / suw)) for g, (su, suw, sr) in acc.items() if su > 0 and suw > 0}
    if f.groups:
        f.eta, f.tau2_b = meta_regression([(f.z[g], th, v) for g, (th, v) in est.items()],
                                          f.tau2_b, f.active, _META_STEPS)
        post, wt = _posterior(f, est, f.eta, f.tau2_b)
        for g in f.b:  # no location data: the prior mean
            f.b[g] = post.get(g, _dot(f.eta, f.z[g]) if f.eta else f.b[g])
        f.wb = wt
    else:
        for g, (th, _) in est.items():
            f.b[g] = th
    su = sse = 0.0
    for w, q, ey, ey2, g, xs, *_ in rows:
        m = cm[xs] + f.b[g]
        su += w * q
        sse += w * q * (ey2 - 2 * m * ey + m * m)
    if su > 0:
        f.sigma = max(math.sqrt(max(sse, 0.0) / su), _SIGMA_MIN)


# ---------------------------------------------------------------- lag distributions

def weekday_shares(obs: list[Obs], ref: int, half_life: float | None) -> list[float] | None:
    """Weekday (Monday first) shares of the real openings known on day `ref`,
    weights halving every `half_life` days of the opening's age; Jeffreys
    estimate (half a count added per weekday) so no weekday gets zero."""
    cnt = [0.5] * 7
    n = 0
    for o in obs:
        if o.t is None or o.r + o.t > min(ref - 1, o.end):
            continue
        day = o.r + o.t
        cnt[(day - 1) % 7] += 0.5 ** ((ref - day) / half_life) if half_life else 1.0
        n += 1
    if not n:
        return None
    tot = sum(cnt)
    return [c / tot for c in cnt]


class Lag:
    """Lag (days from anchor) of a film that opens: F(t) = P(T <= t | opens).
    log(1 + T) ~ N(m, s), or that lag added to a filing-to-rating delay R
    (empirical, {days: probability}). With `cal` = (anchor ordinal, weekday
    shares) each calendar week's mass is spread over its days by the shares."""

    def __init__(self, m: float, s: float, pi: float, delay: dict | None = None, cal=None):
        self.m, self.s, self.pi, self.delay = m, s, pi, delay
        self.cal = None
        if cal and cal[1]:
            cum, acc = [], 0.0
            for v in cal[1]:
                acc += v
                cum.append(acc)
            self.cal = (cal[0], cum)

    def _ln(self, t: float) -> float:
        if t < 0:
            return 0.0
        return N01.cdf((math.log1p(t) - self.m) / self.s)

    def _cont(self, t: float) -> float:
        if self.delay is None:
            return self._ln(t)
        return sum(p * self._ln(t - r) for r, p in self.delay.items())

    def cdf(self, t: float) -> float:
        if self.cal is None:
            return self._cont(t)
        t = math.floor(t)
        if t < 0:
            return 0.0
        anchor, cum = self.cal
        wd = (anchor + t - 1) % 7  # date.fromordinal(n).weekday() == (n - 1) % 7
        ws = t - wd                # lag of the Monday of t's week
        if ws > 0:
            f0 = self._cont(ws - 1)
            return f0 + (self._cont(ws + 6) - f0) * cum[wd]
        w0 = (anchor - 1) % 7      # the anchor's week: only days from the anchor on
        before = cum[w0 - 1] if w0 else 0.0
        span = cum[6] - before
        frac = (cum[wd] - before) / span if span > 0 else 1.0
        return self._cont(ws + 6) * frac


def _quantile(cdf, lo: int, q: float) -> int:
    """Smallest whole day t >= lo with cdf(t) >= q (cdf increasing)."""
    hi, step = lo, 1
    while cdf(hi) < q:
        lo, hi = hi, hi + step
        step *= 2
        if hi > 36500:
            return 36500
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf(mid) >= q:
            hi = mid
        else:
            lo = mid + 1
    return lo


def describe(lag: Lag, anchor: dt.date, elapsed: int, today: dt.date,
             window: tuple[int, int] | None = None, r: float = 1.0,
             p_open: float | None = None) -> dict:
    """Estimate fields of a film not open `elapsed` days after the anchor.
    `window` (days from anchor, inclusive): an announced month; given an
    opening, it falls in the window with probability `r` (spread as the lag
    model inside it) and outside it with 1 - r (as the lag model outside
    it): P(T = t) = q {r g_month(t) + (1 - r) g_outside(t)}. `p_open`: q
    (default: the lag model's P(ever opens | not yet))."""
    f_e = lag.cdf(elapsed - 1) if elapsed > 0 else 0.0
    if p_open is None:
        p_open = lag.pi * (1 - f_e) / (1 - lag.pi * f_e)

    def g_all(t):  # lag given an opening on or after day `elapsed`
        return (lag.cdf(t) - f_e) / (1 - f_e) if f_e < 1 else 1.0
    start = elapsed
    g = g_all
    if window:
        lo, hi = max(window[0], elapsed), window[1]
        f_lo = lag.cdf(lo - 1)
        mass = lag.cdf(hi) - f_lo
        if mass > 1e-12:
            def g_in(t):  # lag given an opening in the window
                return min(1.0, max(0.0, (lag.cdf(t) - f_lo) / mass)) if t < hi else 1.0
        else:  # the window is far in the tail: spread it evenly
            def g_in(t):
                return min(1.0, max(0.0, (t - lo + 1) / (hi - lo + 1)))
        m_in = g_all(hi) - g_all(lo - 1)
        if r >= 1.0 or m_in >= 1 - 1e-9:
            g, start = g_in, lo
        else:
            def g_out(t):
                inside = g_all(min(t, hi)) - g_all(lo - 1) if t >= lo else 0.0
                return min(1.0, max(0.0, (g_all(t) - inside) / (1 - m_in)))

            def g(t):
                return r * g_in(t) + (1 - r) * g_out(t)
    qs = {f"p{int(q * 100)}": _quantile(g, start, q) for q in (0.1, 0.5, 0.8, 0.9)}
    hi_t = _quantile(g, start, 0.999)
    prev = g(start - 1)
    pts = []
    for t in range(start, hi_t + 1):
        cur = g(t)
        pts.append((t, cur - prev))
        prev = cur
    tot = sum(p for _, p in pts) or 1.0
    mean = sum(t * p for t, p in pts) / tot
    var = sum((t - mean) ** 2 * p for t, p in pts) / tot
    this_end = _month_end(today)
    horizons = [this_end, _month_end(this_end + dt.timedelta(days=1)),
                today + dt.timedelta(days=HORIZON_DAYS)]
    by = {}
    for h in sorted(set(horizons)):
        t = (h - anchor).days
        by[h.isoformat()] = round(p_open * g(t), 3) if t >= start else 0.0
    within = {str(d): round(p_open * g(elapsed + d - 1), 3) if elapsed + d - 1 >= start else 0.0
              for d in WITHIN_DAYS}
    # the opening date at 20 evenly spaced levels of the timing given an opening (levels
    # (i + 0.5) / 20): the detail page's quantile dotplot
    q20 = [(anchor + dt.timedelta(days=_quantile(g, start, (i + 0.5) / 20))).isoformat() for i in range(20)]
    return {"p_open": round(p_open, 3), "within": within, "q20": q20,
            "quantiles": {k: (anchor + dt.timedelta(days=v)).isoformat() for k, v in qs.items()},
            "by": by, "sd_days": round(math.sqrt(var), 1)}


# ---------------------------------------------------------------- recalibration
# P(opens) recalibrated on out-of-time predictions as pi* = sigmoid(a + b logit pi)
# (logistic calibration on the logit: it contains the identity a = 0, b = 1; b > 1
# spreads predictions away from the middle, a moves them all), fitted by the
# censored likelihood with each row's lag timing held fixed, so it corrects the
# eventual share and leaves the lag alone. Forms, simplest first; the backtest
# picks one on a later part of the tuning fold and keeps it only if the test
# fold agrees. Why: on the test fold (2026-11-02) rating-day P(open within
# 298 d) ran 0.86 -> 0.91 observed, 0.89 -> 0.94, 0.91 -> 0.97, but 0.66 -> 0.62
# (mean 0.758 vs 0.78). Result that day: the tuning fold picked "none"
# (intercept and slope within one standard error), and a fit on the test
# fold's earlier half (to 2025-01-20) did not beat "none" on its later half
# (intercept a 0.28: mean 0.823 vs 0.797 observed, over-corrected); the later
# half was near calibrated already (0.785 vs 0.797). So nothing is applied;
# every run checks again.
RECAL_FORMS = ("none", "intercept", "slope")
# landmarks (days after the rating) at which the conditional P(opens | none yet)
# is checked on the test fold
LANDMARKS = (30, 60, 90)


def recal_pi(pi: float, ab) -> float:
    if not ab:
        return pi
    a, b = ab
    return min(1 - _EPS, max(_EPS, _sigmoid(a + b * _logit(pi))))


def _recal_rows(pred, rows: list[Obs]) -> list[tuple]:
    """(logit pi, opened, F(t) - F(t - 1) at the opening or F at the last
    check, film weight) per row predicted on its rating day."""
    out = []
    for o, w in zip(rows, _film_weights(rows)):
        lag = pred(o)[0]
        f = lag.cdf(o.t) - lag.cdf(o.t - 1) if o.opened else lag.cdf(o.end - o.r)
        out.append((_logit(lag.pi), o.opened, f, w))
    return out


def _recal_ll(rows: list[tuple], ab) -> list[float]:
    """Per-row log likelihood (film-weighted, see _scaled) under (a, b)."""
    a, b = ab
    k = len(rows) / sum(r[3] for r in rows)
    out = []
    for lg, op, f, w in rows:
        p = _sigmoid(a + b * lg)
        v = p * f if op else 1 - p * f
        out.append(math.log(max(v, 1e-300)) * w * k)
    return out


def fit_recal(rows: list[tuple], form: str):
    """(a, b) of `form` maximizing the likelihood of `rows` (None for "none"):
    a grid over a in [-3, 3] (b in [0.25, 4] for "slope") narrowed around the
    best point (the step halves each round) until the step is below 1e-3."""
    if form == "none" or not rows:
        return None
    lo_a, hi_a = -3.0, 3.0
    lo_b, hi_b = (0.25, 4.0) if form == "slope" else (1.0, 1.0)
    best = (0.0, 1.0)
    while True:
        sa, sb = (hi_a - lo_a) / 8, (hi_b - lo_b) / 8
        grid = [(lo_a + i * sa, lo_b + j * sb) for i in range(9) for j in range(9 if sb else 1)]
        best = max(grid, key=lambda ab: sum(_recal_ll(rows, ab)))
        if max(sa, sb) < 1e-3:
            return round(best[0], 4), round(best[1], 4)
        lo_a, hi_a = best[0] - 2 * sa, best[0] + 2 * sa
        if sb:
            lo_b, hi_b = max(0.05, best[1] - 2 * sb), best[1] + 2 * sb


def recalibrated(pred, ab):
    """`pred` with its P(opens) recalibrated by (a, b)."""
    if not ab:
        return pred

    def p(o: Obs):
        lag, own = pred(o)
        return _with_pi(lag, recal_pi(lag.pi, ab)), own
    return p


def recal_choice(pred, tune: list[Obs], first: dict) -> dict:
    """The form fitted on the earlier half of the tuning fold (by each film's
    first day) and picked on the later half (simplest within one standard
    error of the best), then refitted on the whole fold."""
    if len(tune) < 4:
        return {"form": "none", "ab": None}
    mid = tune[len(tune) // 2].r
    early = [o for o in tune if first[o.film] < mid]
    late = [o for o in tune if first[o.film] >= mid]
    if len(early) < 2 or len(late) < 2:
        return {"form": "none", "ab": None}
    re_, rl = _recal_rows(pred, early), _recal_rows(pred, late)
    scores = [_recal_ll(rl, fit_recal(re_, fm) or (0.0, 1.0)) for fm in RECAL_FORMS]
    form = RECAL_FORMS[_pick(list(RECAL_FORMS), scores)]
    return {"form": form, "ab": fit_recal(_recal_rows(pred, tune), form),
            "loglik_later_half": {fm: round(_mean(s), 4) for fm, s in zip(RECAL_FORMS, scores)}}


def landmark_check(pred_at, test: list[Obs], horizon: int, ab) -> dict:
    """P(open within `horizon` more days | no opening by the landmark) on the
    test rows still unopened at each landmark and followed up that long,
    predicted with what was known on the landmark day, raw and recalibrated."""
    out = {}
    for e in LANDMARKS:
        rows = [o for o in test if not (o.opened and o.t < e) and o.end - o.r >= e + horizon]
        if not rows:
            continue
        pr = pred_at(e)
        raw, cal = [], []
        for o, w in zip(rows, _film_weights(rows)):
            lag = pr(o)[0]
            fe, fh = lag.cdf(e - 1), lag.cdf(e + horizon - 1)
            y = int(o.opened and o.t <= e + horizon - 1)
            for pi, acc in ((lag.pi, raw), (recal_pi(lag.pi, ab), cal)):
                acc.append((pi * (fh - fe) / (1 - pi * fe), y, w))
        out[str(e)] = {"raw": _calibration(raw), "recalibrated": _calibration(cal) if ab else None}
    return out


# ---------------------------------------------------------------- backtest

def maturity_days(obs: list[Obs]) -> int | None:
    """Follow-up after which a row's outcome counts as known: the
    MATURITY_QUANTILE of the real-opening lags of the rows rated in the first
    half of the data's span (followed up for at least half the span, so
    their slow openings are seen)."""
    if not obs:
        return None
    first, last = min(o.r for o in obs), max(o.end for o in obs)
    lags = sorted(o.t for o in obs if o.r <= first + (last - first) / 2 and o.opened)
    if len(lags) < 2:
        return None
    return lags[min(len(lags) - 1, int(math.ceil(MATURITY_QUANTILE * len(lags))) - 1)]


def _cond_ll(lag: Lag, o: Obs, elapsed: int = 0) -> float:
    """Log probability of a row's outcome by whole days (opening on day t,
    or none up to its last check), given no opening before `elapsed`."""
    pi = lag.pi
    base = 1 - pi * lag.cdf(elapsed - 1) if elapsed > 0 else 1.0
    if o.opened:
        p = pi * (lag.cdf(o.t) - lag.cdf(o.t - 1))
    else:
        p = 1 - pi * lag.cdf(o.end - o.r)
    return math.log(max(p, 1e-300)) - math.log(max(base, 1e-300))


def _film_weights(test: list[Obs]) -> list[float]:
    """1 / (ratings of the same film in the set): each film counts once."""
    n: dict = {}
    for o in test:
        n[o.film] = n.get(o.film, 0) + 1
    return [1 / n[o.film] for o in test]


def _scaled(scores: list[float], w: list[float]) -> list[float]:
    """Per-row scores reweighted so their plain mean is the film-weighted mean."""
    k = len(w) / sum(w)
    return [s * wi * k for s, wi in zip(scores, w)]


def _pick(cands: list, scores: list[list[float]]) -> int:
    """Index of the simplest candidate (earliest in `cands`) whose mean test
    log likelihood is within one standard error of the best one's."""
    means = [sum(s) / len(s) for s in scores]
    best = max(range(len(cands)), key=lambda i: means[i])
    for i in range(len(cands)):
        if not _better(scores[best], scores[i]):
            return i
    return best


def _better(new: list[float], old: list[float]) -> bool:
    """`new` beats `old` by more than one standard error of the paired difference."""
    d = [a - b for a, b in zip(new, old)]
    n = len(d)
    if n < 2:
        return False
    md = sum(d) / n
    se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1) / n)
    return md > se


def _wmedian(vals: list[tuple[float, float]]):
    """Weighted median of (value, weight) pairs."""
    srt = sorted(vals)
    half, acc = sum(w for _, w in srt) / 2, 0.0
    for v, w in srt:
        acc += w
        if acc >= half:
            return v
    return None


def _wmean(vals: list[tuple[float, float]]) -> float | None:
    tw = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / tw if tw else None


def _deciles(items: list[tuple[float, int, float]]) -> list[dict]:
    """Reliability: mean prediction vs observed rate in ten equal-count bins."""
    items = sorted(items)
    n = len(items)
    out = []
    for k in range(10):
        part = items[k * n // 10:(k + 1) * n // 10]
        if part:
            out.append({"p": round(_wmean([(p, w) for p, _, w in part]), 3),
                        "observed": round(_wmean([(y, w) for _, y, w in part]), 3), "n": len(part)})
    return out


def _calibration(items: list[tuple[float, int, float]], deciles: bool = True) -> dict:
    """items: (prediction, outcome, film weight)."""
    if not items:
        return {"n": 0}
    out = {"n": len(items), "mean_p": round(_wmean([(p, w) for p, _, w in items]), 3),
           "observed": round(_wmean([(y, w) for _, y, w in items]), 3),
           "brier": round(_wmean([((p - y) ** 2, w) for p, y, w in items]), 4),
           "log_score": round(_wmean([(-math.log(max(p if y else 1 - p, 1e-12)), w)
                                      for p, y, w in items]), 4)}
    if deciles:
        out["deciles"] = _deciles(items)
    return out


def metrics(pred, test: list[Obs], horizon: int, elapsed=None, signal=None) -> dict:
    """Accuracy of `pred(o) -> (Lag, own weight)` on test rows predicted
    `elapsed(o)` days after their anchor (default 0) given no opening by
    then; each film weighted once. P(open within d days) for d in
    WITHIN_DAYS (rows followed up that long) and within the maturity
    `horizon` (every row): Brier, log score, reliability by decile, and by
    company class (established: own data weigh at least half) x signal
    state (`signal(o)`, default: a trailer/poster before the rating);
    integrated Brier over weekly horizons up to `horizon`; censored log
    score; quantiles, p50 error and the opening day's log score over every
    row that opened by its last check. Hit rates of p50's displayed
    intervals (HIT_KEYS) over the openers (`hit_rate`, `hit_films`) and
    over every row, a row without an opening within `horizon` a miss
    (`hit_rate_all`, `hit_films_all`: film-level trials and successes)."""
    signal = signal or (lambda o: o.x[FEATURES.index("promo")])
    wts = _film_weights(test)
    err, inside, day_ll, wkday, cll = [], {50: [], 80: [], 90: []}, [], [], []
    hits: dict = {k: [] for k in HIT_KEYS}      # openers (conditional on opening)
    hits_all: dict = {k: [] for k in HIT_KEYS}  # every row; no opening within horizon = miss
    band, width, cover80 = [], [], []
    cal: dict = {}
    by_d: dict = {}
    ibs: dict = {}
    steps = list(range(7, horizon + 1, 7))
    for o, w in zip(test, wts):
        lag, own = pred(o)
        e = elapsed(o) if elapsed else 0
        f_e = lag.cdf(e - 1) if e > 0 else 0.0
        base = 1 - lag.pi * f_e
        follow = o.end - o.r - e

        def p_within(d):
            return lag.pi * (lag.cdf(e + d - 1) - f_e) / base

        def y_within(d):
            return int(o.opened and o.t <= e + d - 1)
        cls = "established" if own >= 0.5 else "new_or_small"
        sig = "signal" if signal(o) else "no_signal"
        item = (p_within(horizon), y_within(horizon), w)
        if item[0] >= BAND:
            band.append(item)
        for key in ("all", cls):
            cal.setdefault(key, []).append(item)
        for d in WITHIN_DAYS:
            if follow >= d:
                it = (p_within(d), y_within(d), w)
                by_d.setdefault(d, {}).setdefault("all", []).append(it)
                by_d[d].setdefault(f"{cls}/{sig}", []).append(it)
        for d in steps:
            ibs.setdefault(d, []).append(((p_within(d) - y_within(d)) ** 2, w))
        cll.append((_cond_ll(lag, o, e), w))
        if not o.opened:
            for k in HIT_KEYS:
                hits_all[k].append((0.0, w))
            continue

        def g(t):
            return (lag.cdf(t) - f_e) / (1 - f_e) if f_e < 1 else 1.0
        qs = {q: _quantile(g, e, q / 100) for q in (10, 50, 80, 90)}
        for q in (50, 80, 90):
            inside[q].append((float(o.t <= qs[q]), w))
        err.append((abs(qs[50] - o.t), w))
        # the displayed intervals of p50: +/-3 days of the day, its ISO week,
        # its calendar month
        p50_day, open_day = dt.date.fromordinal(o.r + qs[50]), dt.date.fromordinal(o.r + o.t)
        for k, v in (("3", abs(qs[50] - o.t) <= 3),
                     ("week", _iso_week(p50_day) == _iso_week(open_day)),
                     ("month", (p50_day.year, p50_day.month) == (open_day.year, open_day.month))):
            hits[k].append((float(v), w))
            hits_all[k].append((float(v and o.t < horizon), w))
        width.append((qs[90] - qs[10], w))
        cover80.append((float(qs[10] <= o.t <= qs[90]), w))
        wkday.append((float((o.r + qs[50]) % 7 == (o.r + o.t) % 7), w))
        day_ll.append((-math.log(max(g(o.t) - g(o.t - 1), 1e-300)), w))
    out = {"n": len(test), "n_films": round(sum(wts)), "n_opened": len(err)}
    if err:
        srt = sorted(err)
        half, acc, med = sum(w for _, w in srt) / 2, 0.0, None
        for v, w in srt:
            acc += w
            if acc >= half:
                med = v
                break
        out.update({"mae_p50_days": round(_wmean(err), 1), "median_ae_p50_days": med,
                    "coverage": {f"p{q}": round(_wmean(v), 3) for q, v in inside.items()},
                    "day_log_score": round(_wmean(day_ll), 3),
                    "p50_on_opening_weekday": round(_wmean(wkday), 3),
                    "hit_rate": {k: round(_wmean(v), 3) for k, v in hits.items()},
                    "hit_n": len(err), "hit_films": _film_hits(hits),
                    "interval_p10_p90": {"median_width_days": _wmedian(width),
                                         "coverage": round(_wmean(cover80), 3)}})
    if test:
        out["hit_rate_all"] = {k: round(_wmean(v), 3) for k, v in hits_all.items()}
        out["hit_films_all"] = _film_hits(hits_all)
    out["band_0.9_1.0"] = _calibration(band, deciles=False)
    out["censored_log_score"] = round(-_wmean(cll), 4)
    out["integrated_brier"] = round(sum(_wmean(v) for v in ibs.values()) / len(ibs), 4) if ibs else None
    out[f"p_open_{horizon}d"] = {k: _calibration(v) for k, v in cal.items()}
    out["p_open_within"] = {str(d): {k: _calibration(v, deciles=(k == "all"))
                                     for k, v in sorted(parts.items())}
                            for d, parts in sorted(by_d.items())}
    return out


def _film_hits(hits: dict) -> dict:
    """Film-level trials and successes of film-weighted (hit, weight) items:
    each film is one trial, its success the mean over its ratings."""
    first = next(iter(hits.values()), [])
    return {"n": round(sum(w for _, w in first), 3),
            "hits": {k: round(sum(v * w for v, w in items), 3) for k, items in hits.items()}}


def predictor(f: Fit, cal=None, by_distributor: bool = False):
    def pred(o: Obs):
        g = o.gd if by_distributor else o.g
        m, s, pi, _ = f.dist(g, o.x)
        return Lag(m, s, pi, cal=(o.r, cal) if cal else None), f.own_weight(g)
    return pred


def rolling(make, start: int, last: int) -> dict:
    """{origin: fit} for the origins start, start + REFIT_DAYS, ... <= last;
    make(origin, previous fit or None) fits on what was known on the origin
    (warm-started from the previous one)."""
    fits, prev = {}, None
    for org in range(start, last + 1, REFIT_DAYS):
        f = make(org, prev)
        if f is not None:
            fits[org] = prev = f
    return fits


def rolling_predictor(fits: dict, cals: dict | None = None, by_distributor: bool = False,
                      day=None):
    """Predicts a row with the latest fit (and its calendar) on or before
    `day(o)` (default: the row's anchor day)."""
    from bisect import bisect_right
    origins = sorted(fits)
    preds = {org: predictor(fits[org], cals.get(org) if cals else None, by_distributor)
             for org in origins}

    def pred(o: Obs):
        d = day(o) if day else o.r
        return preds[origins[max(0, bisect_right(origins, d) - 1)]](o)
    return pred


def milestone_pairs(obs: list[Obs], kind: str) -> list[tuple[Obs, int, Obs]]:
    """(row, milestone lag, row re-anchored on the milestone) for rows with
    the milestone after the rating and no opening before it."""
    out = []
    for o in obs:
        k = o.ms.get(kind)
        if k is None or o.r + k > o.end or (o.t is not None and o.t < k):
            continue
        out.append((o, k, Obs(o.r + k, None if o.t is None else o.t - k, o.end, o.g, o.x,
                              None, {}, o.token, o.gd, o.film)))
    return out


def _mean(s: list[float]) -> float:
    return round(sum(s) / len(s), 4)


def _first_day(rows: list[Obs]) -> dict:
    """{film: its earliest anchor day} (a film's rows share one fold)."""
    first: dict = {}
    for o in rows:
        if o.film not in first or o.r < first[o.film]:
            first[o.film] = o.r
    return first


def fit_settings(bt: dict) -> dict:
    """fit() keyword arguments of the model a backtest chose."""
    return {"half_life": bt.get("half_life_days"), "feats": tuple(bt.get("features", ())),
            "censored_groups": bt.get("company_location_from", "") != "opened rows"}


def backtest(obs: list[Obs]) -> dict:
    """Nested time split over matured rows (followed up for maturity_days):
    the latest TEST_SHARE is the untouched test fold; the TEST_SHARE before
    it is the tuning fold; a film's rows share the fold of its first rating.
    Half-life, features, company-effect variant, calendar and milestones are
    chosen by fits on what was known on the first tuning day, scored on the
    tuning fold. The chosen model is refitted every REFIT_DAYS from the
    first test day, each time on what was known then, and reported on the
    test fold only (each row predicted by the latest refit on or before its
    rating day)."""
    return _backtest(obs)[0]


def _backtest(obs: list[Obs]) -> tuple[dict, dict | None]:
    horizon = maturity_days(obs)
    if horizon is None:
        return {"n_test": 0, "reason": "too few openings to set the maturity horizon"}, None
    eligible = sorted((o for o in obs if o.end - o.r >= horizon), key=lambda o: o.r)
    n = len(eligible)
    if n < 8:
        return {"n_test": 0, "maturity_days": horizon,
                "reason": f"fewer than 8 rows followed up for {horizon} days"}, None
    cut1 = eligible[int(n * (1 - 2 * TEST_SHARE))].r
    cut2 = eligible[int(n * (1 - TEST_SHARE))].r
    first = _first_day(obs)
    tune = [o for o in eligible if cut1 <= o.r < cut2 and first[o.film] >= cut1]
    test = [o for o in eligible if o.r >= cut2 and first[o.film] >= cut2]
    h_order = [None] + sorted((h for h in H_GRID if h), reverse=True)  # simplest first
    fits_h = [fit(obs, cut1, h_order[0])]
    # single-origin comparisons start every fit cold, so two candidates are
    # not separated by how far their EM ran (a warm start ends nearer the
    # fixed point and would win by noise)
    if fits_h[0] is None or len(tune) < 2 or len(test) < 2:
        return {"n_test": len(test), "maturity_days": horizon,
                "reason": "too few rows before or after the split"}, None
    for hh in h_order[1:]:
        fits_h.append(fit(obs, cut1, hh))
    tw = _film_weights(tune)

    def ll(fi, cal=None):
        pr = predictor(fi, cal)
        return _scaled([_cond_ll(pr(o)[0], o) for o in tune], tw)

    scores = [ll(f) for f in fits_h]
    hi = _pick(h_order, scores)
    cfg = {"half_life": h_order[hi], "feats": (), "censored_groups": True}
    cur_fit, cur_ll = fits_h[hi], scores[hi]
    gain = {}
    for feat in FEATURES:
        c = {**cfg, "feats": cfg["feats"] + (feat,)}
        cf = fit(obs, cut1, **c)
        c_ll = ll(cf)
        gain[feat] = round(_mean(c_ll) - _mean(cur_ll), 4)
        if _better(c_ll, cur_ll):
            cfg, cur_fit, cur_ll = c, cf, c_ll
    # company location effects from opened rows only (the previous model):
    # kept only if the tuning fold prefers it
    oo_fit = fit(obs, cut1, **{**cfg, "censored_groups": False})
    oo_ll = ll(oo_fit)
    if _better(oo_ll, cur_ll):
        cfg, cur_fit, cur_ll = {**cfg, "censored_groups": False}, oo_fit, oo_ll
    cals = [weekday_shares(obs, cut1, hw) for hw in h_order]
    cal_scores = [ll(cur_fit, c) for c in cals]
    wi = _pick(h_order, cal_scores)
    use_cal = _better(cal_scores[wi], cur_ll)
    cal_h = h_order[wi] if use_cal else None

    def cal_at(ref):
        return weekday_shares(obs, ref, cal_h) if use_cal else None

    ms_used = {k: _milestone_choice(obs, k, cut1, cut2, horizon, cfg, cur_fit, cal_at(cut1))
               for k in MILESTONES}
    first_t = _first_day(obs)
    rc = recal_choice(predictor(cur_fit, cal_at(cut1)), tune, first_t)

    # the untouched test fold: refits every REFIT_DAYS from its first day
    last = test[-1].r
    fits = rolling(lambda org, prev: fit(obs, org, init=prev or cur_fit, **cfg), cut2, last)
    cals_t = {org: cal_at(org) for org in fits} if use_cal else None
    overall = rolling(lambda org, prev: fit(obs, org, groups=False, init=prev, **cfg), cut2, last)
    single = fits[min(fits)]
    other = fit(obs, cut2, **{**cfg, "censored_groups": not cfg["censored_groups"]})
    by_dist = fit([o.regroup(o.gd) for o in obs], cut2, **cfg)
    shares = cals_t[min(fits)] if cals_t else None
    raw_pred = rolling_predictor(fits, cals_t)
    model_pred = recalibrated(raw_pred, rc["ab"])
    overall_pred = rolling_predictor(overall, cals_t)
    xw = _film_weights(test)

    def test_ll(pr):
        return _mean(_scaled([_cond_ll(pr(o)[0], o) for o in test], xw))
    censored = cfg["censored_groups"]
    out = {"n_test": len(test), "n_test_opened": sum(o.opened for o in test),
           "maturity_days": horizon, "maturity_quantile": MATURITY_QUANTILE,
           "tune_from": dt.date.fromordinal(cut1).isoformat(),
           "test_from": dt.date.fromordinal(cut2).isoformat(),
           "test_to": dt.date.fromordinal(last).isoformat(),
           "refit_days": REFIT_DAYS, "test_refits": len(fits),
           "half_life_days": cfg["half_life"], "features": list(cfg["feats"]),
           "company_location_from": "all rows (censored by EM)" if censored else "opened rows",
           "tuning": {"loglik_by_half_life": {str(hh): _mean(s) for hh, s in zip(h_order, scores)},
                      "feature_gain": gain, "loglik_calendar": _mean(cal_scores[wi]),
                      "loglik_company_location_opened_rows_only": _mean(oo_ll),
                      "loglik_no_calendar": _mean(cur_ll)},
           "calendar": {"weekday": use_cal, "half_life_days": cal_h},
           "test_loglik": {"model": test_ll(model_pred),
                           "model_single_refit": test_ll(predictor(single, shares)),
                           "overall": test_ll(overall_pred),
                           ("company_location_opened_rows_only" if censored else
                            "company_location_all_rows"): test_ll(predictor(other, shares)),
                           "by_distributor_known_at_rating": test_ll(predictor(by_dist, shares, True))},
           "model": metrics(model_pred, test, horizon),
           "overall": metrics(overall_pred, test, horizon),
           "recalibration": _recal_report(rc, raw_pred, model_pred, test, horizon, xw,
                                          lambda e: rolling_predictor(fits, cals_t, day=lambda o: o.r + e)),
           "by_distributor_known_at_rating": metrics(predictor(by_dist, shares, True), test, horizon)}
    out["milestones"] = {k: _milestone_report(obs, k, cut2, last, horizon, cfg, fits, cals_t, used)
                         for k, used in ms_used.items()}
    if rc["ab"]:
        out["model_uncalibrated"] = metrics(raw_pred, test, horizon)
    return out, {"fits": fits, "cals": cals_t, "pred": model_pred, "cut2": cut2, "horizon": horizon}


def _recal_report(rc: dict, raw_pred, model_pred, test: list[Obs], horizon: int, xw: list[float],
                  pred_at) -> dict:
    """The chosen form, its test-fold effect (log likelihood, P(open within
    `horizon`) calibration), the landmark checks, and the (a, b) for the live
    model: the form refitted on the test fold's out-of-time predictions (the
    latest ones), used only if the tuning-fold fit did not lower the test
    fold's log likelihood."""
    def ll(pr):
        return round(_mean(_scaled([_cond_ll(pr(o)[0], o) for o in test], xw)), 4)

    def cal(pr):
        items = []
        for o, w in zip(test, xw):
            lag = pr(o)[0]
            items.append((lag.pi * lag.cdf(horizon - 1), int(o.opened and o.t <= horizon - 1), w))
        return _calibration(items)
    ab = rc["ab"]
    t_raw, t_cal = ll(raw_pred), ll(model_pred)
    used = bool(ab) and t_cal >= t_raw
    live = fit_recal(_recal_rows(raw_pred, test), rc["form"]) if used else None
    return {"form": rc["form"], "tuning_fit": list(ab) if ab else None,
            "loglik_later_half": rc.get("loglik_later_half"),
            "test_loglik": {"raw": t_raw, "recalibrated": t_cal},
            f"p_open_{horizon}d": {"raw": cal(raw_pred), "recalibrated": cal(model_pred) if ab else None},
            "landmarks": landmark_check(pred_at, test, horizon, ab),
            "used": used, "live": list(live) if live else None}


def _milestone_tests(pairs: list, start: int, stop: int | None, horizon: int) -> list:
    """Pairs whose milestone day lies in [start, stop) with `horizon` days of
    follow-up from the milestone (where they are scored), a film's pairs in
    the fold of its first milestone day."""
    first = _first_day([d for _, _, d in pairs])
    return [(o, k, d) for o, k, d in pairs
            if start <= d.r and (stop is None or d.r < stop) and first[d.film] >= start
            and d.end - d.r >= horizon]


def _milestone_choice(obs, kind, cut1, cut2, horizon, cfg, rating_fit, shares) -> dict:
    """On the tuning fold: does the model anchored on the milestone beat the
    rating model given no opening by the milestone day?"""
    pairs = milestone_pairs(obs, kind)
    tune = _milestone_tests(pairs, cut1, cut2, horizon)
    mfit = fit([d for _, _, d in pairs], cut1, **cfg) if len(tune) >= 2 else None
    if mfit is None:
        return {"used": False, "n_tune": len(tune)}
    w = _film_weights([d for _, _, d in tune])
    rp, mp = predictor(rating_fit, shares), predictor(mfit, shares)
    a = _scaled([_cond_ll(rp(o)[0], o, k) for o, k, _ in tune], w)
    b = _scaled([_cond_ll(mp(d)[0], d) for _, _, d in tune], w)
    return {"used": _better(b, a), "n_tune": len(tune),
            "tune_loglik": {"rating_model": _mean(a), "milestone_model": _mean(b)}}


def _milestone_report(obs, kind, cut2, last, horizon, cfg, rating_fits, cals, choice) -> dict:
    """Test-fold accuracy, at the milestone day, of both models (each
    refitted every REFIT_DAYS, a row predicted by the latest fit on or
    before its milestone day)."""
    pairs = milestone_pairs(obs, kind)
    test = _milestone_tests(pairs, cut2, None, horizon)
    out = {"rows": len(pairs), **choice, "n_test": len(test)}
    if len(test) < 2:
        return out
    rows = [d for _, _, d in pairs]
    mfits = rolling(lambda org, prev: fit(rows, org, init=prev, **cfg), cut2,
                    max(last, max(d.r for _, _, d in test)))
    if not mfits:
        return out
    lag_of = {id(o): k for o, k, _ in test}
    rp = rolling_predictor(rating_fits, cals, day=lambda o: o.r + lag_of[id(o)])
    cal_m = None
    if cals:
        org_c = sorted(cals)
        from bisect import bisect_right
        cal_m = {org: cals[org_c[max(0, bisect_right(org_c, org) - 1)]] for org in mfits}
    mp = rolling_predictor(mfits, cal_m)

    def always(o):  # every row has the milestone
        return 1
    out.update({"observed_open_rate": round(sum(d.opened for _, _, d in test) / len(test), 3),
                "milestone_model": metrics(mp, [d for _, _, d in test], horizon, signal=always),
                "rating_model": metrics(rp, [o for o, _, _ in test], horizon,
                                        elapsed=lambda o: lag_of[id(o)], signal=always)})
    return out


# ---------------------------------------------------------------- apply

def _counts(days: list[int]) -> dict:
    return {d: days.count(d) for d in sorted(set(days))}


def _delays(lags: dict, films: list[dict]) -> dict:
    """Days from filing to rating (rows with both dates and current records)
    and from the first trailer/poster rating to the film rating (current
    records; negative when the film was rated first)."""
    seen, receipt = set(), []
    pairs = [(r.get("rating_no"), r.get("receipt_date"), r.get("rating_date")) for r in lags.get("rows", [])]
    pairs += [(f["stages"].get("rating", {}).get("rating_no"), f["stages"].get("intake", {}).get("date"),
               f["stages"].get("rating", {}).get("date")) for f in films]
    for no, rc, rt in pairs:
        rc, rt = _valid_day(rc), _valid_day(rt)
        if rc and rt and rt >= rc and no not in seen:
            seen.add(no)
            receipt.append((_d(rt) - _d(rc)).days)
    promo = []
    for f in films:
        pd = _valid_day(f["stages"].get("promo", {}).get("first_date"))
        rt = _valid_day(f["stages"].get("rating", {}).get("date"))
        if pd and rt:
            promo.append((_d(rt) - _d(pd)).days)
    return {"receipt": _counts(receipt), "promo": _counts(promo)}


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def model_summary(f: Fit) -> dict:
    """Coefficients of a fit (log-lag location and logit of P(opens))."""
    out = {"features": list(f.feats), "sigma_log": round(f.sigma, 3),
           "beta": {k: round(v, 3) for k, v in zip(f.feats, f.beta)},
           "gamma": {k: round(v, 3) for k, v in zip(f.feats, f.gamma)},
           "companies": len(f.n_rows), "em_iterations": f.iterations}
    if f.groups:
        names = ["intercept", *GROUP_FEATURES]
        out.update({"company_prior_location": {k: _r(v) for k, v in zip(names, f.eta or [])},
                    "company_prior_logit": {k: _r(v) for k, v in zip(names, f.delta or [])},
                    "tau2_location": round(f.tau2_b, 4), "tau2_logit": round(f.tau2_a, 4),
                    "icc": round(f.tau2_b / (f.tau2_b + f.sigma ** 2), 3),
                    "k_rows": _r(f.k_rows if f.k_rows < math.inf else None, 1),
                    "companies_established": sum(n >= f.k_rows for n in f.count.values())})
    return out


def segment_rates(obs: list[Obs], ref: int, horizon: int | None) -> dict:
    """{segment: {"n", "opened", "rate"}} over the rows of each segment rated
    before day `ref` and followed up (up to ref) for at least `horizon` days:
    the share that opened within `horizon` days (Jeffreys estimate, half a
    count added to each outcome)."""
    acc: dict = {}
    for o in obs:
        if not o.seg or o.r >= ref:
            continue
        follow = min(ref, o.end) - o.r
        if horizon is not None and follow < horizon:
            continue
        e = acc.setdefault(o.seg, [0, 0])
        e[0] += 1
        e[1] += int(o.t is not None and o.r + o.t <= min(ref, o.end)
                    and (horizon is None or o.t < horizon))
    return {s: {"n": n, "opened": k, "rate": round((k + 0.5) / (n + 1), 4)}
            for s, (n, k) in sorted(acc.items())}


def evaluate(obs: list[Obs]) -> dict:
    """Backtest of the theatrical model on the rows outside every segment,
    plus the whole population on the same untouched test fold: segment rows
    predicted by their segment's share, the rest by the model, both as known
    on the latest refit day on or before the row's rating (P(open within the
    maturity horizon))."""
    from bisect import bisect_right
    cand = [o for o in obs if not o.seg]
    bt, ctx = _backtest(cand)
    if ctx is None:
        return bt
    horizon, cut2 = ctx["horizon"], ctx["cut2"]
    origins = sorted(ctx["fits"])
    rates_at = {org: segment_rates(obs, org, horizon) for org in origins}
    rates = rates_at[origins[0]]
    pr = ctx["pred"]
    first = _first_day(obs)
    test = [o for o in obs if o.r >= cut2 and o.end - o.r >= horizon and first[o.film] >= cut2]
    wts = _film_weights(test)
    items, seg_items = [], {}
    for o, w in zip(test, wts):
        y = int(o.opened and o.t <= horizon - 1)
        if o.seg:
            r_o = rates_at[origins[max(0, bisect_right(origins, o.r) - 1)]]
            p = r_o.get(o.seg, {}).get("rate", 0.0)
            seg_items.setdefault(o.seg, []).append((p, y, w))
        else:
            lag = pr(o)[0]
            p = lag.pi * lag.cdf(horizon - 1)
        items.append((p, y, w))
    bt["segments"] = {s: {**rates.get(s, {}), "test": _calibration(v, deciles=False)}
                      for s, v in sorted(seg_items.items())}
    bt["population"] = {f"p_open_{horizon}d": _calibration(items)}
    return bt


# ---------------------------------------------------------------- signal states and tiers

# Signal state of a record, strongest first: opened (a source day that has
# passed), bookable (tickets on sale), day_announced (a future source day:
# KOBIS, a chain, the KOBIS schedule or a chain program), month_announced,
# ad_filing (an ORS advertising filing, `stages.ors_ad`), promo_after_rating
# (a trailer/poster rating after the film rating), rating_only, filed (not
# rated yet).
STATES = ("opened", "bookable", "day_announced", "month_announced", "ad_filing",
          "promo_after_rating", "rating_only", "filed")
# states without a source date: a segment record in one of them gets its
# segment's opening share and the estimate tier; in any other state a source
# dates the film and it is judged like every other film
SEGMENT_STATES = ("ad_filing", "promo_after_rating", "rating_only", "filed")
# Display tiers, maintainer's acceptance criteria 2026-11-01: confirmed when
# the state's hit rate within +/-3 days is at least 0.90, likely (the ISO
# week) inside the displayed week at least 0.80, expected (the month) inside
# the month at least 0.80; else estimate (p10..p90 months and P(open)). A
# rate counts by the lower end of its Wilson score interval over films (one
# trial per film), so a small history grants no tier.
TIER_GATES = (("confirmed", "3", 0.90), ("likely", "week", 0.80), ("expected", "month", 0.80))
# what each hit counts: the opening inside the interval the tier displays,
# except `3`: a confirmed tier shows one day but is gated with +/-3 days of
# tolerance (the maintainer's criterion), not on the exact day
HIT_KEYS = ("3", "week", "month")
HIT_RULES = {"3": "opening within +/-3 days of the displayed day",
             "week": "opening in the displayed ISO week",
             "month": "opening in the displayed calendar month"}
# z of the Wilson interval: its lower end is the two-sided 95% bound
WILSON_Z = 1.96
# a planned day or month is judged once it lies this many days in the past:
# an opening in the planned day's ISO week is at most 6 days after it, so a
# late opening that still hits a day gate is seen
GRACE_DAYS = 7


def wilson_lower(k: float, n: float, z: float = WILSON_Z) -> float:
    """Lower end of the Wilson score interval of k successes in n."""
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / den)


def _tier(n: float, hits: dict) -> tuple[str, dict]:
    """(tier, Wilson lower bounds rounded for display); the gates compare
    the unrounded bounds."""
    raw = {k: wilson_lower(hits.get(k, 0), n) for k in HIT_KEYS}
    lower = {k: round(v, 3) for k, v in raw.items()}
    for name, key, gate in TIER_GATES:
        if raw[key] >= gate:
            return name, lower
    return "estimate", lower


def _kobis_open(rec: dict, srcs: dict, today_s: str, planned: str) -> str:
    """The KOBIS opening (first or re-release) of the current cycle on or
    before today nearest to a planned day or month, else ""."""
    start = cycle_start(rec)
    ds = [d for d in [srcs.get("kobis", ""), srcs.get("kobis_api", ""), *srcs.get("kobis_rereleases", []),
                      srcs.get("release_lags", "")]
          if d and d <= today_s and (not start or d >= start)]
    if not ds:
        return ""
    ref = _d(planned if len(planned) == 10 else planned + "-01")
    return min(ds, key=lambda d: abs((_d(d) - ref).days))


def _judge(planned: str, opened: str) -> dict:
    """Outcome of one planned day or month against the opening ("" if none)."""
    out = {"n": 1, "opened": int(bool(opened)), "3": 0, "week": 0, "month": 0}
    if not opened:
        return out
    out["month"] = int(opened[:7] == planned[:7])
    if len(planned) == 10:
        out["3"] = int(abs((_d(opened) - _d(planned)).days) <= 3)
        out["week"] = int(_iso_week(_d(opened)) == _iso_week(_d(planned)))
    return out


def _add(acc: dict, key: str, res: dict) -> None:
    e = acc.setdefault(key, {"n": 0, "opened": 0, "3": 0, "week": 0, "month": 0})
    for k, v in res.items():
        e[k] += v


# Snapshot sources whose passed dates are judged by KOBIS's recorded opening:
# KOBIS presale (booking), Lotte and CGV. Left out: the KOBIS schedule (the
# same registry as the judge) and Megabox, which lists event and classic
# screenings KOBIS does not record (29 of the 76 Megabox listings dated on or
# before 2026-11-01 had no KOBIS opening that day).
_SNAPSHOT_DAY = {"booking": "bookable", "booking_rerelease": "bookable",
                 "lotte": "day_announced", "cgv": "day_announced"}
_SNAPSHOT_MONTH = ("lotte_month", "cgv_month")


def _row_film(row: dict) -> str:
    """Film key of a release_lags row: its KOBIS code, else its title."""
    return row.get("kobis_cd") or "t:" + norm_key(split_format(row.get("title", ""))[0])


def _rec_film(rec: dict) -> str:
    """Film key of a record, as _row_film."""
    st = rec["stages"]
    return (st.get("kobis", {}).get("movie_cd") or st.get("release_history", {}).get("kobis_cd")
            or "t:" + norm_key(split_format(rec.get("title", ""))[0]))


def commitment_counts(films: list[dict], lags: dict, today: dt.date) -> dict:
    """Outcomes of announced openings per state ({"history": ..., "snapshot":
    ..., "films": ...}, each {state: {n, opened, 3, week, month}}).

    history: release_lags rows whose first planned opening (`planned_first`,
    a day or a month, recorded with the day it was seen) lies at least
    GRACE_DAYS in the past: delivered if the row has an opening.
    snapshot: current records whose booking / Lotte / CGV day or month lies
    at least GRACE_DAYS in the past, judged by the KOBIS opening of the cycle
    (survivor-biased: chains drop listings that were cancelled).
    films: history and snapshot together with one trial per film and state
    (the mean of its announcements' outcomes), for the tiers' Wilson bounds."""
    today_s = today.isoformat()
    cut = (today - dt.timedelta(days=GRACE_DAYS)).isoformat()
    judged: dict = {}  # (state, film) -> [outcome, ...]

    def add(acc, state, film, res):
        _add(acc, state, res)
        judged.setdefault((state, film), []).append(res)

    rows = {r.get("rating_no"): r for r in lags.get("rows", []) if r.get("rating_no")}
    hist: dict = {}
    for row in lags.get("rows", []):
        pf = row.get("planned_first", "")
        if len(pf) == 10 and _valid_day(pf) and pf <= cut:
            add(hist, "day_announced", _row_film(row), _judge(pf, _valid_day(row.get("open_dt"))))
        elif len(pf) == 7 and _valid_month(pf) and _month_end(_d(pf + "-01")).isoformat() <= cut:
            add(hist, "month_announced", _row_film(row), _judge(pf, _valid_day(row.get("open_dt"))))
    snap: dict = {}
    for rec in films:
        if "dormant" in rec.get("tags", []):
            continue
        srcs = sources(rec, rating_row(rec, rows))
        start = cycle_start(rec)
        film = _rec_film(rec)
        for key, state in _SNAPSHOT_DAY.items():
            d = srcs.get(key, "")
            if d and d <= cut and (not start or d >= start):
                add(snap, state, film, _judge(d, _kobis_open(rec, srcs, today_s, d)))
        for key in _SNAPSHOT_MONTH:
            m = srcs.get(key, "")
            if m and _month_end(_d(m + "-01")).isoformat() <= cut and (not start or m >= start[:7]):
                add(snap, "month_announced", film, _judge(m, _kobis_open(rec, srcs, today_s, m)))
    per_film: dict = {}
    for (state, _), outcomes in judged.items():
        _add(per_film, state, {k: (1 if k == "n" else sum(o[k] for o in outcomes) / len(outcomes))
                               for k in outcomes[0]})
    return {"history": hist, "snapshot": snap, "films": per_film}


def commitment_rates(counts: dict) -> dict:
    """q (P(opens)), r (P(on time | opens): +/-3 days, the ISO week, the
    month) per announced state from history and snapshot counts together;
    `films`: the same counts with one trial per film (for the tiers). A
    state without counts borrows: bookable and month_announced from
    day_announced (a day announcement fixes its month too). `provisional`
    while no history row has matured."""
    out = {}
    for state in ("bookable", "day_announced", "month_announced"):
        tot = {"n": 0, "opened": 0, **{k: 0 for k in HIT_KEYS}}
        for part in ("history", "snapshot"):
            for k, v in counts.get(part, {}).get(state, {}).items():
                tot[k] += v
        fc = counts.get("films", {}).get(state) or tot
        out[state] = {"n": tot["n"], "opened": tot["opened"],
                      "hits": {k: tot[k] for k in HIT_KEYS},
                      "films": {"n": round(fc["n"], 3), "opened": round(fc["opened"], 3),
                                "hits": {k: round(fc[k], 3) for k in HIT_KEYS}},
                      "history_n": counts.get("history", {}).get(state, {}).get("n", 0),
                      "snapshot_n": counts.get("snapshot", {}).get(state, {}).get("n", 0)}

    def month_only(hits):
        return {**{k: 0 for k in HIT_KEYS}, "month": hits["month"]}
    for state in ("bookable", "month_announced"):
        if not out[state]["n"] and out["day_announced"]["n"]:
            day = out["day_announced"]
            hits = dict(day["hits"]) if state == "bookable" else month_only(day["hits"])
            fhits = dict(day["films"]["hits"]) if state == "bookable" else month_only(day["films"]["hits"])
            out[state] = {**day, "hits": hits, "films": {**day["films"], "hits": fhits},
                          "borrowed_from": "day_announced", "history_n": 0, "snapshot_n": 0}
    for e in out.values():
        n, k = e["n"], e["opened"]
        e["q"] = round((k + 0.5) / (n + 1), 4) if n else None
        e["r"] = {h: round((v + 0.5) / (k + 1), 4) for h, v in e["hits"].items()} if k else {}
        e["provisional"] = e.get("history_n", 0) == 0
    return out


def tier_table(bt: dict, rates: dict, lags: dict) -> dict:
    """Hit rates, their Wilson lower bounds and the granted tier per state,
    with one trial per film (n: films). `rate` / `lower` count a film that
    did not open (model states: within the maturity horizon) as a miss;
    `conditional` gives the same over the films that opened. Model states:
    the backtest's p50 against the opening over matured test-fold rows.
    Announced states: the announced day / month against the KOBIS opening.
    opened: KOBIS openings of release_lags rows with box-office data, a hit
    when the film had screenings."""
    out: dict = {}

    def rated(n, hits):
        return {k: round(hits.get(k, 0) / n, 3) if n else None for k in HIT_KEYS}

    def put(state, n, hits, basis, provisional=False, cond=None):
        tier, lower = _tier(n, hits)
        out[state] = {"n": _r(n, 1), "rate": rated(n, hits), "lower": lower, "tier": tier,
                      "basis": basis, "provisional": provisional}
        if cond is not None:
            cn, ch = cond
            out[state]["conditional"] = {"n": _r(cn, 1), "rate": rated(cn, ch), "lower": _tier(cn, ch)[1]}

    per_film: dict = {}
    for r in lags.get("rows", []):
        sc = r.get("screenings")
        if _valid_day(r.get("open_dt")) and isinstance(sc, int) and not isinstance(sc, bool):
            per_film.setdefault(_row_film(r), []).append(sc > 0)
    k = sum(sum(v) / len(v) for v in per_film.values())
    hits = {key: k for key in HIT_KEYS}
    put("opened", len(per_film), hits, "release_lags openings with screenings",
        cond=(len(per_film), hits))
    for state in ("bookable", "day_announced", "month_announced"):
        e = rates.get(state, {})
        f = e.get("films") or {"n": e.get("n", 0), "opened": e.get("opened", 0), "hits": e.get("hits", {})}
        put(state, f["n"], f["hits"], "announced vs KOBIS opening"
            + (f" (borrowed from {e['borrowed_from']})" if e.get("borrowed_from") else ""),
            e.get("provisional", True), cond=(f["opened"], f["hits"]))
        out[state]["n_announcements"] = e.get("n", 0)
    ms = bt.get("milestones", {}).get("promo", {})
    for state, m, basis in (
            ("rating_only", bt.get("model", {}), "backtest: rating model p50"),
            ("promo_after_rating", ms.get("milestone_model" if ms.get("used") else "rating_model", {}),
             "backtest: milestone p50")):
        fa, fo = m.get("hit_films_all", {}), m.get("hit_films", {})
        put(state, fa.get("n", 0), fa.get("hits", {}), basis,
            cond=(fo.get("n", 0), fo.get("hits", {})))
    for state, basis in (("ad_filing", "no history yet"), ("filed", "no backtest")):
        put(state, 0, {}, basis, True)
    return out


def state_report(bt: dict, rates: dict, tiers: dict) -> dict:
    """Accuracy by signal state: horizon Brier and the 0.9-1.0 band of
    P(open within the maturity horizon), p10..p90 width at its achieved
    coverage and +/-3 day, ISO week and month hit rates (model states: matured
    test fold; announced states: announced day / month against the KOBIS
    opening, `provisional` while only the snapshot measures them)."""
    h = bt.get("maturity_days")
    out = {}
    ms = bt.get("milestones", {}).get("promo", {})
    for state, m in (("rating_only", bt.get("model", {})),
                     ("promo_after_rating", ms.get("milestone_model" if ms.get("used") else "rating_model", {}))):
        if not m:
            continue
        cal = m.get(f"p_open_{h}d", {}).get("all", {})
        out[state] = {"n": m.get("n"), "n_opened": m.get("n_opened"),
                      "brier": cal.get("brier"), "log_score": cal.get("log_score"),
                      "mean_p": cal.get("mean_p"), "observed": cal.get("observed"),
                      "band_0.9_1.0": m.get("band_0.9_1.0"), "interval_p10_p90": m.get("interval_p10_p90"),
                      "hit_rate": m.get("hit_rate"), "hit_rate_all": m.get("hit_rate_all"),
                      "tier": tiers.get(state, {}).get("tier")}
    for state in ("bookable", "day_announced", "month_announced"):
        e = rates.get(state, {})
        n = e.get("n", 0)
        out[state] = {"n": n, "opened": e.get("opened"),
                      "brier": round(((1 - e["q"]) ** 2 * e["opened"] + e["q"] ** 2 * (n - e["opened"])) / n, 4)
                      if n and e.get("q") is not None else None,
                      "hit_rate": {k: round(v / n, 3) for k, v in e.get("hits", {}).items()} if n else {},
                      "tier": tiers.get(state, {}).get("tier"), "provisional": e.get("provisional", True),
                      **({"borrowed_from": e["borrowed_from"]} if e.get("borrowed_from") else {})}
    out["ad_filing"] = {"n": 0, "tier": tiers.get("ad_filing", {}).get("tier"),
                        "note": "no history yet: estimated as its rating / milestone state"}
    return out


def _iso_week(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def display(tier: str, state: str, p_open, date: str = "", quantiles: dict | None = None) -> dict:
    """The `display` value: tier, the date at the tier's precision, P(open),
    the signal state, `validated`. `date`: the announced day / month or the
    p50 day. A tier's date and the p10..p90 months are backed by measured
    hit rates / coverage (`validated` true); an announced date whose state
    passed no tier, without model quantiles, is shown as announced with
    `validated` false."""
    text, validated = "", True
    if tier == "confirmed" and len(date) == 10:
        text = date
    elif tier == "likely" and len(date) == 10:
        text = _iso_week(_d(date))
    elif tier in ("likely", "expected") and date:
        text = date[:7]
    elif quantiles and quantiles.get("p10") and quantiles.get("p90"):
        text = f"{quantiles['p10'][:7]}..{quantiles['p90'][:7]}"
    elif date:
        text, validated = date, False
    if tier == "likely" and len(date) != 10:
        tier = "expected"
    return {"tier": tier, "text_date": text, "p_open": p_open, "basis_state": state,
            "validated": validated and bool(text)}


def signal_state(rec: dict, conf, today_s: str, anchor_kind: str) -> str:
    """The record's signal state (STATES)."""
    st = rec["stages"]
    if conf and conf[1] == "day":
        if conf[0] <= today_s:
            return "opened"
        return "bookable" if st.get("booking") else "day_announced"
    if conf:
        return "month_announced"
    if st.get("ors_ad"):
        return "ad_filing"
    if anchor_kind.endswith("_after_rating"):
        return "promo_after_rating"
    if anchor_kind == "rating":
        return "rating_only"
    return "filed"


# ---------------------------------------------------------------- apply

def _fit_target(obs: list[Obs], bt: dict, ref: int, milestones: bool):
    cfg = fit_settings(bt)
    model = fit(obs, ref, **cfg)
    ms_meta = {}
    if model is None:
        return None, ms_meta
    cal = bt.get("calendar", {})
    if cal.get("weekday"):
        model.cal = weekday_shares(obs, ref, cal.get("half_life_days"))
    for kind in MILESTONES if milestones else ():
        pairs = milestone_pairs(obs, kind)
        mb = bt.get("milestones", {}).get(kind, {})
        ms_meta[kind] = {"rows": len(pairs), "used": bool(mb.get("used")),
                         "observed_open_rate": mb.get("observed_open_rate")}
        if mb.get("used"):
            mf = fit([d for _, _, d in pairs], ref, **cfg)
            if mf:
                model.milestones[kind] = mf
    return model, ms_meta


def apply(films: list[dict], lags: dict, today: dt.date) -> dict:
    """Set `release` on every record and add the current records to `lags`
    (the content of data/release_lags.json), both in place; returns the
    model's meta entry. Two outcomes: any theatrical opening (`p_open`, the
    dates) and a real one (`p_open_real`, token releases counted as none)."""
    from .theatrical import applicant_history
    acc = accumulate(lags, films, today)
    tracked = {f["stages"]["rating"]["rating_no"] for f in films
               if f["stages"].get("rating", {}).get("rating_no")}
    rows = {r.get("rating_no"): r for r in lags.get("rows", []) if r.get("rating_no")}
    scheduled = {}
    for f in films:
        no = f["stages"].get("rating", {}).get("rating_no")
        conf = confirmed(f, sources(f, rating_row(f, rows)))
        if no and conf and conf[1] == "day" and conf[0] > today.isoformat():
            scheduled[no] = conf[0]
    obs = {t: observations(lags, tracked, today, scheduled, t) for t in ("any", "real")}
    bts = {t: evaluate(obs[t]) for t in ("any", "real")}
    bt = bts["any"]
    ref = today.toordinal() + 1
    cand = {t: [o for o in obs[t] if not o.seg] for t in obs}
    model, ms_meta = _fit_target(cand["any"], bt, ref, True)
    model_real, _ = _fit_target(cand["real"], bts["real"], ref, False)
    # recalibrated P(opens) of the rating models (milestone fits stay as fitted)
    for mdl, b_ in ((model, bt), (model_real, bts["real"])):
        live = (b_.get("recalibration") or {}).get("live")
        if mdl and live:
            mdl.recal = tuple(live)
    overall = fit(cand["any"], ref, **{**fit_settings(bt), "groups": False})
    if model:
        model.booking_seen = {r["rating_no"]: r["booking_seen"] for r in lags.get("rows", [])
                              if r.get("booking_seen") and r.get("rating_no")}
    delays = _delays(lags, films)
    rates = commitment_rates(commitment_counts(films, lags, today))
    tiers = tier_table(bt, rates, lags)
    seg_rates = {t: segment_rates(obs[t], ref, bts[t].get("maturity_days")) for t in obs}
    ctx = {"real": model_real, "rates": rates, "tiers": tiers, "seg_rates": seg_rates,
           "history": applicant_history(lags, today), "rows": rows}

    counts: dict = {}
    shown: dict = {}
    for rec in films:
        rel = estimate(rec, today, model, delays, ctx)
        if rel is None:
            rec.pop("release", None)
            continue
        rec["release"] = rel
        counts[rel["basis"]] = counts.get(rel["basis"], 0) + 1
        key = f"{rel['display']['basis_state']}/{rel['display']['tier']}"
        shown[key] = shown.get(key, 0) + 1

    feats = tuple(bt.get("features", ()))
    meta = {"rows": len(obs["any"]), "rows_opened": sum(o.t is not None for o in obs["any"]),
            "rows_real": sum(o.t is not None for o in obs["real"]),
            "rows_token": sum(o.token for o in obs["real"]),
            "rows_segment": {s: sum(o.seg == s for o in obs["any"]) for s in SEGMENTS},
            "rows_scheduled": sum(o.sched is not None for o in obs["any"]), **acc,
            "records": dict(sorted(counts.items())),
            "records_by_state_tier": dict(sorted(shown.items())),
            "half_life_days": bt.get("half_life_days"), "features": list(feats),
            "maturity_days": bt.get("maturity_days"),
            "segment_rates": seg_rates, "commitments": rates, "tiers": tiers,
            "tier_gates": {name: {"hit": key, "rule": HIT_RULES[key], "wilson_lower_at_least": gate}
                           for name, key, gate in TIER_GATES}}
    if model and overall:
        d = describe(Lag(_dot(overall.beta, [0] * len(feats)) + overall.b["*"], overall.sigma,
                         _sigmoid(overall.a["*"])), today, 0, today)
        new_film = model.dist("\x00new", (0,) * len(FEATURES))
        meta.update({"pi_pooled": round(_sigmoid(overall.a["*"]), 3),
                     "lag_median_days": round(math.expm1(overall.b["*"]), 1),
                     "lag_sd_days": d["sd_days"], "model": model_summary(model),
                     "real_model": model_summary(model_real) if model_real else None,
                     "weekday_shares": [round(v, 3) for v in model.cal] if model.cal else None,
                     "milestones": ms_meta})
        if new_film:
            meta["new_company_feature_film_p_open"] = round(new_film[2], 3)
    for kind, counts_d in delays.items():
        meta[f"{kind}_to_rating"] = {"n": sum(counts_d.values()), "median_days": _median_counts(counts_d)}
    bt["by_state"] = state_report(bt, rates, tiers)
    meta["backtest"] = bt
    meta["backtest_real"] = bts["real"]
    return meta


def _median_counts(counts: dict):
    n = sum(counts.values())
    acc = 0
    for d, c in sorted(counts.items()):
        acc += c
        if acc * 2 >= n:
            return d
    return None


def anchor_of(rec: dict) -> tuple[str, str] | None:
    st = rec["stages"]
    for kind, d in (("rating", st.get("rating", {}).get("date")),
                    ("receipt", st.get("intake", {}).get("date")),
                    ("promo", st.get("promo", {}).get("first_date")),
                    ("ors_ad", st.get("ors_ad", {}).get("first_date")
                     or (rec.get("first_seen", {}).get("ors_ad", "")[:10] if st.get("ors_ad") else ""))):
        if _valid_day(d):
            return kind, _valid_day(d)
    return None


def _delay_given(counts: dict, elapsed: int) -> dict | None:
    """Delay distribution given no rating yet after `elapsed` days; a film
    later than every observed delay is taken as rated today at the earliest.
    None without observed delays."""
    if not counts:
        return None
    later = {d: c for d, c in counts.items() if d >= elapsed}
    tot = sum(later.values())
    return {d: c / tot for d, c in later.items()} if tot else {elapsed: 1.0}


def latest_milestone(rec: dict, model: Fit, today_s: str) -> tuple[str, str] | None:
    """(kind, date) of the latest milestone after the rating that has a
    model, on or before today."""
    rd = _valid_day(rec["stages"].get("rating", {}).get("date"))
    if not rd or not model.milestones:
        return None
    fs = rec.get("first_seen", {})
    chain = [fs[c] for c in ("megabox", "lotte", "cgv") if _valid_day(fs.get(c))]
    dates = {"promo": next((d for d in promo_dates(rec) if d > rd), ""),
             "kobis": _valid_day(fs.get("kobis")), "chain": min(chain) if chain else "",
             "booking": getattr(model, "booking_seen", {}).get(
                 rec["stages"]["rating"].get("rating_no"), "")}
    found = [(d, k) for k, d in dates.items() if k in model.milestones and d and rd < d <= today_s]
    if not found:
        return None
    d, k = max(found)
    return k, d


def _cond_pi(pi: float, f_e: float) -> float:
    """P(ever opens | no opening yet) of a cure model with eventual share pi
    and lag CDF f_e at the elapsed time."""
    den = 1 - pi * f_e
    return pi * (1 - f_e) / den if den > 0 else 0.0


def _with_pi(lag: Lag, pi: float) -> Lag:
    """The same lag distribution (delay, calendar) with another P(opens)."""
    out = Lag(lag.m, lag.s, pi, lag.delay)
    out.cal = lag.cal
    return out


def _p_given(m: float, s: float, pi: float, elapsed: int) -> float:
    lag = Lag(m, s, pi)
    f_e = lag.cdf(elapsed - 1) if elapsed > 0 else 0.0
    return pi * (1 - f_e) / (1 - pi * f_e)


def real_ratio(rec: dict, model: Fit | None, real: Fit | None, today: dt.date) -> float | None:
    """P(real opening) / P(any opening) of the two rating-anchored applicant
    models for this film (None without both): the share of its theatrical
    chance that is a real release, given no opening since the rating (a film
    not rated yet: as on its rating day)."""
    rd = _valid_day(rec["stages"].get("rating", {}).get("date"))
    if not model or not real:
        return None
    x = rec_features(rec)
    g = group_key("", rec.get("applicant", ""))
    el = max(0, (today - _d(rd)).days) if rd else 0
    m1, s1, p1, _ = model.dist(g, x)
    m2, s2, p2, _ = real.dist(g, x)
    pa, pr = _p_given(m1, s1, p1, el), _p_given(m2, s2, p2, el)
    return min(1.0, pr / pa) if pa > 0 else None


def estimate(rec: dict, today: dt.date, model: Fit | None, delays: dict,
             ctx: dict | None = None) -> dict | None:
    """The `release` value of one record, or None. `ctx` (apply): the real
    opening model, commitment rates, tiers, segment rates, applicant history
    and release_lags rows by rating number."""
    if "dormant" in rec.get("tags", []):
        return None
    ctx = ctx or {}
    today_s = today.isoformat()
    row = rating_row(rec, ctx.get("rows"))
    srcs = sources(rec, row)
    conf = confirmed(rec, srcs)
    rates, tiers = ctx.get("rates", {}), ctx.get("tiers", {})
    seg = rec_segment(rec, row, ctx.get("history", {}))
    ratio = real_ratio(rec, model, ctx.get("real"), today)

    def finish(out: dict, state: str, p, p_real, date: str, quantiles=None) -> dict:
        # a segment is a rating-day judgment: once a source dates the film
        # (a day, a month, a passed listing) it is judged like any film
        tier = ("estimate" if seg and state in SEGMENT_STATES
                else tiers.get(state, {}).get("tier", "estimate"))
        out.update({"p_open": p, "p_open_real": p_real, "signal_state": state})
        if seg:
            out["segment"] = seg
        out["display"] = display(tier, state, p, date, quantiles)
        return out

    if conf and conf[1] == "day":
        out = {"date": conf[0], "precision": "day", "basis": conf[2], "sources": srcs}
        state = signal_state(rec, conf, today_s, "")
        if state == "opened":
            # KOBIS recorded the opening, or only a source's listing date has
            # passed (a chain, program, presale or schedule day: the film is
            # listed as showing, but this is no observed opening and says
            # nothing about a real release)
            out["opened_by"] = "kobis" if conf[2] == "kobis" else "listing"
            row_ = row or {}
            real = row_.get("real_open") if conf[2] == "kobis" and row_.get("open_dt") == conf[0] \
                else None
            return finish(out, state, 1.0, (1.0 if real else 0.0) if isinstance(real, bool) else None,
                          conf[0])
        c = rates.get(state, {})
        p = c.get("q")
        out["commitment"] = {"q": p, "r": c.get("r", {}), "n": c.get("n", 0),
                             "provisional": c.get("provisional", True)}
        return finish(out, state, p, round(p * ratio, 3) if p is not None and ratio is not None else None,
                      conf[0])
    anc = anchor_of(rec)
    lag, name, n_dist, company, est = None, "overall", 0, "", False
    if anc and model:
        kind, adate = anc
        el = max(0, (today - _d(adate)).days)
        rr = "rerelease" in rec.get("tags", [])
        x = rec_features(rec, kind)
        # applicant companies also once KOBIS names a distributor: the
        # distributor-grouped model is only backtested as if known at the
        # rating (optimistic), so it is a diagnostic, not a live model
        g = group_key("", rec.get("applicant", ""))
        use = model
        ms = latest_milestone(rec, model, today_s) if kind == "rating" else None
        if ms:
            use = model.milestones[ms[0]]
            anc = (f"{ms[0]}_after_rating", ms[1])
            kind, adate = anc
        m, s, pi, known = use.dist(g, x)
        delay = None
        # an unrated film: delay to its rating, then the rating model; a
        # trailer/poster-only re-release needs no new rating: no estimate
        usable = kind not in ("promo", "ors_ad") or not rr
        if kind in ("receipt", "promo", "ors_ad") and usable:
            # an ORS filing waits for its rating like a trailer/poster rating
            delay = _delay_given(delays.get("promo" if kind == "ors_ad" else kind, {}), el)
            usable = delay is not None
        if usable:
            lag = Lag(m, s, pi, delay, cal=(_d(adate).toordinal(), model.cal) if model.cal else None)
            company = "applicant"
            est = bool(use.own_weight(g))
            if known:
                name, n_dist = "distributor", round(use.n_rows[g], 1)
    seg_p = ctx.get("seg_rates", {}).get("any", {}).get(seg, {}).get("rate") if seg else None
    seg_real = ctx.get("seg_rates", {}).get("real", {}).get(seg, {}).get("rate") if seg else None
    if conf:  # a month only
        out = {"date": conf[0], "precision": "month", "basis": conf[2], "sources": srcs}
        first = _d(conf[0] + "-01")
        last = _month_end(first)
        c = rates.get("month_announced", {})
        if lag is None:
            return finish(out, "month_announced", c.get("q"), None, conf[0])
        if last >= today:
            adate_d = _d(anc[1])
            el = max(0, (today - adate_d).days)
            q, r = c.get("q"), c.get("r", {}).get("month")
            # without a measured reliability the month does not narrow the estimate
            window = ((first - adate_d).days, (last - adate_d).days) if r is not None else None
            d = describe(lag, adate_d, el, today, window, r if r is not None else 1.0,
                         q if r is not None else None)
            if r is not None:
                out["commitment"] = {"q": q, "r": c.get("r", {}), "n": c.get("n", 0),
                                     "provisional": c.get("provisional", True)}
            out.update({"anchor": {"kind": anc[0], "date": anc[1]}, "elapsed_days": el, **d,
                        "model": name, "n_distributor": n_dist, "company": company, "established": est})
            p_real = round(d["p_open"] * ratio, 3) if ratio is not None else None
            return finish(out, "month_announced", d["p_open"], p_real, conf[0], d["quantiles"])
        # the month passed without an opening: estimate without it
    if lag is None:
        return None
    adate_d = _d(anc[1])
    el = max(0, (today - adate_d).days)
    if seg_p is not None:
        # the segment's opening share as the eventual P(opens) (no segment
        # row opened after the maturity horizon, see SEGMENTS), conditioned
        # on no opening so far with the film's lag model as the timing
        f_e = lag.cdf(el - 1) if el > 0 else 0.0
        d = describe(_with_pi(lag, seg_p), adate_d, el, today)
        p_real = round(_cond_pi(seg_real, f_e), 3) if seg_real is not None else None
    else:
        d = describe(lag, adate_d, el, today)
        p_real = round(d["p_open"] * ratio, 3) if ratio is not None else None
    out = {"date": d["quantiles"]["p50"], "precision": "estimate", "basis": "model",
           "sources": srcs, "anchor": {"kind": anc[0], "date": anc[1]}, "elapsed_days": el,
           **d, "model": name, "n_distributor": n_dist, "company": company, "established": est}
    state = signal_state(rec, None, today_s, anc[0])
    return finish(out, state, d["p_open"], p_real, d["quantiles"]["p50"], d["quantiles"])
