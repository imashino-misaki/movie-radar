"""Static website build: data/*.json -> the files the web page reads (pure, plus one writer).

Outputs, relative to the repository root (GitHub Pages serves the root of main, no build step):

* ``data/site/index.js``   ``window.MR``: the film list, events, first sightings, box office.
* ``data/site/details.js`` ``window.MRD``: per-film detail, loaded on demand by the page.
* ``feed.xml``             RSS: everything the radar reports (one item per film per day).
* ``new.xml``              RSS: first sightings only (one item per film, stable GUID).

Every output is a function of the data files alone: no run time, stable ordering, so a run whose
data did not change rewrites nothing. The page (``index.html``, ``app.js``, ``style.css`` at the
repository root) is static and not generated.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote
from xml.sax.saxutils import escape as xesc

Film = dict[str, Any]

SITE_DIR = "site"  # under the data directory
INDEX_JS = "index.js"
DETAILS_JS = "details.js"
FEED = "feed.xml"
NEW_FEED = "new.xml"
HISTORY = "forecast_history.json"  # in the data directory, committed with the data
STATUS_JS = "status.js"  # per-source freshness; changes every run, so it is kept out of index.js

# Freshness banner: a source counts as stale when its last good run is older than STALE_FACTOR
# collection intervals. RUN_INTERVAL_HOURS mirrors the schedule in .github/workflows/collect.yml
# (cron "7 */2 * * *"; a test keeps the two in step); the factor 2 is the owner's rule.
RUN_INTERVAL_HOURS = 2
STALE_FACTOR = 2
# Sources shown on the page, in display order (meta.json "sources" keys -> label). Helper stages
# (lookups, posters, company names) are not listed: they do not decide what the page shows.
SOURCE_LABELS = (
    ("kmrb_rating", "영등위 등급 분류"),
    ("kmrb_promo", "영등위 홍보물 심의"),
    ("ors_intake", "등급 분류 신청"),
    ("ors_ad", "광고물 심의 신청"),
    ("kobis", "KOBIS 영화 목록"),
    ("kobis_upcoming", "KOBIS 개봉 예정"),
    ("kobis_schedule", "KOBIS 개봉 일정"),
    ("kobis_booking", "KOBIS 예매율"),
    ("kobis_boxoffice", "KOBIS 박스오피스"),
    ("megabox", "메가박스"),
    ("megabox_program", "메가박스 기획전"),
    ("lotte", "롯데시네마"),
    ("cgv", "CGV"),
)

# Word scale for model estimates, applied to ``release.p_open_real`` (the probability of ever
# opening for real, else ``release.p_open``;
# the last ``release.by`` value when it is missing). Thresholds of the approved
# viewer design, stated to users in the page's "확률 읽는 법" note: 유력 >= 80 %, 가능 50-79 %,
# 불확실 < 50 %.
LIKELY = 0.80
POSSIBLE = 0.50
# The detail page draws an estimate as DOTS similar films. A film whose p_open rounds to none of
# them (p_open < 1/(2*DOTS)) is "rare": the page shows no expected date for it, since the median
# date (given it opens) read as a coming opening for films that almost never open (shorts and
# VOD-only applicants: release.SEGMENTS, 77 of 159 estimates on 2026-11-02)
DOTS = 20
# Events and first sightings older than this (before the data date) are not exported; the page
# shows 7 and 14 days of them.
EVENT_KEEP_DAYS = 30
FEED_ITEMS = 80  # feed.xml: newest (film, day) groups
# Public address of the site (GitHub Pages of the repository): RSS links must be absolute URLs.
SITE_URL = "https://imashino-misaki.github.io/movie-radar/"

# ---------------------------------------------------------------------------- small helpers
WD = "월화수목금토일"
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
GRADE = {"전체관람가": "전체", "12세이상관람가": "12세", "15세이상관람가": "15세",
         "청소년관람불가": "청불", "제한상영가": "제한"}
GRADE_LONG = {"전체": "전체 관람가", "12세": "12세 관람가", "15세": "15세 관람가",
              "청불": "청소년 관람불가", "제한": "제한상영가"}
KIND = {"poster": "포스터", "trailer": "예고편"}


# the bracket of a "[...] title" chain listing that marks a staged performance
# broadcast (hidden with concerts) rather than a festival / special screening
_PERFORMANCE = re.compile(r"오페라|발레|뮤지컬|연극|콘서트|라이브|실황|nt\s*live|\bmet\b|\brbo\b|opera|ballet|bolshoi",
                          re.I)


def screening_kind(title: str, tags: list, mb: dict | None, lt: dict | None, cg: dict | None) -> str:
    """The kind of a screening that is not a regular theatrical run, or "".

    sport: a live sports broadcast (match wording in the title; no chain marks sports).
    ev: concerts, live viewings and staged-performance broadcasts.
    fest: festival and special screenings.
    Title wording decides first (a "KBO ... 7차전" listing is sport even though CGV
    files it with concerts); then the chains' own markers: CGV lists all of its
    non-film content (concerts, performances, sports) under ICECON, Lotte badges
    concerts and live viewings 롯시플 and names a festival screening's festival, and
    Megabox flags performance broadcasts and lecture screenings 클래식 소사이어티.
    """
    if "sports" in tags:
        return "sport"
    bracket = re.match(r"^\s*\[([^\]]*)\]", title)
    if "concert" in tags or (bracket and _PERFORMANCE.search(bracket.group(1))):
        return "ev"
    if bracket or (lt or {}).get("festival"):
        return "fest"
    if ((lt or {}).get("badge") == "plus" or (mb or {}).get("classic")
            or "icecon" in ((cg or {}).get("labels") or [])):
        return "ev"
    return ""


ANIME_GENRE = "애니메이션"


def official_genres(recs: list[dict]) -> list[str]:
    """Every official genre the records' stages carry: Lotte's genre, the KOBIS
    Open API's genres, the KOBIS detail page's and the KOBIS calendar's."""
    out: list[str] = []
    for r in recs:
        st = r.get("stages") or {}
        if (st.get("lotte") or {}).get("genre"):
            out.append(st["lotte"]["genre"])
        for name in ("kobis", "kobis_detail", "schedule"):
            out += [g for g in (st.get(name) or {}).get("genres") or [] if g]
    return out


def is_anime(recs: list[dict], tags) -> bool:
    """The 애니메이션 category. Precedence: an official genre (official_genres)
    decides whenever any source gives one; only a film with no genre from any
    source falls back to the title heuristic tag "anime" (tags.py)."""
    genres = official_genres(recs)
    if genres:
        return any(ANIME_GENRE in g for g in genres)
    return "anime" in tags


def tnorm(s: str | None) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", (s or "").lower())


def company_key(name: str | None) -> str:
    """Hangul part of a company name without legal forms ("씨제이포디플렉스 주식회사 (CJ 4D" ->
    "씨제이포디플렉스")."""
    s = re.sub(r"\(주\)|㈜|\(유\)|\(재\)|\(사\)|주식회사|유한책임회사|유한회사", "", name or "")
    return re.sub(r"[^가-힣]", "", s)


def day(value: str | None) -> str:
    """A first_seen value as a UTC date: most are dates, ``ors_ad`` is a run time."""
    return (value or "")[:10]


def choseong(s: str) -> str:
    out = []
    for ch in s:
        c = ord(ch)
        if 0xAC00 <= c <= 0xD7A3:
            out.append(CHO[(c - 0xAC00) // 588])
        elif not ch.isspace():
            out.append(ch.lower())
    return "".join(out)


FULLWIDTH = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}


def norm(s: str) -> str:
    """Search key: lower case, full-width ASCII folded, everything but letters and digits dropped
    ("슬픈 열대", "슬픈열대" and "슬픈-열대!" give one key). app.js normalizes the query the same way."""
    return re.sub(r"[\W_]+", "", (s or "").translate(FULLWIDTH).lower())


def clean_co(name: str | None) -> str:
    """Company name for display: legal forms and a trailing Latin spelling dropped."""
    if not name:
        return ""
    s = re.sub(r"\(주\)|㈜|주식회사|\(유\)|유한회사|유한책임회사|\(재\)|\(사\)", " ", name)
    s = s.replace("(", " (").strip()
    if re.search(r"[가-힣]", s):
        s = re.sub(r"(\s+[A-Za-z0-9&.,'\-]+)+$", "", s)
    s = re.sub(r"\s*\(\s*\)", "", s)
    if re.search(r"[가-힣]", s):  # a Latin spelling in parentheses, possibly cut: "(CJ 4DPLEX Co., Ltd.)"
        s = re.sub(r"\s*\([A-Za-z0-9&.,'\-\s]*\)?\s*$", "", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def grade_of(st: dict) -> str:
    for k in ("rating", "kobis_detail", "megabox", "lotte", "cgv", "past_rating"):
        g = (st.get(k) or {}).get("grade")
        if g in GRADE:
            return GRADE[g]
    return ""


def runtime_of(st: dict) -> int | None:
    kd_ = st.get("kobis_detail") or {}
    if kd_.get("runtime"):
        return int(kd_["runtime"])
    m = re.match(r"(\d+)\s*분", (st.get("rating") or {}).get("runtime") or "")
    return int(m.group(1)) if m else None


def fmt_of_text(s: str) -> set[str]:
    s = s.lower()
    out = set()
    if "imax" in s or "아이맥스" in s:
        out.add("imax")
    if "4dx" in s or re.search(r"\b4d\b", s):
        out.add("4d")
    if "screenx" in s or "스크린엑스" in s or re.search(r"\(sx\)", s):
        out.add("sx")
    if "dolby" in s or "돌비" in s:
        out.add("dolby")
    return out


def kd(iso: str, wd: bool = True) -> str:
    d = dt.date.fromisoformat(iso)
    s = f"{d.month}월 {d.day}일"
    return f"{s} ({WD[d.weekday()]})" if wd else s


def mon(ym: str, cur_year: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    return (f"{y}년 " if y != cur_year else "") + f"{m}월"


def pct(p: float) -> str:
    """Never 100 % for anything not confirmed."""
    return f"{min(99, round(p * 100))}%"


# ---------------------------------------------------------------------------- certainty
def certainty(rel: dict) -> dict:
    """One certainty per film, shared by every view. Keys: p (day / month / est / none), w (word
    key), and per kind: day: date | month: m | est: md / lo / hi (the model's median, p10 and p90
    dates, given it opens), mm (the median's month, the page's grouping), po (``release.p_open``:
    P(it ever opens | not open yet)), by [[deadline, p]...] (cumulative, for the detail chart),
    pp (p at the last deadline), dl (the last deadline: the model's last ``by`` key)."""
    prec = rel.get("precision")
    if prec in ("day", "week") and rel.get("date"):
        return {"p": "day", "w": "day", "date": rel["date"]}
    if prec == "month" and rel.get("date"):
        return {"p": "month", "w": "month", "m": rel["date"][:7]}
    if prec == "estimate":
        by = sorted((rel.get("by") or {}).items())
        q = rel.get("quantiles") or {}
        # the shown probability is a real opening (KOFIC: 40+ screenings in the year,
        # lags_build.real_open); a token release (few screenings ahead of VOD) counts as
        # none. pa: any opening, token included (the detail page names it when it differs)
        pa = rel.get("p_open")
        po = rel.get("p_open_real") if rel.get("p_open_real") is not None else pa
        pp = by[-1][1] if by else None
        # the word grades the probability shown next to it: p_open (the last deadline's p without it)
        p = po if po is not None else pp
        w = "unknown" if p is None else "likely" if p >= LIKELY else "possible" if p >= POSSIBLE else "unknown"
        if po is not None and round(DOTS * po) == 0:
            w = "rare"
        md = q.get("p50") or None
        return {"p": "est", "w": w, "md": md, "lo": q.get("p10") or None, "hi": q.get("p90") or None,
                "mm": md[:7] if md else None, "po": round(po, 3) if po is not None else None,
                "pa": round(pa, 3) if pa is not None else None,
                "by": [[k, round(v, 3)] for k, v in by], "pp": round(pp, 3) if pp is not None else None,
                "dl": by[-1][0] if by else None,
                # sm: the model has no record of the company and uses the overall fit only
                # ("자료 적음"). Not `established`: the any-opening model's companies differ so
                # little that no company outweighs the prior (k_rows 283.6, 0 companies, 2026-11-02)
                "sm": rel.get("model") == "overall",
                # seg: the rating-day segment behind a rare estimate (short / vod_applicant / ...)
                "seg": rel.get("segment") or None}
    return {"p": "none", "w": "none"}


WORD = {"day": "확정", "month": "월 공개", "likely": "유력", "possible": "가능", "unknown": "불확실",
        "rare": "가능성 낮음", "none": "근거 부족"}


def median_text(iso: str) -> str:
    """"예상 12월 2일경": the model's median opening date."""
    return f"예상 {kd(iso, False)}경"


def cert_text(c: dict, today: str) -> str:
    """Same wording as app.js cert(): used in the feeds."""
    cur = int(today[:4])
    if c["p"] == "day":
        return f"{kd(c['date'])} 개봉" if c["date"] <= today else f"{kd(c['date'])} 개봉 확정"
    if c["p"] == "month":
        return f"{mon(c['m'], cur)} 개봉 예정"
    if c["p"] == "est" and c.get("w") == "rare":
        return "개봉 가능성 낮음"
    if c["p"] == "est" and c.get("md"):
        return median_text(c["md"])
    return "시기 미정 · 아직 소식 없음"


# ---------------------------------------------------------------------------- dedupe
PREC_RANK = {"day": 0, "week": 0, "month": 1, "estimate": 2}
THEATRICAL_ORDER = {"confirmed": 0, "candidate": 1, "token": 2, "unlikely": 3}


def _rel_value(f: Film) -> tuple | None:
    r = f.get("release") or {}
    return (r.get("precision"), r.get("date")) if r.get("precision") in ("day", "month") else None


# Identifiers that name one film, per kind. Records of the same normalised title merge only when
# they share a value of some kind and no kind has values on both sides without one in common
# (two KOBIS codes, two rating numbers: two films). KOBIS ``audit_nos`` are the KMRB rating
# numbers KOBIS lists for the film. Filing receipt numbers (intake, advertising) are not used: the
# collector joins filings to records by title, so a shared receipt only repeats the title match.
ID_FIELDS = (
    ("kobis", (("kobis", "movie_cd"), ("kobis_detail", "kobis_cd"), ("release_history", "kobis_cd"),
               ("lotte", "kobis_cd"), ("booking", "movie_cd"), ("schedule", "movie_cd"))),
    ("rating", (("rating", "rating_no"), ("kobis", "audit_nos"))),
    ("megabox", (("megabox", "movie_nos"),)),
    ("lotte", (("lotte", "movie_code"),)),
)


def identifiers(rec: Film) -> dict[str, set[str]]:
    """{kind: values} of the film identifiers a record carries (see ID_FIELDS)."""
    st = rec.get("stages") or {}
    out: dict[str, set[str]] = {}
    for kind, fields in ID_FIELDS:
        vals: set[str] = set()
        for stage, key in fields:
            v = st[stage].get(key) if isinstance(st.get(stage), dict) else None
            vals.update(str(x) for x in (v if isinstance(v, list) else [v]) if x)
        if vals:
            out[kind] = vals
    return out


def _union_ids(recs: Iterable[Film]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        for k, v in identifiers(r).items():
            out[k] |= v
    return out


def _conflicts(a: dict[str, set[str]], b: dict[str, set[str]], kinds: Iterable[str]) -> bool:
    return any(a.get(k) and b.get(k) and not (a[k] & b[k]) for k in kinds)


def same_film(a: list[Film], b: list[Film]) -> bool:
    """May two groups of records (each one film) of the same normalised title be merged? Only on a
    shared identifier, with no conflicting one, and not when they announce different dates."""
    ia, ib = _union_ids(a), _union_ids(b)
    if not any(ia.get(k, set()) & v for k, v in ib.items()) or _conflicts(ia, ib, list(ib)):
        return False
    va = {v for v in map(_rel_value, a) if v}
    vb = {v for v in map(_rel_value, b) if v}
    return not (va and vb and va != vb)  # two different announced dates: two release cycles


def _companies(recs: Iterable[Film]) -> set[str]:
    """company_key of every applicant and KOBIS distributor the records name."""
    out: set[str] = set()
    for r in recs:
        kob = (r.get("stages") or {}).get("kobis")
        names = [r.get("applicant")] + list((kob or {}).get("distributors") or []) if isinstance(kob, dict) \
            else [r.get("applicant")]
        out |= {company_key(n) for n in names}
    return out - {""}


def join_unidentified(clusters: list[list[Film]], kinds: dict[int, str]) -> list[list[Film]]:
    """Join same-title clusters that carry no film identifier at all to the one identified cluster
    of the same company. Such a cluster is a filing the collector has not tied to a rating yet
    (an intake before its rating decision): once the rating is decided, KOBIS lists its number and
    the identifier rule joins them anyway. Joined only when exactly one identified cluster names
    the filing's applicant (as applicant or KOBIS distributor), that cluster has not already opened
    (a filing for a film already showing starts another release cycle), and neither side announces
    a date the other contradicts. Marks each joined head in ``kinds`` as "same_company"."""
    bare = [c for c in clusters if not _union_ids(c)]
    if not bare:
        return clusters
    known = [c for c in clusters if _union_ids(c)]
    for c in bare:
        co = {company_key(r.get("applicant")) for r in c} - {""}
        hits = [k for k in known if co and co & _companies(k)]
        if len(hits) != 1:
            continue
        k = hits[0]
        if any((r.get("release") or {}).get("signal_state") == "opened" for r in k):
            continue
        vk = {v for v in map(_rel_value, k) if v}
        vc = {v for v in map(_rel_value, c) if v}
        if vk and vc and vk != vc:
            continue
        kinds[id(c[0])] = "same_company"
        k += c
        clusters = [x for x in clusters if x is not c]
    return clusters


def _primary_key(f: Film) -> tuple:
    st = f.get("stages") or {}
    return (PREC_RANK.get((f.get("release") or {}).get("precision") or "", 3),
            0 if st.get("kobis") else 1, 0 if f.get("poster") else 1, f["id"])


# Edition labels a title can end with, outside the trailing-parenthesis formats the collector
# already groups (collector.text.split_format): "언섀터 스크린엑스", "명탐정 코난: ... 자막".
EDITIONS = {"스크린엑스": "sx", "screenx": "sx", "sx": "sx", "4dx": "4d", "mx4d": "4d",
            "imax": "imax", "아이맥스": "imax", "돌비": "dolby", "dolby": "dolby",
            "자막": "sub", "더빙": "dub"}
_EDITION = "|".join(sorted((re.escape(k) for k in EDITIONS), key=len, reverse=True))
_EDITION_RE = re.compile(rf"^(.*\S)(?:\s*\(\s*({_EDITION})\s*\)|\s+({_EDITION}))\s*$", re.I)


def split_edition(title: str) -> tuple[str, str]:
    """(title without a trailing edition label, the label's tag or "")."""
    m = _EDITION_RE.match(title or "")
    if not m or not tnorm(m.group(1)):
        return title or "", ""
    return m.group(1), EDITIONS[(m.group(2) or m.group(3)).lower()]


def dedupe(films_raw: list[Film]) -> tuple[list[Film], dict[str, str], list[dict], dict[str, list[Film]],
                                           dict[str, set[str]]]:
    """Merge records of one film (on deep copies).

    1. Same normalised title, compatible (see ``compatible``): the best-dated record leads and
       takes the others' stages, first sightings, tags and poster.
    2. A record whose title is another film's title plus an edition label ("언섀터 스크린엑스")
       folds into that film when exactly one film has the plain title and both have the same
       applicant company; it only adds a format tag.
    Returns (films without merged / folded records, alias id -> film id, merge log,
    film id -> its original records, film id -> edition tags)."""
    films = copy.deepcopy(films_raw)
    aliases: dict[str, str] = {}
    log: list[dict] = []
    members: dict[str, list[Film]] = {f["id"]: [copy.deepcopy(f)] for f in films}
    groups: dict[str, list[Film]] = defaultdict(list)
    for f in films:
        if not f.get("variant_of"):
            groups[tnorm(f.get("title"))].append(f)
    for key in sorted(groups):
        if len(groups[key]) < 2 or not key:
            continue
        # clusters in lead order; join any two that are one film until nothing changes, so a record
        # that links two others (a KOBIS code on one side, a chain number on the other) joins all
        clusters = [[f] for f in sorted(groups[key], key=_primary_key)]
        joined = True
        while joined:
            joined = False
            for i, j in ((i, j) for i in range(len(clusters)) for j in range(i + 1, len(clusters))):
                if same_film(clusters[i], clusters[j]):
                    clusters[i] += clusters.pop(j)
                    joined = True
                    break
        kinds = {id(c[0]): "same_title" for c in clusters}
        clusters = join_unidentified(clusters, kinds)
        for prim, *others in clusters:
            for other in others:
                aliases[other["id"]] = prim["id"]
                members[prim["id"]] += members.pop(other["id"])
                log.append({"kind": kinds.get(id(other), "same_title"), "title": prim["title"], "into": prim["id"],
                            "id": other["id"], "other_title": other["title"]})
    films = [f for f in films if f["id"] not in aliases]

    editions: dict[str, set[str]] = defaultdict(set)
    plain: dict[str, list[Film]] = defaultdict(list)
    for f in films:
        if not f.get("variant_of"):
            plain[tnorm(f.get("title"))].append(f)
    for f in films:
        if f.get("variant_of") or f["id"] in aliases:
            continue
        base, tag = split_edition(f.get("title") or "")
        if not tag:
            continue
        targets = plain.get(tnorm(base), [])
        if len(targets) != 1 or targets[0] is f:
            continue
        target = targets[0]
        keys = {company_key(m.get("applicant")) for m in members[target["id"]]} - {""}
        if company_key(f.get("applicant")) not in keys:
            continue
        # each edition has a rating number of its own; another KOBIS code means another film
        if _conflicts(_union_ids(members[target["id"]]), identifiers(f), ("kobis",)):
            continue
        aliases[f["id"]] = target["id"]
        members[target["id"]] += members.pop(f["id"])
        editions[target["id"]].add(tag)
        log.append({"kind": "edition", "title": target["title"], "into": target["id"], "id": f["id"],
                    "other_title": f["title"]})
    films = [f for f in films if f["id"] not in aliases]
    ids = {f["id"] for f in films}
    for f in films:
        v = f.get("variant_of")
        if v in aliases:
            f["variant_of"] = aliases[v]
        if f.get("variant_of") and f["variant_of"] not in ids:
            f.pop("variant_of")  # its lead is gone: show it on its own
    leads = {f["id"] for f in films if not f.get("variant_of")}
    for f in films:  # a collector variant (variant_of) is its lead's film: its records count for the lead
        if f.get("variant_of") in leads:
            members[f["variant_of"]] += [copy.deepcopy(m) for m in members[f["id"]]]
    for f in films:
        if f["id"] in leads:
            for m in members[f["id"]][1:]:
                absorb(f, m)
    return films, dict(sorted(aliases.items())), log, members, editions


def absorb(lead: Film, other: Film) -> None:
    """Fold one more record of the film into its lead (in place): stages the lead lacks, the
    earliest first sightings, tags, poster, original title, the strongest theatrical class, and the
    release when the other record has a more precise one (an announced or confirmed day over a
    month over a model estimate, whatever source gave it)."""
    for k, v in (other.get("stages") or {}).items():
        lead.setdefault("stages", {}).setdefault(k, v)
    for k, v in (other.get("first_seen") or {}).items():
        fs = lead.setdefault("first_seen", {})
        fs[k] = min(v, fs.get(k, v))
    lead["tags"] = sorted(set(lead.get("tags") or []) | set(other.get("tags") or []))
    if not lead.get("poster") and other.get("poster"):
        lead["poster"] = other["poster"]
    if not lead.get("original_title"):
        lead["original_title"] = other.get("original_title") or ""
    pt, ot = lead.setdefault("theatrical", {}), other.get("theatrical") or {}
    if THEATRICAL_ORDER.get(ot.get("class", ""), 9) < THEATRICAL_ORDER.get(pt.get("class", ""), 9):
        pt["class"] = ot.get("class")
    if not pt.get("applicant_history") and ot.get("applicant_history"):
        pt["applicant_history"] = ot["applicant_history"]
    if _rel_rank(other) < _rel_rank(lead):
        lead["release"] = copy.deepcopy(other["release"])


def _rel_rank(f: Film) -> int:
    """PREC_RANK of a record's release (3: none); a day or month needs its date."""
    rel = f.get("release") or {}
    prec = rel.get("precision") or ""
    if prec in ("day", "week", "month") and not rel.get("date"):
        return 3
    return PREC_RANK.get(prec, 3)


def stable_id(records: list[Film]) -> str:
    """A film identity for feed GUIDs, calendar UIDs and the page seen-keys: the id of the record
    the collector saw first (earliest first_seen day; ties: the smallest id). The lead changes when
    a better-dated record joins, but records found later join with later sightings, so this stays
    put as the group grows; it is a pure function of the records, hence deterministic."""
    def seen(r: Film) -> str:
        return min((day(v) for v in (r.get("first_seen") or {}).values() if day(v)), default="9999-99-99")
    return min(records, key=lambda r: (seen(r), r["id"]))["id"]


# ---------------------------------------------------------------------------- first sightings
# Order of sources when several traces share the first date (the label lists them in this order).
TRACE_ORDER = ("ors_ad", "intake", "promo", "rating", "kobis", "megabox", "lotte", "cgv", "program")
AD_KIND = {"poster": "포스터", "trailer": "예고편"}
CHAIN_NAME = {"megabox": "메가박스", "lotte": "롯데시네마", "cgv": "CGV"}


def source_baselines(films: Iterable[Film]) -> dict[str, str]:
    """First collection day per source: whatever was first seen that day was already there when the
    source was first read (a baseline, not news)."""
    base: dict[str, str] = {}
    for f in films:
        for k, v in (f.get("first_seen") or {}).items():
            d = day(v)
            if d:
                base[k] = min(d, base.get(k, d))
    return base


def _trace_label(src: str, stage: dict) -> str:
    if src == "ors_ad":
        kinds = [AD_KIND[k] for k in stage.get("kinds", []) if k in AD_KIND]
        return f"{'·'.join(kinds)} 광고 심의 신청" if kinds else "광고물 심의 신청"
    if src == "intake":
        return "등급 심의 신청"
    if src == "promo":
        # the stage keeps the kinds of all its ratings, not per date: one kind names it, several do not
        kinds = [KIND.get(k) or str(k) for k in stage.get("kinds", [])]
        return f"{kinds[0]} 심의" if len(kinds) == 1 else f"홍보물 심의 ({'·'.join(kinds) or '포스터·예고편'})"
    if src == "rating":
        return "등급 결정"
    if src == "kobis":
        return "KOBIS 등록"
    if src == "program":
        return f"메가박스 {stage.get('program') or '기획전'}"
    return f"{CHAIN_NAME.get(src, src)} 목록 등장"


def promo_dates(stage: dict) -> list[str]:
    """Every trailer/poster rating date of a promo stage, oldest first."""
    return sorted({d for d in (stage.get("dates_seen") or []) + [stage.get("first_date") or ""] if d})


def traces(rec: Film) -> list[tuple[str, str, int]]:
    """Dated traces of one record: (date, source, real). real = 1: a date the source records
    (receipt, rating, trailer/poster rating date); 0: the day the collector first saw it."""
    st, fs = rec.get("stages") or {}, rec.get("first_seen") or {}
    out: list[tuple[str, str, int]] = []
    if (st.get("intake") or {}).get("date"):
        out.append((st["intake"]["date"], "intake", 1))
    ad = st.get("ors_ad")
    if ad:
        dates = [d for d in ad.get("dates") or [] if d] or ([ad["first_date"]] if ad.get("first_date") else [])
        if dates:
            out.append((min(dates), "ors_ad", 1))
        elif day(fs.get("ors_ad")):  # filed, not received yet (신청완료): only the sighting day
            out.append((day(fs.get("ors_ad")), "ors_ad", 0))
    promo_days = promo_dates(st.get("promo") or {})
    if promo_days:
        out.append((promo_days[0], "promo", 1))
    if (st.get("rating") or {}).get("date"):
        out.append((st["rating"]["date"], "rating", 1))
    for src in ("kobis", "megabox", "lotte", "cgv", "program"):
        seen = day(fs.get(src))
        if not st.get(src) or not seen:
            continue
        opened = st[src].get("release_date") or ""
        if src in ("megabox", "lotte") and len(opened) == 10 and opened < seen:
            continue  # listed after it opened: a running film, not a first trace
        out.append((seen, src, 0))
    return out


def kobis_baseline_code(films: Iterable[Film], base: dict[str, str]) -> str:
    """Highest KOBIS movie code among films first seen on KOBIS's first collection day. KOBIS
    numbers registrations in order (year + sequence), so a code at or below it was registered by
    then, also when a list added later shows it for the first time."""
    codes = [(f.get("stages") or {}).get("kobis", {}).get("movie_cd") or "" for f in films
             if (f.get("stages") or {}).get("kobis") and day((f.get("first_seen") or {}).get("kobis")) == base.get("kobis")]
    return max(codes, default="")


def first_sighting(records: list[Film], base: dict[str, str], kobis_code: str = "",
                   keep_boot: bool = False) -> dict | None:
    """The earliest trace of a film across its records, or None when it cannot be dated: a trace
    seen on its source's first collection day (or a KOBIS code at or below ``kobis_code``) means the
    film existed by then, so the film counts only if a recorded date is earlier still. Past ratings
    and release history (an earlier run of a re-release) are not traces of this release."""
    all_tr = [(d, src, real, r["id"], r) for r in records for d, src, real in traces(r)]
    if not all_tr:
        return None

    def bootstrap(d: str, src: str, rec: Film) -> bool:
        if d <= base.get(src, ""):
            return True
        code = ((rec.get("stages") or {}).get("kobis") or {}).get("movie_cd") or ""
        return src == "kobis" and bool(kobis_code) and bool(code) and code <= kobis_code

    boot = [d for d, src, real, _, rec in all_tr if not real and bootstrap(d, src, rec)]
    real_dates = [d for d, _, real, _, _ in all_tr if real]
    first = min(d for d, *_ in all_tr)
    undated = bool(boot and not (real_dates and min(real_dates) < min(boot)))
    if undated and not keep_boot:
        return None
    tied = sorted((t for t in all_tr if t[0] == first), key=lambda t: (TRACE_ORDER.index(t[1]), t[3]))
    labels = list(dict.fromkeys(_trace_label(src, (rec.get("stages") or {}).get(src) or {})
                                for _, src, _, _, rec in tied))
    lead = tied[0]
    out = {"d": first, "w": " · ".join(labels), "s": lead[1], "r": lead[2], "g": f"{lead[1]}-{lead[3]}"}
    if keep_boot:
        out["b"] = int(undated)  # 1: already there when collection started: the real first trace is older
    return out


# ---------------------------------------------------------------------------- forecast history
# Owner-approved display rules for "업데이트": a film's forecast is recorded again only when it
# changed meaningfully against its last record: the median moved at least MEDIAN_SHIFT_DAYS, the
# p10-p90 range shrank by at least RANGE_SHRINK of its width, or the precision (day / month /
# estimate / none), the announced date or month, or the display tier changed.
MEDIAN_SHIFT_DAYS = 7
RANGE_SHRINK = 0.25
FORECAST_KINDS = ("conf", "moved", "mon", "shift", "narrow", "upd", "tier")
# Display tiers (collector/release.py TIER_GATES) in words: the interval a tier vouches for.
TIER_WORD = {"confirmed": "확정", "likely": "주 단위", "expected": "월 단위", "estimate": "추정"}


def snapshot(rel: dict) -> dict:
    """What the history keeps of a film's forecast."""
    c = certainty(rel)
    snap: dict[str, Any] = {"p": c["p"]}
    if c["p"] == "day":
        snap["date"] = c["date"]
    elif c["p"] == "month":
        snap["date"] = c["m"]
    elif c["p"] == "est":
        snap.update(p50=c.get("md"), p10=c.get("lo"), p90=c.get("hi"))
    tier = (rel.get("display") or {}).get("tier")
    if tier:
        snap["tier"] = tier
    return snap


def _days(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def _width(s: dict) -> int | None:
    return _days(s["p10"], s["p90"]) if s.get("p10") and s.get("p90") else None


def changed(prev: dict, cur: dict) -> bool:
    if prev.get("p") != cur.get("p") or prev.get("tier") != cur.get("tier"):
        return True
    if cur["p"] in ("day", "month"):
        return prev.get("date") != cur.get("date")
    if cur["p"] == "est":
        a, b = prev.get("p50"), cur.get("p50")
        if bool(a) != bool(b) or (a and b and abs(_days(a, b)) >= MEDIAN_SHIFT_DAYS):
            return True
        wa, wb = _width(prev), _width(cur)
        return bool(wa and wb is not None and wb <= wa * (1 - RANGE_SHRINK))
    return False


def update_history(history: dict, films_raw: list[Film], today: str) -> dict:
    """The forecast history after this run (pure): per film id, the list of
    {"d": day recorded, **snapshot}. A film seen for the first time is seeded (no event); an entry is
    appended only when ``changed``; so a rerun with unchanged data returns an equal history. A film
    whose id changed (records merged under another lead) keeps the history of any of its records."""
    films, _, _, members, _ = dedupe(films_raw)
    old = history.get("films", {})
    out: dict[str, list[dict]] = {}
    for f in films:
        if f.get("variant_of"):
            continue
        ids = [f["id"]] + [m["id"] for m in members.get(f["id"], []) if m["id"] != f["id"]]
        entries = next((list(old[i]) for i in ids if i in old), None)
        cur = snapshot(f.get("release") or {})
        if not entries:
            entries = [{"d": today, **cur}]
        elif changed({k: v for k, v in entries[-1].items() if k != "d"}, cur):
            entries.append({"d": today, **cur})
        out[f["id"]] = entries
    return {"schema": 1, "films": dict(sorted(out.items()))}


def _part(iso: str) -> str:
    """"11월 말" / "12월 초" / "12월 중순": a coarse date for a range."""
    d = dt.date.fromisoformat(iso)
    return f"{d.month}월 {'초' if d.day <= 10 else '중순' if d.day <= 20 else '말'}"


def _state_text(s: dict, cur_year: int) -> str:
    """A history entry in words, as the page shows it."""
    p = s.get("p")
    if p == "day" and s.get("date"):
        return f"{kd(s['date'], False)} 개봉"
    if p == "month" and s.get("date"):
        return f"{mon(s['date'], cur_year)} 개봉"
    if p == "est":
        return f"예상 {kd(s['p50'], False)}경" if s.get("p50") else "예상일 없음"
    return "시기 미정"


def forecast_events(entries: list[dict], cur_year: int) -> list[tuple[str, str, str, int]]:
    """(day, kind, words, n) per recorded change, n its index in ``entries`` (a stable key: the
    history is append-only). The first entry is the seed and makes none; every later entry makes
    exactly one: the specific wording below, else the change in words ("upd"), else (only the
    display tier moved) the tiers ("tier")."""
    out = []
    for n, (prev, cur) in enumerate(zip(entries, entries[1:]), 1):
        if not changed({k: v for k, v in prev.items() if k != "d"}, {k: v for k, v in cur.items() if k != "d"}):
            continue
        pp, cp = prev.get("p"), cur.get("p")
        ev: tuple[str, str] | None = None
        if cp == "day" and pp == "day" and prev.get("date") != cur.get("date"):
            # an old opening replaced by a re-release date names its year
            was = (f"{prev['date'][:4]}년 " if prev["date"][:4] != cur["date"][:4] else "") + kd(prev["date"], False)
            ev = ("moved", f"개봉일 변경 {was} → {kd(cur['date'], False)}")
        elif cp == "day" and pp != "day":
            ev = ("conf", f"월 공개 → {kd(cur['date'], False)} 확정" if pp == "month"
                  else f"개봉일 확정 {kd(cur['date'], False)}")
        elif cp == "month" and pp == "month" and prev.get("date") != cur.get("date"):
            ev = ("mon", f"개봉 월 변경 {mon(prev['date'], cur_year)} → {mon(cur['date'], cur_year)}")
        elif cp == "month" and pp != "month":
            ev = ("mon", f"{mon(cur['date'], cur_year)} 개봉 발표")
        elif cp == "est" and cur.get("p50"):
            a, b = prev.get("p50"), cur["p50"]
            if pp in ("day", "month"):
                ev = ("shift", f"개봉일 미정으로 바뀜 · 예상 {kd(b, False)}경")
            elif pp == "est" and a and abs(_days(a, b)) >= MEDIAN_SHIFT_DAYS:
                ev = ("shift", f"예상 {kd(a, False)}경 → {kd(b, False)}경")
            elif pp == "est" and changed({k: v for k, v in prev.items() if k not in ("d", "tier")},
                                         {k: v for k, v in cur.items() if k not in ("d", "tier")}) \
                    and cur.get("p10") and cur.get("p90"):
                ev = ("narrow", f"예상 범위가 좁혀졌어요 ({_part(cur['p10'])}~{_part(cur['p90'])})")
        if ev is None:
            was, now = _state_text(prev, cur_year), _state_text(cur, cur_year)
            if was != now:
                ev = ("upd", f"{was} → {now}")
            else:
                ev = ("tier", f"예측 등급 {TIER_WORD.get(prev.get('tier'), '없음')} → "
                              f"{TIER_WORD.get(cur.get('tier'), '없음')}")
        out.append((cur["d"], ev[0], ev[1], n))
    return out


# ---------------------------------------------------------------------------- build
CHAIN_ST = {"now": "상영·예매 중", "upcoming": "개봉 예정 목록에 있음", "ended": "목록에서 내려감"}
NEXT = {
    "filed": "등급 결정 (영등위)",
    "rating_only": "포스터·예고편 심의 또는 KOBIS 개봉일 등록",
    "promo_after_rating": "KOBIS 개봉일 등록 또는 극장 예매 오픈",
    "ad_filing": "KOBIS 개봉일 등록 또는 극장 예매 오픈",
    "month_announced": "정확한 개봉일 발표",
    "day_announced": "극장 예매 오픈",
    "bookable": "개봉",
}
EVW = {"filed": "등급 심의 신청", "rated": "등급 결정", "promo": "포스터·예고편 심의", "ad": "광고 심의 신청",
       "date": "개봉일 발표", "month": "개봉월 발표", "kobis": "KOBIS 등록", "chain": "극장 목록 등장",
       "open": "예매 시작", "excl": "단독 상영 발표"}
CHN = {"megabox": "메가박스", "lotte": "롯데시네마"}


def _js(name: str, obj: Any) -> str:
    return f"window.{name}=" + json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + ";\n"


def status(meta: dict) -> dict:
    """Freshness bundle from meta.json: the last run, the stale limit and, per shown source,
    [key, label, status, last good run (UTC ISO) or ""]. Sources the run skipped and never had
    (a chain not collected yet) are left out."""
    sources = meta.get("sources") or {}
    rows = []
    for key, label in SOURCE_LABELS:
        src = sources.get(key)
        if not src:
            continue
        st, ok = src.get("status") or "", src.get("last_ok_utc") or ""
        if st == "skipped" and not ok:
            continue
        rows.append([key, label, st, ok])
    return dict(run=meta.get("last_run_utc") or "", iv=RUN_INTERVAL_HOURS,
                stale=RUN_INTERVAL_HOURS * STALE_FACTOR, src=rows)


def _rss(title: str, description: str, items: list[str]) -> str:
    return ('<?xml version="1.0" encoding="utf-8"?>\n<rss version="2.0"><channel>'
            f"<title>{xesc(title)}</title><link>{xesc(SITE_URL)}</link><description>{xesc(description)}</description>"
            + "".join(items) + "</channel></rss>\n")


def film_url(fid: str) -> str:
    return f"{SITE_URL}#/film/{quote(fid, safe='')}"


def _pub(d: str) -> str:
    return dt.datetime.fromisoformat(d + "T09:00:00+09:00").strftime("%a, %d %b %Y %H:%M:%S %z")


def build(films_raw: list[Film], bo_days: dict[str, list[dict]], lag_rows: list[dict],
          poster_exists: Callable[[str], bool] = lambda name: True,
          history: dict | None = None) -> tuple[dict[str, str], dict]:
    """-> ({output path relative to the repository root: text}, stats). Pure. ``history``: the
    forecast history (``update_history``), the source of forecast-change events."""
    hist = (history or {}).get("films", {})
    films, aliases, merge_log, members, editions = dedupe(films_raw)
    seen_days = [day(d) for f in films for d in (f.get("first_seen") or {}).values() if day(d)]
    last_date = max(seen_days) if seen_days else dt.date(1970, 1, 1).isoformat()
    first_date = min(seen_days) if seen_days else last_date
    cur_year = int(last_date[:4])
    base = source_baselines(films)
    kobis_code = kobis_baseline_code(films, base)

    by_id = {f["id"]: f for f in films}
    variant_formats: dict[str, set[str]] = defaultdict(set)
    alt_titles: dict[str, set[str]] = defaultdict(set)
    for fid, recs in members.items():
        for r in recs:
            alt_titles[aliases.get(fid, fid)].update(x for x in (r.get("title"), r.get("original_title")) if x)
    for f in films:
        if f.get("variant_of"):
            variant_formats[f["variant_of"]].update(fmt_of_text(f.get("title") or ""))
            alt_titles[f["variant_of"]].update(x for x in (f.get("title"), f.get("original_title")) if x)
        for v in f.get("formats") or []:
            variant_formats[f["id"]].update(fmt_of_text(v.get("format") or ""))

    # ------------------------------------------------ box office (the latest day)
    # a box-office row joins a film by its KOBIS code, else by a rating number the release_lags rows
    # of that code carry; never by title, and never to a film that has another KOBIS code
    cd2id: dict[str, str] = {}
    rating2ids: dict[str, set[str]] = defaultdict(set)
    codes: dict[str, set[str]] = defaultdict(set)
    for f in films:
        fid = f.get("variant_of") or f["id"]
        ids = identifiers(f)
        for c in sorted(ids.get("kobis", ())):
            cd2id.setdefault(c, fid)
        codes[fid] |= ids.get("kobis", set())
        for no in ids.get("rating", ()):
            rating2ids[no].add(fid)
    lags_by_cd: dict[str, list[dict]] = defaultdict(list)
    for r in lag_rows:
        if isinstance(r, dict) and r.get("kobis_cd"):
            lags_by_cd[r["kobis_cd"]].append(r)
    bo_date = max(bo_days) if bo_days else ""
    bo_rows = []
    for r in bo_days.get(bo_date, []) if bo_date else []:
        fid = cd2id.get(r["movie_cd"])
        title = by_id[fid]["title"] if fid else ""
        open_dt = r.get("open_dt") or ""
        if not fid and r["movie_cd"] in lags_by_cd:
            lrs = lags_by_cd[r["movie_cd"]]
            title = lrs[0].get("title") or ""
            open_dt = lrs[0].get("open_dt") or open_dt
            cands = {c for lr in lrs for c in rating2ids.get(lr.get("rating_no") or "", ())
                     if not codes[c] or r["movie_cd"] in codes[c]}
            fid = next(iter(cands)) if len(cands) == 1 else None
        if fid:
            rel = (by_id[fid].get("release") or {}).get("date") or ""
            if len(rel) == 10 and open_dt < rel <= bo_date:
                open_dt = rel  # re-release: count days from the re-release
        dn = (dt.date.fromisoformat(bo_date) - dt.date.fromisoformat(open_dt)).days + 1 if open_dt else None
        bo_rows.append(dict(r=r["rank"], id=fid, t=title or f"제목 미수집 (KOBIS {r['movie_cd']})",
                            a=r["audience"], acc=r["audience_acc"], chg=r["rank_change"], nw=bool(r["new"]),
                            open=open_dt, dn=dn))
    bo_ids = {r["id"] for r in bo_rows if r["id"]}

    # ------------------------------------------------ per film: list item + detail
    index: list[dict] = []
    details: dict[str, dict] = {}
    for f in films:
        if f.get("variant_of"):
            continue
        st = f["stages"]
        rel = f.get("release") or {}
        tags = set(f.get("tags") or [])
        title, otitle = f.get("title") or "", f.get("original_title") or ""
        fmts = set(variant_formats.get(f["id"], set())) | fmt_of_text(title + " " + otitle) | editions.get(f["id"], set())
        kdet = st.get("kobis_detail") or {}
        for x in kdet.get("formats", []):
            xl = x.lower()
            if xl.startswith("4d"):
                fmts.add("4d")
            elif "imax" in xl:
                fmts.add("imax")
            elif "screenx" in xl:
                fmts.add("sx")
            elif "dolby" in xl:
                fmts.add("dolby")
        dist = next((c["name"] for c in kdet.get("companies", []) if c.get("role") == "배급사"), "") \
            or ((st.get("kobis") or {}).get("distributors") or [""])[0]
        dist = clean_co(dist)
        app = clean_co(f.get("applicant"))
        c = certainty(rel)
        c["ss"] = rel.get("signal_state")
        bk = st.get("booking")
        mb, lt, cg = st.get("megabox"), st.get("lotte"), st.get("cgv")
        now_ev = (f["id"] in bo_ids) or any((x or {}).get("status") == "now" for x in (mb, lt, cg)) \
            or bool(bk and bk.get("total_audience"))
        outside = (f.get("theatrical") or {}).get("class") == "unlikely" or "dormant" in tags
        cats = []
        if is_anime([f, *members.get(f["id"], [])], tags):  # official genre first, title heuristic only without one
            cats.append("ani")
        if "rerelease" in tags or (st.get("release_history") or {}).get("rereleases"):
            cats.append("rr")
        kind = screening_kind(title, tags, mb, lt, cg)
        if kind:
            cats.append(kind)
        chains = []
        if mb:
            chains.append("mb")
        if lt:
            chains.append("lt")
        if st.get("cgv") or "chain_cgv" in tags:
            chains.append("cgv")
        excl = []
        if (cg or {}).get("cgv_only") or "cgv_only" in ((cg or {}).get("labels") or []):
            excl.append("cgv")
        pg = st.get("program") or {}
        if (mb or {}).get("exclusive") or (pg.get("exclusive") and pg.get("chain") == "megabox"):
            excl.append("mb")
        if pg.get("exclusive") and pg.get("chain") == "cgv" and "cgv" not in excl:
            excl.append("cgv")
        if (lt or {}).get("badge") == "pick":
            excl.append("lt")
        uid = stable_id(members.get(f["id"], [f]))
        poster = Path(f["poster"]).name if f.get("poster") else None
        if poster and not poster_exists(poster):
            poster = None
        item: dict[str, Any] = dict(id=f["id"], t=title, ot=otitle, dist=dist, app=app, g=grade_of(st), d=c,
                                    cats=cats, fmts=sorted(fmts), ch=chains, now=now_ev, ox=outside, pst=poster,
                                    ex=excl)
        if uid != f["id"]:
            item["u"] = uid  # stable identity (stable_id); absent: the id itself
        if bk:
            item["bk"] = dict(r=bk.get("rank"), rate=bk.get("rate"), pa=bk.get("presale_audience"),
                              ta=bk.get("total_audience"))
        if mb:
            item["mb"] = mb.get("status")
        if lt:
            item["lt"] = lt.get("status")
        if cg:
            item["cg"] = cg.get("status")
        alts = sorted(alt_titles.get(f["id"], set()) - {title, otitle})
        fields = [x for x in (title, otitle, *alts, dist, app) if x]
        item["k"] = "|".join(dict.fromkeys(choseong(norm(x)) for x in fields))
        item["n"] = "|".join(dict.fromkeys(norm(x) for x in fields))
        index.append(item)

        # detail: evidence timeline [date, label, sub, source, real(1) / detected(0)]
        fs = f.get("first_seen") or {}
        tl: list[list] = []
        if st.get("intake"):
            i = st["intake"]
            tl.append([i.get("date") or "", "등급 심의 신청", f"접수 {i.get('receipt_no', '')}", "영등위", 1])
        if st.get("past_rating"):
            p = st["past_rating"]
            tl.append([p.get("date") or "", "이전 등급 결정 (재개봉 전 기록)",
                       GRADE_LONG.get(GRADE.get(p.get("grade") or "", ""), p.get("grade") or ""), "영등위", 1])
        for x in promo_dates(st.get("promo") or {}):
            tl.append([x, _trace_label("promo", st["promo"]), "개봉 준비 신호", "영등위", 1])
        if st.get("rating"):
            r = st["rating"]
            g = r.get("grade") or ""
            sub = " · ".join(x for x in (GRADE_LONG.get(GRADE.get(g, ""), g), r.get("runtime") or "") if x)
            tl.append([r.get("date") or "", "등급 결정", sub, "영등위", 1])
        if st.get("ors_ad"):
            a = st["ors_ad"]
            label = _trace_label("ors_ad", a)
            dates = a.get("dates") or ([a["first_date"]] if a.get("first_date") else [])
            for x in dates:
                tl.append([x, label, "개봉 준비 신호", "영등위", 1])
            if not dates and day(fs.get("ors_ad")):
                tl.append([day(fs.get("ors_ad")), label, "접수 전 (신청완료)", "영등위", 0])
        if st.get("kobis"):
            k = st["kobis"]
            sch = st.get("schedule") or {}
            when = k.get("open_dt") or sch.get("date") or (mon(sch["month"], cur_year) if sch.get("month") else "")
            if len(when) == 10:
                when = kd(when)
            tl.append([day(fs.get("kobis")), "KOBIS 등록 확인", f"개봉일 {when}" if when else "개봉일 미정", "KOBIS", 0])
        for key, nm in (("megabox", "메가박스"), ("lotte", "롯데시네마")):
            cc = st.get(key)
            if cc:
                rd = cc.get("release_date") or ""
                tl.append([day(fs.get(key)), f"{nm} 목록에 등장",
                           CHAIN_ST.get(cc.get("status") or "", "") + (f" · 개봉 {kd(rd)}" if len(rd) == 10 else ""), nm, 0])
        if pg:
            tl.append([day(fs.get("program")),
                       f"메가박스 {pg.get('program') or '기획전'}" + (" · 단독 상영" if pg.get("exclusive") else ""),
                       f"상영 {kd(pg['date'])}" if pg.get("date") else "", "메가박스", 0])
        tl = [x for x in tl if x[0]]
        tl.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        comps = [[cc["role"], clean_co(cc.get("name"))] for cc in kdet.get("companies", [])
                 if cc.get("role") in ("배급사", "수입사", "제작사")]
        chain_seen = [day(fs[k]) for k in ("megabox", "lotte") if day(fs.get(k))]
        details[f["id"]] = dict(
            tl=tl, ft=first_sighting(members.get(f["id"], [f]), base, kobis_code, keep_boot=True),
            syn=kdet.get("synopsis", ""), comps=comps, rt=runtime_of(st),
            co=(st.get("rating") or {}).get("country", ""), q=(rel.get("quantiles") or None),
            hist=(f.get("theatrical") or {}).get("applicant_history"),
            nxt=NEXT.get(rel.get("signal_state") or "") if c["p"] != "day" or c["date"] > last_date else None,
            mbd=(mb or {}).get("release_date"), ltd=(lt or {}).get("release_date"),
            cgd=(cg or {}).get("release_date") or None, q20=rel.get("q20") if c["p"] == "est" else None, bkas=(bk or {}).get("as_of"),
            ltpick=(lt or {}).get("badge") == "pick", prog=pg.get("program") if pg else None,
            mbno=((mb or {}).get("movie_nos") or [None])[0], ltno=(lt or {}).get("movie_code") or None,
            cgno=(cg or {}).get("movie_no") or None,
            stages=dict(intake=(st.get("intake") or {}).get("date"), rating=(st.get("rating") or {}).get("date"),
                        promo=(st.get("promo") or {}).get("first_date"),
                        ad=(st.get("ors_ad") or {}).get("first_date") or day(fs.get("ors_ad")) or None,
                        kobis=day(fs.get("kobis")) or None, chain=min(chain_seen, default=None)))

    # ------------------------------------------------ first sightings
    idx_by_id = {x["id"]: x for x in index}
    keep_from = (dt.date.fromisoformat(last_date) - dt.timedelta(days=EVENT_KEEP_DAYS)).isoformat()
    sightings: list[dict] = []
    for f in films:
        it = idx_by_id.get(f["id"])
        if not it or it["ox"]:
            continue
        s = first_sighting(members.get(f["id"], [f]), base, kobis_code)
        if s and s["d"] >= keep_from:
            sightings.append({"id": f["id"], **s})
    sightings.sort(key=lambda s: (s["d"], s["id"]), reverse=True)
    fs_by_id = {s["id"]: s for s in sightings}

    # ------------------------------------------------ events
    # One event = one thing that happened to one film. d = the date shown: the real-world date when
    # the source records one (r = 1), else the day it was first seen (r = 0). Events with r = 0 seen
    # on the source's first collection day are the baseline import and are dropped. f = 1: the
    # film's first sighting.
    events: list[dict] = []
    for f in films:
        it = idx_by_id.get(f["id"])
        if not it or it["ox"]:
            continue
        c = it["d"]
        if c["p"] == "day" and c["date"] <= last_date:
            continue  # already open: lives in 상영 중, not in the radar
        st, fs = f["stages"], f.get("first_seen") or {}
        fsd = (fs_by_id.get(f["id"]) or {}).get("d")

        def ev(kind: str, date: str, real: int, x: str = "", n: int | None = None) -> None:
            if date and date >= keep_from:
                e = dict(id=f["id"], k=kind, d=date, r=real, x=x)
                if n is not None:
                    e["n"] = n  # forecast change: its history index, unique per change
                if date == fsd:
                    e["f"] = 1
                events.append(e)

        def fresh(stage: str) -> bool:
            return bool(day(fs.get(stage))) and day(fs.get(stage)) > base.get(stage, "")

        if (st.get("intake") or {}).get("date"):
            ev("filed", st["intake"]["date"], 1)
        if (st.get("rating") or {}).get("date"):
            ev("rated", st["rating"]["date"], 1, GRADE_LONG.get(GRADE.get(st["rating"].get("grade") or "", ""), ""))
        if st.get("promo"):
            p = st["promo"]
            ds = [x for x in (p.get("dates_seen") or [p.get("first_date") or ""]) if x and x >= keep_from]
            if ds:
                ev("promo", ds[-1], 1, "·".join((KIND.get(k) or str(k)) for k in p.get("kinds", [])))
        if st.get("ors_ad"):
            a = st["ors_ad"]
            kinds = "·".join(AD_KIND[k] for k in a.get("kinds", []) if k in AD_KIND)
            dates = a.get("dates") or ([a["first_date"]] if a.get("first_date") else [])
            if dates:
                ev("ad", dates[-1], 1, kinds)
            elif fresh("ors_ad"):
                ev("ad", day(fs.get("ors_ad")), 0, kinds)
        if fresh("kobis"):
            k = st["kobis"]
            sch = st.get("schedule") or {}
            dday = k.get("open_dt") or sch.get("date")
            if dday:
                ev("date", day(fs.get("kobis")), 0, dday)
            elif sch.get("month"):
                ev("month", day(fs.get("kobis")), 0, sch["month"])
            else:
                ev("kobis", day(fs.get("kobis")), 0)
        for key in ("megabox", "lotte"):
            if st.get(key) and fresh(key):
                ev("open" if st[key].get("status") == "now" else "chain", day(fs.get(key)), 0, key)
        if st.get("program") and fresh("program") and st["program"].get("exclusive"):
            ev("excl", day(fs.get("program")), 0, st["program"].get("chain", ""))
        for d, kind, words, n in forecast_events(hist.get(f["id"], []), cur_year):
            ev(kind, d, 0, words, n)
    events.sort(key=lambda e: (e["d"], e["id"], e["k"], e.get("n", 0), e["x"]), reverse=True)

    bkas = max(((f["stages"].get("booking") or {}).get("as_of") or "" for f in films), default="")
    bundle = dict(first=first_date, last=last_date, th=dict(likely=LIKELY, possible=POSSIBLE),
                  bo=dict(date=bo_date, rows=bo_rows), bkas=bkas, films=index, ev=events, fs=sightings,
                  alias=aliases)

    # ------------------------------------------------ feeds
    per: dict[tuple[str, str], list[dict]] = {}
    for e in events:
        per.setdefault((e["d"], e["id"]), []).append(e)
    items = []
    for (d, fid), es in list(per.items())[:FEED_ITEMS]:
        words = " · ".join(dict.fromkeys(
            e["x"] if e["k"] in FORECAST_KINDS
            else (CHN.get(e["x"], "") + " " if e["k"] in ("chain", "open") else "") + EVW[e["k"]] for e in es))
        items.append(f"<item><title>{xesc(idx_by_id[fid]['t'])} · {xesc(words)}</title>"
                     f"<description>{xesc(cert_text(idx_by_id[fid]['d'], last_date))}</description>"
                     f"<link>{xesc(film_url(fid))}</link>"
                     f"<guid isPermaLink=\"false\">{xesc(idx_by_id[fid].get('u') or fid)}-{d}</guid>"
                     f"<pubDate>{_pub(d)}</pubDate></item>")
    feed = _rss("무비 레이더", "개봉일 확정·변경, 예상일 변화, 극장 목록 등장, 등급·심의 소식", items)
    new_items = []
    for s in sightings:
        it = idx_by_id[s["id"]]
        who = it["dist"] or it["app"]
        desc = " · ".join(x for x in (s["w"], who, kd(s["d"]) + (" 확인" if not s["r"] else "")) if x)
        new_items.append(f"<item><title>{xesc(it['t'])}</title><description>{xesc(desc)}</description>"
                         f"<link>{xesc(film_url(s['id']))}</link>"
                         f"<guid isPermaLink=\"false\">new-{xesc(s['g'])}</guid><pubDate>{_pub(s['d'])}</pubDate></item>")
    new_feed = _rss("무비 레이더 · 새로 포착", "처음 포착된 영화: 등급·광고 심의 신청, KOBIS·극장 목록 첫 등장", new_items)

    outputs = {
        f"data/{SITE_DIR}/{INDEX_JS}": _js("MR", bundle),
        f"data/{SITE_DIR}/{DETAILS_JS}": _js("MRD", details),
        FEED: feed,
        NEW_FEED: new_feed,
    }
    stats = {"films": len(index), "events": len(events), "first_sightings": len(sightings),
             "merged": merge_log}
    return outputs, stats


def _load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write(data_dir: Path, today: str | None = None) -> dict:
    """Build from ``data_dir`` and write the outputs under its parent (the repository root),
    rewriting only files whose content changed. With ``today`` (the run's KST date), first update
    ``data_dir/forecast_history.json``; without it the history is only read. -> stats with
    ``written`` (paths, relative to the repository root)."""
    films = _load(data_dir / "films.json", {"films": []})
    films = films["films"] if isinstance(films, dict) else films
    bo_days = _load(data_dir / "boxoffice.json", {}).get("days", {})
    lags = _load(data_dir / "release_lags.json", {}).get("rows", [])
    posters = data_dir / "posters"
    history = _load(data_dir / HISTORY, {"schema": 1, "films": {}})
    outputs: dict[str, str] = {}
    if today:
        history = update_history(history, films, today)
        outputs[f"data/{HISTORY}"] = json.dumps(history, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    built, stats = build(films, bo_days, lags, lambda name: (posters / name).is_file(), history)
    outputs.update(built)
    outputs[f"data/{SITE_DIR}/{STATUS_JS}"] = _js("MRS", status(_load(data_dir / "meta.json", {})))
    root = data_dir.parent
    written = []
    for rel, text in outputs.items():
        rel = rel.replace("data/", f"{data_dir.name}/", 1) if rel.startswith("data/") else rel
        path = root / rel
        data = text.encode("utf-8")
        if path.exists() and path.read_bytes() == data:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        written.append(rel)
    stats["written"] = written
    return stats
