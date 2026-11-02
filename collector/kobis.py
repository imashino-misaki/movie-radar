"""KOBIS (영화관입장권통합전산망) Open API: release dates and distributors.

Endpoints (REST, JSON), documented at
https://www.kobis.or.kr/kobisopenapi/homepg/apiservice/searchServiceInfo.do
  searchMovieList.json  key, curPage, itemPerPage, openStartDt/openEndDt (YYYY open year)
                        -> movieListResult.movieList[]: movieCd, movieNm, movieNmEn,
                           prdtYear, openDt, typeNm, prdtStatNm, nationAlt, genreAlt,
                           repNationNm, repGenreNm, directors[], companys[] (제작사)
  searchMovieInfo.json  key, movieCd
                        -> movieInfoResult.movieInfo: movieCd, movieNm, movieNmEn,
                           movieNmOg, openDt, prdtStatNm, nations[{nationNm}],
                           genres[{genreNm}], audits[{auditNo, watchGradeNm}],
                           companys[{companyCd, companyNm, companyNmEn, companyPartNm}]
  errors                -> {"faultInfo": {"message": ..., "errorCode": ...}}
The docs list field names only; the JSON wrapper names above are the API's
well-known response shape (not verifiable here without a key).

The API key comes only from the KOBIS_API_KEY environment variable. It is added
to the request parameters at the last moment and never logged, stored, or put
in an exception message: every message that leaves this module goes through
redact(). Without a key the stage is skipped.

Per run: list pages for the current and next open year, keep rows that are
개봉예정 or opened within the last RECENT_DAYS days, drop 성인물 genres, then
fetch movie info for new movieCds (and refresh upcoming titles at most every
REFRESH_DAYS days) within a hard per-run call cap. Results are cached in
data/kobis.json. Entries that would not be published are cached without titles.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from urllib.parse import quote

from .http import BlockedError, check_block
from .privacy import mask_applicant
from .companies import company_key
from .text import grade_key

log = logging.getLogger(__name__)

ENV_KEY = "KOBIS_API_KEY"
BASE = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/"
ITEM_PER_PAGE = 100
RECENT_DAYS = 14
REFRESH_DAYS = 7
# KOBIS does not publish a daily quota on its docs pages; ~3,000 calls/day is the
# commonly cited figure. Two scheduled runs x 300 stays far below it.
CALL_CAP = 300

ADULT_GENRE = "성인물"
RESTRICTED_GRADES = ("청소년관람불가", "제한상영가")
UPCOMING_STATUS = "개봉예정"
# statuses of a film registered for release (the website list's status filter
# also gives 개봉준비: a release being prepared, often without a date yet)
UPCOMING_STATUSES = (UPCOMING_STATUS, "개봉준비")
DISTRIBUTOR_PART = "배급사"


class KobisError(RuntimeError):
    """KOBIS answered with faultInfo or an unexpected payload."""


class BudgetExceeded(KobisError):
    pass


def get_key() -> str:
    return os.environ.get(ENV_KEY, "").strip()


def redact(text: str, key: str) -> str:
    if key:
        for form in {key, quote(key, safe="")}:
            text = text.replace(form, "***")
    return text


class KobisClient:
    """Wraps PoliteClient; owns the key and the per-run call budget."""

    def __init__(self, client, key: str, cap: int = CALL_CAP, base: str = BASE):
        self._client, self._key, self.cap, self.calls = client, key, cap, 0
        self._base = base

    @property
    def remaining(self) -> int:
        return self.cap - self.calls

    def call(self, endpoint: str, params: dict) -> dict:
        if self.calls >= self.cap:
            raise BudgetExceeded(f"call cap {self.cap} reached")
        self.calls += 1
        log.info("kobis %s %s", endpoint, params)  # params never contain the key
        try:
            resp = self._client.get(self._base + endpoint, params={**params, "key": self._key})
            text = resp.text
        except BlockedError as exc:
            raise BlockedError(redact(f"{endpoint}: {exc}", self._key)) from None
        except Exception as exc:  # noqa: BLE001 - re-raise without anything key-bearing
            raise KobisError(redact(f"{endpoint}: {type(exc).__name__}: {exc}", self._key)) from None
        try:
            data = resp.json()
        except ValueError:
            check_block(text, markup_ok=False)
            raise KobisError(f"{endpoint}: response is not JSON") from None
        if not isinstance(data, dict):
            raise KobisError(f"{endpoint}: unexpected payload")
        fault = data.get("faultInfo")
        if fault:
            code = redact(str(fault.get("errorCode", "")), self._key)
            msg = redact(str(fault.get("message", "")), self._key)[:120]
            raise KobisError(f"{endpoint}: faultInfo errorCode={code} {msg}")
        return data


# ---------------------------------------------------------------- parsing

def parse_list(data: dict) -> tuple[list[dict], int]:
    res = data.get("movieListResult")
    if not isinstance(res, dict) or not isinstance(res.get("movieList"), list):
        raise KobisError("searchMovieList: missing movieListResult.movieList")
    return res["movieList"], int(res.get("totCnt") or 0)


def parse_info(data: dict) -> dict:
    res = data.get("movieInfoResult")
    info = res.get("movieInfo") if isinstance(res, dict) else None
    if not isinstance(info, dict) or not info.get("movieCd"):
        raise KobisError("searchMovieInfo: missing movieInfoResult.movieInfo")
    return {
        "movieCd": info["movieCd"],
        "names": {"ko": info.get("movieNm", ""), "en": info.get("movieNmEn", ""),
                  "og": info.get("movieNmOg", "")},
        "openDt": info.get("openDt", ""),
        "prdtStatNm": info.get("prdtStatNm", ""),
        "genres": [g.get("genreNm", "") for g in info.get("genres", [])],
        "nations": [n.get("nationNm", "") for n in info.get("nations", [])],
        # masked before caching: data/kobis.json is published too
        "distributors": [mask_applicant(c.get("companyNm", "")) for c in info.get("companys", [])
                         if c.get("companyPartNm") == DISTRIBUTOR_PART],
        "audits": [{"auditNo": audit_no(a), "watchGradeNm": a.get("watchGradeNm", "")}
                   for a in info.get("audits", [])],
    }


def iso_date(yyyymmdd: str) -> str:
    s = (yyyymmdd or "").replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else ""


def keep_row(row: dict, today: dt.date) -> bool:
    """List filter: upcoming or recently opened, and not an adult genre."""
    if ADULT_GENRE in (row.get("genreAlt") or ""):
        return False
    if row.get("prdtStatNm") in UPCOMING_STATUSES:
        return True
    open_dt = iso_date(row.get("openDt", ""))
    return bool(open_dt) and open_dt >= (today - dt.timedelta(days=RECENT_DAYS)).isoformat()


# ---------------------------------------------------------------- publish rule

def audit_no(audit: dict) -> str:
    """KOBIS rating number, stripped: KOBIS pads some with spaces (checked
    2026-11-01: " 2026-MF02931", "2024-MF00695 "), which broke the join with
    the KMRB rating number."""
    return (audit.get("auditNo") or "").strip()


def trusted_distributor(entry: dict, trusted: set[str]) -> bool:
    """A distributor KOBIS lists is trusted (a trusted KMRB applicant, or a
    company the KOBIS register check trusts). Compared without legal form,
    spaces and punctuation: KOBIS writes "워너브러더스 코리아(주)" for the
    applicant KMRB lists as "워너브러더스코리아(주)" (checked 2026-11-01)."""
    keys = {company_key(t) for t in trusted}
    return any(company_key(mask_applicant(d)) in keys for d in entry.get("distributors", []))


def blocked_distributor(entry: dict, blocked: frozenset | set) -> bool:
    """A distributor is on the manual block list (lists.py; company keys)."""
    return any(company_key(mask_applicant(d)) in blocked for d in entry.get("distributors", []))


def publishable(entry: dict, trusted: set[str], withheld_audits: set[str],
                blocked: frozenset | set = frozenset()) -> bool:
    """Not 성인물, not matching a withheld KMRB rating, and a trusted
    distributor when the film has a restricted (청소년관람불가 / 제한상영가)
    audit or no opening date yet. A film without a date and without a rating
    carries the least information: the website's upcoming list also holds
    films registered years ago and never updated, some of them adult titles
    filed under other genres, so only a trusted distributor publishes it."""
    if any(ADULT_GENRE in g for g in entry.get("genres", [])) or blocked_distributor(entry, blocked):
        return False
    audits = entry.get("audits", [])
    if any(audit_no(a) in withheld_audits for a in audits):
        return False
    # old audits keep the old grade name (연소자관람불가, 18세관람가): compared as 청소년관람불가
    restricted = any(grade_key(a.get("watchGradeNm", "")) in RESTRICTED_GRADES for a in audits)
    if restricted or not entry.get("openDt"):
        return trusted_distributor(entry, trusted)
    return True


def strip_names(entry: dict) -> dict:
    """Withheld entries keep only what is needed to re-evaluate them - no titles."""
    out = {k: v for k, v in entry.items() if k != "names"}
    out["withheld"] = True
    return out


def is_upcoming(entry: dict, today: dt.date) -> bool:
    return (entry.get("prdtStatNm") in UPCOMING_STATUSES
            or iso_date(entry.get("openDt", "")) >= today.isoformat())


def needs_info(entry: dict | None, today: dt.date, trusted: set[str],
               withheld_audits: set[str], blocked: frozenset | set = frozenset()) -> bool:
    if entry is None or not entry.get("fetched_at"):
        return True
    if entry.get("withheld") and publishable(entry, trusted, withheld_audits, blocked):
        return True  # became publishable: its titles were never stored
    if is_upcoming(entry, today):
        fetched = dt.date.fromisoformat(entry["fetched_at"])
        return (today - fetched).days >= REFRESH_DAYS
    return False


# ---------------------------------------------------------------- fetch

def fetch(kc: KobisClient, cache: dict, today: dt.date, trusted: set[str],
          withheld_audits: set[str], stamp: str | None = None,
          extra_codes: tuple[str, ...] = (), undated_codes: tuple[str, ...] = (),
          blocked: frozenset | set = frozenset()) -> dict:
    """Fetch list + needed info into `cache` (movieCd -> entry). Returns stats.
    A list failure raises (the caller keeps the old cache); running out of the
    call budget during info calls only defers the rest to the next run.

    `extra_codes`: KOBIS codes a cinema list gives (Lotte); their info is
    fetched once (refreshed like the others while upcoming) for the first
    opening date, after the list's own titles. They do not become KOBIS-only
    records (`kept_ids` lists only the list's titles).

    `undated_codes`: films the website lists as 개봉예정 / 개봉준비 (not 성인물);
    the Open API list misses those without an opening date. They are kept like
    the list's own titles (info fetched after them; published per
    publishable())."""
    rows: list[dict] = []
    page = 1
    while True:
        params = {"curPage": str(page), "itemPerPage": str(ITEM_PER_PAGE),
                  "openStartDt": str(today.year), "openEndDt": str(today.year + 1)}
        page_rows, total = parse_list(kc.call("searchMovieList.json", params))
        if not page_rows and len(rows) < total:
            raise KobisError(f"searchMovieList: page {page} empty, {len(rows)}/{total} rows")
        rows.extend(page_rows)
        if not page_rows or page * ITEM_PER_PAGE >= total:
            break
        page += 1

    adult = sum(1 for r in rows if ADULT_GENRE in (r.get("genreAlt") or ""))
    kept = {r["movieCd"]: r for r in rows if r.get("movieCd") and keep_row(r, today)}
    today_s = today.isoformat()           # KST date: drives the refresh rule (fetched_at)
    stamp = stamp or today_s              # UTC run date: first_seen / last_seen
    for cd, r in kept.items():  # list fields are fresh every run; no info call needed
        if cd in cache:
            cache[cd].update(openDt=r.get("openDt", ""), prdtStatNm=r.get("prdtStatNm", ""),
                             last_seen=stamp)

    todo = [cd for cd in kept if needs_info(cache.get(cd), today, trusted, withheld_audits, blocked)]
    todo.sort(key=lambda cd: (cd in cache, cache.get(cd, {}).get("fetched_at", "")))  # new first
    undated = sorted({cd for cd in undated_codes if cd and cd not in kept})
    todo += sorted((cd for cd in undated if needs_info(cache.get(cd), today, trusted, withheld_audits, blocked)),
                   key=lambda cd: (cd in cache, cache.get(cd, {}).get("fetched_at", "")))
    extra = sorted({cd for cd in extra_codes if cd and cd not in kept and cd not in undated
                    and needs_info(cache.get(cd), today, trusted, withheld_audits, blocked)})
    todo += sorted(extra, key=lambda cd: (cd in cache, cache.get(cd, {}).get("fetched_at", "")))
    fetched = deferred = 0
    for cd in todo:
        if kc.remaining <= 0:
            deferred = len(todo) - fetched
            break
        info = parse_info(kc.call("searchMovieInfo.json", {"movieCd": cd}))
        prev = cache.get(cd, {})
        info.update(fetched_at=today_s, first_seen=prev.get("first_seen", stamp),
                    last_seen=stamp)
        cache[cd] = info
        fetched += 1

    # never store titles of entries we would not publish: every cached entry,
    # including ones no longer kept (e.g. the list now marks them 성인물)
    adult_now = {r["movieCd"] for r in rows if ADULT_GENRE in (r.get("genreAlt") or "")}
    for cd, entry in list(cache.items()):
        entry["distributors"] = [mask_applicant(d) for d in entry.get("distributors", [])]
        if "names" in entry and (cd in adult_now or not publishable(entry, trusted, withheld_audits, blocked)):
            cache[cd] = strip_names(entry)

    return {"list_rows": len(rows), "adult_genre_dropped": adult, "kept": len(kept),
            "info_fetched": fetched, "info_deferred": deferred, "extra_codes": len(extra),
            "undated_codes": len(undated), "kept_ids": sorted(set(kept) | set(undated))}
