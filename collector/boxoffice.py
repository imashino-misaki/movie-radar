"""KOBIS Open API daily box office: audience numbers of films now showing.

Endpoint (REST, JSON), documented at
https://www.kobis.or.kr/kobisopenapi/homepg/apiservice/searchServiceInfo.do?serviceId=searchDailyBoxOffice
  boxoffice/searchDailyBoxOfficeList.json
      key, targetDt (yyyymmdd), itemPerPage (default and maximum 10),
      multiMovieYn / repNationCd / wideAreaCd (filters, default: all)
      -> boxOfficeResult: boxofficeType, showRange,
         dailyBoxOfficeList[]: rnum, rank, rankInten, rankOldAndNew (OLD / NEW),
         movieCd, movieNm, openDt, salesAmt, salesShare, salesInten,
         salesChange, salesAcc, audiCnt, audiInten, audiChange, audiAcc,
         scrnCnt, showCnt (all strings)
The docs list field names only; the boxOfficeResult / dailyBoxOfficeList
wrapper is the API's well-known response shape (not verifiable here without a
key). The key handling is kobis.py's (KobisClient, redact).

A day's box office is final once published, so each date is fetched once and
kept in data/boxoffice.json ({date: [rows]}). Per run: the days from
yesterday (KST) back WINDOW_DAYS days that are not stored yet, newest first,
at most MAX_CALLS calls. Yesterday's list may not be published yet when the
run happens: an empty answer for yesterday is not stored and asked again next
run; an empty answer for an older day is stored as [] (no data for that day).
Days older than the window are dropped. Steady state: one call a day, plus
one a run while yesterday's list is not out yet.

Only the top 10 is available (itemPerPage maximum). Rows are stored without
titles: the record the row joins has the title, and rows that join no record
are only counted.
"""
from __future__ import annotations

import datetime as dt

from . import kobis
from .kobis import KobisError, iso_date
from .kobis_web import record_codes

BASE = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/"
ENDPOINT = "searchDailyBoxOfficeList.json"
ITEM_PER_PAGE = 10  # the documented maximum
# Daily audience swings with the weekday (weekends draw several times a
# weekday), so a trend compares a day with the same weekday a week earlier:
# two full weeks give this week against last week.
WINDOW_DAYS = 14
# The first run fills the whole window; later runs need one call a day. 14
# calls next to the ~46 the other KOBIS calls make per run, against the
# 3,000 calls/day the key allows.
MAX_CALLS = WINDOW_DAYS

ROW_FIELDS = ("rank", "movie_cd", "open_dt", "audience", "audience_acc", "sales", "sales_acc",
              "screens", "shows", "rank_change", "new")
STAGE_FIELDS = tuple(f for f in ROW_FIELDS if f not in ("movie_cd", "open_dt"))


def _int(value) -> int:
    try:
        return int(str(value).replace(",", "").strip() or 0)
    except ValueError:
        return 0


def parse(data: dict) -> list[dict]:
    res = data.get("boxOfficeResult")
    if not isinstance(res, dict) or not isinstance(res.get("dailyBoxOfficeList"), list):
        raise KobisError("searchDailyBoxOfficeList: missing boxOfficeResult.dailyBoxOfficeList")
    rows = []
    for r in res["dailyBoxOfficeList"]:
        if not r.get("movieCd"):
            continue
        rows.append({
            "rank": _int(r.get("rank")),
            "movie_cd": r["movieCd"],
            "open_dt": iso_date(r.get("openDt", "")),
            "audience": _int(r.get("audiCnt")),
            "audience_acc": _int(r.get("audiAcc")),
            "sales": _int(r.get("salesAmt")),
            "sales_acc": _int(r.get("salesAcc")),
            "screens": _int(r.get("scrnCnt")),
            "shows": _int(r.get("showCnt")),
            "rank_change": _int(r.get("rankInten")),
            "new": r.get("rankOldAndNew") == "NEW",
        })
    return sorted(rows, key=lambda r: r["rank"])


def window(today_kst: dt.date) -> list[str]:
    """The window's dates, newest (yesterday) first."""
    return [(today_kst - dt.timedelta(days=i)).isoformat() for i in range(1, WINDOW_DAYS + 1)]


def fetch(kc: kobis.KobisClient, history: dict, today_kst: dt.date) -> dict:
    """Fetch the window's missing days into `history` (date -> rows, in place)
    and drop days older than the window. Raises on a failed call (the caller
    keeps the stored history)."""
    dates = window(today_kst)
    yesterday = dates[0]
    for d in [d for d in history if d not in dates]:
        del history[d]
    missing = [d for d in dates if d not in history]
    fetched, pending = [], False
    for d in missing:
        if kc.remaining <= 0:
            break
        rows = parse(kc.call(ENDPOINT, {"targetDt": d.replace("-", ""),
                                        "itemPerPage": str(ITEM_PER_PAGE)}))
        if not rows and d == yesterday:
            pending = True  # not published yet: ask again next run
            continue
        history[d] = rows
        fetched.append(d)
    return {"fetched": len(fetched), "deferred": len([d for d in dates if d not in history]) - pending,
            "yesterday_pending": pending, "days_stored": len(history)}


def _codes(rec: dict) -> set[str]:
    st = rec["stages"]
    return (record_codes(rec) | {st.get("kobis_detail", {}).get("kobis_cd", ""),
                                 st.get("booking", {}).get("movie_cd", "")}) - {""}


def apply(films: list[dict], history: dict) -> dict:
    """Attach stage "boxoffice" (in place): the film's latest stored day
    ({date, STAGE_FIELDS...}) and `series`, [date, audience] of every stored
    day it was in the top 10, oldest first. Records the history does not
    list lose the stage. Rows of the latest day that join no record are
    counted (`top10_unmatched`), not added."""
    by_cd: dict[str, list[tuple[str, dict]]] = {}
    for d in sorted(history):
        for row in history[d]:
            by_cd.setdefault(row["movie_cd"], []).append((d, row))
    n = 0
    all_codes: set[str] = set()
    for rec in films:
        codes = _codes(rec)
        all_codes |= codes
        hits = [by_cd[c] for c in sorted(codes) if c in by_cd]
        if not hits:
            rec["stages"].pop("boxoffice", None)
            continue
        days = max(hits, key=lambda h: (h[-1][0], len(h)))  # the code seen most recently
        d, row = days[-1]
        rec["stages"]["boxoffice"] = {"date": d, **{k: row[k] for k in STAGE_FIELDS},
                                      "series": [[day, r["audience"]] for day, r in days]}
        n += 1
    latest = max((d for d in history if history[d]), default="")
    return {"with_boxoffice": n, "latest_day": latest,
            "top10_unmatched": sum(1 for r in history.get(latest, [])
                                   if r["movie_cd"] not in all_codes)}
