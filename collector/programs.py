"""Chain exclusive programmes: stage "program" = {chain, program, month, date,
exclusive, source_url}.

Megabox "MEGA ONLY" (its exclusive-release brand): the movie page's MEGA ONLY
tab (megabox.co.kr/movie/singl) loads the list endpoint of megabox.py with
specialType=singl (checked 2026-11-01: 6 films, the same 6 for pageType
ticketing and rfilmDe). Each month's lineup is announced as an event "MEGA
ONLY 2026년 11월호" (event list endpoint, searched by that title; the current
event and the ended ones are two requests). The lineup itself is one image
with no text (2026-11-01: 1100 x 12897 px), so titles announced before
Megabox lists them cannot be read; a listed film gets the month of the lineup
whose event period holds its release date. So Megabox costs three requests.
The general list's singlPlayAt flag (megabox.py, the main page's MEGA ONLY
badge) is a different thing: exclusive screenings, re-releases and events
included (18 rows on 2026-11-01, 4 of them in the tab).

Lotte: the movie list rows carry SoloOpenYN (단독 개봉); no row had Y on
2026-11-01 (52 now showing, 39 upcoming), so the flag is passed through
untested against a real exclusive. Lotte's 롯시픽 badge (TagDivCd 2) is its
exclusive line-up: the November 2026 press release ("11월 단독 개봉 콘텐츠")
named 7 films and the 6 already listed all carried it (checked 2026-11-02), so
a 롯시픽 row is an exclusive programme entry named "롯시픽". 롯시플 (live
viewings) is not exclusive.

CGV: the uploaded chart (cgv.py) labels films of its CGV Only tab or with the
CGV ONLY icon "cgv_only"; no request.

Programmes only join records (by the chain's own film number, else by title);
a programme film no record has is counted, never made into a record.
"""
from __future__ import annotations

import re

from . import megabox
from .http import FetchError, check_block
from .text import TitleIndex, clean

MEGABOX_TAB_URL = "https://www.megabox.co.kr/movie/singl"
MEGABOX_EVENT_LIST = "https://www.megabox.co.kr/on/oh/ohe/Event/eventMngDiv.do"
MEGABOX_EVENT_URL = "https://www.megabox.co.kr/event/detail?eventNo="
MEGABOX_EVENT_REFERER = "https://www.megabox.co.kr/event/movie"
MEGA_ONLY = "MEGA ONLY"
# the event page's own page sizes: 1000 for current events, 12 for ended ones
EVENT_PAGE = {"ONG": "1000", "END": "12"}
LOTTE_URL = "https://www.lottecinema.co.kr/NLCHS/Movie/List?flag=1"
LOTTE_PROGRAM = "단독개봉"
LOTTE_PICK = "롯시픽"
CGV_URL = "https://cgv.co.kr/cnm/cgvChart/movieChart"
CGV_PROGRAM = "CGV ONLY"
CHAIN_ORDER = ("megabox", "lotte", "cgv")

_ITEM = re.compile(r'<li[^>]*>.*?data-no="(\d+)".*?<p class="tit">(.*?)</p>.*?'
                   r'<p class="date">(.*?)</p>', re.S)
_ISSUE = re.compile(r"(\d{2,4})\s*년\s*(\d{1,2})\s*월호")
_PERIOD = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\s*~\s*(\d{4})\.(\d{2})\.(\d{2})")


def parse_events(text: str) -> list[dict]:
    """Monthly MEGA ONLY lineup events of an event-list answer:
    [{"event_no", "title", "month", "start", "end"}]."""
    if "event-list" not in text:
        check_block(text, markup_ok=False)
        raise FetchError("megabox events: list markup missing")
    out = []
    for no, title, date in _ITEM.findall(text):
        title, date = clean(title), clean(date)
        issue, period = _ISSUE.search(title), _PERIOD.search(date)
        if MEGA_ONLY not in title.upper() or not issue or not period:
            continue  # stamp events, short-film curation, ... (no monthly lineup)
        year = int(issue.group(1))
        year += 2000 if year < 100 else 0
        g = period.groups()
        out.append({"event_no": no, "title": title, "month": f"{year:04d}-{int(issue.group(2)):02d}",
                    "start": f"{g[0]}-{g[1]}-{g[2]}", "end": f"{g[3]}-{g[4]}-{g[5]}"})
    return out


def event_body(state: str) -> dict:
    return {"currentPage": "1", "recordCountPerPage": EVENT_PAGE[state], "eventStatCd": state,
            "eventTitle": MEGA_ONLY, "eventDivCd": "", "eventTyCd": ""}


def tab_body() -> dict:
    return {**megabox.body(1), "pageType": "ticketing", "specialType": "singl"}


def fetch_megabox(client) -> dict:
    """{"rows": MEGA ONLY tab rows, "events": monthly lineups, "stats"}."""
    resp = client.post(megabox.LIST_URL, json=tab_body(), headers={"Referer": MEGABOX_TAB_URL})
    rows, total = megabox.parse_page(resp.text)
    if len(rows) < total:
        raise FetchError(f"megabox MEGA ONLY tab: {len(rows)}/{total} rows")
    events: dict[str, dict] = {}
    for state in EVENT_PAGE:
        resp = client.post(MEGABOX_EVENT_LIST, json=event_body(state),
                           headers={"Referer": MEGABOX_EVENT_REFERER})
        for e in parse_events(resp.text):
            events.setdefault(e["event_no"], e)
    evs = sorted(events.values(), key=lambda e: e["start"])
    return {"rows": rows, "events": evs, "stats": {
        "megabox_tab_rows": len(rows), "megabox_lineups": [e["month"] for e in evs]}}


def lineup_for(events: list[dict], date: str) -> dict | None:
    """The lineup event whose period holds `date` (the latest one if several)."""
    found = [e for e in events if date and e["start"] <= date <= e["end"]]
    return max(found, key=lambda e: e["start"]) if found else None


def _entry(chain, program, month, date, exclusive, url) -> dict:
    return {"chain": chain, "program": program, "month": month, "date": date,
            "exclusive": exclusive, "source_url": url}


def apply(films: list[dict], megabox_res: dict | None, lotte_rows: list[dict] | None,
          cgv_rows_seen: bool, stamp: str) -> dict:
    """Set stage "program" on records in place (pure, no I/O). A record keeps
    its last programme when the programme no longer lists it."""
    found: dict[str, list[dict]] = {}
    stats = {"program_rows": 0, "program_matched": 0, "program_unmatched": 0}
    index = TitleIndex(films, wanted=[r["title"] for r in (megabox_res or {}).get("rows", [])])

    def join(rec: dict | None, entry: dict) -> None:
        stats["program_rows"] += 1
        if rec is None:
            stats["program_unmatched"] += 1
            return
        stats["program_matched"] += 1
        found.setdefault(rec["id"], []).append(entry)

    if megabox_res:
        by_no = {n: f for f in films for n in f["stages"].get("megabox", {}).get("movie_nos", [])}
        for r in megabox_res["rows"]:
            ev = lineup_for(megabox_res["events"], r["release_date"])
            entry = _entry("megabox", MEGA_ONLY, ev["month"] if ev else "", r["release_date"], True,
                           MEGABOX_EVENT_URL + ev["event_no"] if ev else MEGABOX_TAB_URL)
            rec = by_no.get(r["movie_no"])
            if rec is None:
                cands = index.find(megabox.base_title(r["title"]))
                rec = max(cands, key=lambda f: f["id"]) if cands else None
            join(rec, entry)
    if lotte_rows:
        by_code = {f["stages"]["lotte"]["movie_code"]: f for f in films
                   if f["stages"].get("lotte", {}).get("movie_code")}
        for r in lotte_rows:
            if r.get("exclusive"):
                month = r["release_date"][:7] if r["release_date"] else r.get("release_month", "")
                name = LOTTE_PICK if r.get("badge") == "pick" else LOTTE_PROGRAM
                join(by_code.get(r["movie_code"]),
                     _entry("lotte", name, month, r["release_date"], True, LOTTE_URL))
    if cgv_rows_seen:
        for f in films:
            st = f["stages"].get("cgv", {})
            if "cgv_only" in st.get("labels", []) and st.get("status") != "ended":
                month = (st.get("release_date") or "")[:7] or st.get("release_month", "")
                join(f, _entry("cgv", CGV_PROGRAM, month, st.get("release_date", ""), True, CGV_URL))
    for f in films:
        entries = sorted(found.get(f["id"], []), key=lambda e: CHAIN_ORDER.index(e["chain"]))
        if entries:
            stage = dict(entries[0])
            if len(entries) > 1:
                stage["also"] = entries[1:]
            f["stages"]["program"] = stage
            f["first_seen"]["program"] = min(stamp, f["first_seen"].get("program", stamp))
    stats["with_program"] = sum("program" in f["stages"] for f in films)
    return stats
