"""Megabox public movie list (now showing + upcoming).

One anonymous POST to the list endpoint that backs megabox.co.kr/movie. With
pageType=rfilmDe and onairYn=N the endpoint returns every listed film, both
상영중 (now showing) and 상영예정 (upcoming), in one response (checked
2026-10-31: 89 + 32 rows, totCnt 121). pageType=ticketing returns only the
bookable subset; each row of the full list carries the same answer in
bokdAbleYn (see parse_row), so one request is enough.

Only list-level public fields are kept: title, release date, age grade,
now/upcoming status, whether tickets are on sale, and the Megabox-exclusive flag. No synopsis, images or
anything booking-related is stored.
"""
from __future__ import annotations

import hashlib
import html
import json
import re

from .http import FetchError, check_block
from .text import clean, norm_key

LIST_URL = "https://www.megabox.co.kr/on/oh/oha/Movie/selectMovieList.do"
REFERER = "https://www.megabox.co.kr/movie"
PAGE_SIZE = 300   # one page covers the whole list today (totCnt 121)
MAX_PAGES = 5     # stop paging after this many requests even if totCnt says more

STATUS = {"상영중": "now", "상영예정": "upcoming"}

# version prefixes Megabox puts in front of a title, e.g. "(더빙) 나루토", "(자막) …"
_VERSION_PREFIX = re.compile(r"^\s*[\(\[](더빙|자막|우리말|한글자막|원어|무삭제)[^\)\]]*[\)\]]\s*")


def base_title(title: str) -> str:
    """Title without Megabox version prefixes such as (더빙)/(자막)."""
    prev = None
    while prev != title:
        prev, title = title, _VERSION_PREFIX.sub("", title)
    return title.strip()


def title_key(title: str) -> str:
    return norm_key(base_title(title))


def iso(yyyymmdd: str) -> str:
    s = (yyyymmdd or "").strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else ""


def unescape_all(text: str) -> str:
    """Some titles arrive entity-escaped twice (e.g. "&amp;#40;"): unescape until stable."""
    for _ in range(3):
        nxt = html.unescape(text)
        if nxt == text:
            break
        text = nxt
    return text


IMG_ORIGIN = "https://img.megabox.co.kr"


def parse_row(m: dict) -> dict:
    stat = (m.get("movieStatNm") or "").strip()
    return {
        "movie_no": str(m.get("movieNo") or ""),
        "title": unescape_all(clean(m.get("movieNm") or "")),
        "release_date": iso(m.get("rfilmDeReal") or ""),
        "grade": (m.get("admisClassNm") or "").strip(),
        "status": STATUS.get(stat, stat),
        # tickets on sale: 상영중 does not mean bookable. Checked 2026-11-03: of 82 상영중 rows
        # 4 had bokdAbleYn N (two 2025 films re-listed with no showtimes, one with its opening
        # 8 days ahead); 1 of 30 상영예정 rows had Y (presale). The 79 rows of pageType=ticketing
        # (the booking page's film list) were the Y rows; schdlYn agreed on every row
        "bookable": m.get("bokdAbleYn") == "Y",
        # Megabox-exclusive screening: singlPlayAt is the flag behind the main
        # page's MEGA ONLY badge (checked 2026-11-01: 18 of 122 rows Y, the
        # badged film among them); megaOnlyAt was N on every row
        "exclusive": m.get("singlPlayAt") == "Y" or m.get("megaOnlyAt") == "Y",
        # 클래식 소사이어티: staged-performance broadcasts and lecture screenings (The Met,
        # NT Live, RBO, 시네도슨트: 8 of 114 rows Y, checked 2026-11-02). festivalYn was N
        # on every row, including festival screenings, so it is not used
        "classic": m.get("classicAt") == "Y",
        # poster image path on Megabox's image host (checked 2026-11-01)
        "poster": IMG_ORIGIN + m["imgPathNm"] if str(m.get("imgPathNm") or "").startswith("/") else "",
    }


def parse_page(text: str) -> tuple[list[dict], int]:
    """Return (rows, totCnt). Raises BlockedError / FetchError on a bad page."""
    try:
        data = json.loads(text)
    except ValueError:
        check_block(text, markup_ok=False)
        raise FetchError("megabox list: response is not JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("movieList"), list):
        raise FetchError("megabox list: movieList missing")
    return [parse_row(m) for m in data["movieList"]], int(data.get("totCnt") or 0)


def body(page: int) -> dict:
    return {"currentPage": str(page), "recordCountPerPage": str(PAGE_SIZE),
            "pageType": "rfilmDe", "ibxMovieNmSearch": "", "onairYn": "N",
            "specialType": "", "specialYn": "N"}


def fetch_list(client) -> dict:
    """Fetch the full list. Returns {"rows": [...], "stats": {...}}."""
    rows: list[dict] = []
    total = 0
    for page in range(1, MAX_PAGES + 1):
        resp = client.post(LIST_URL, json=body(page), headers={"Referer": REFERER})
        got, total = parse_page(resp.text)
        rows += got
        if not got or len(rows) >= total:
            break
    if not rows:
        # an empty list would mark every film as no longer listed: treat as a failure
        raise FetchError("megabox list: empty")
    if len(rows) < total:
        # a short list would mark the missing films as ended
        raise FetchError(f"megabox list: {len(rows)}/{total} rows")
    seen: set[str] = set()
    rows = [r for r in rows if r["movie_no"] and not (r["movie_no"] in seen or seen.add(r["movie_no"]))]
    return {"rows": rows, "stats": {
        "rows": len(rows), "total_reported": total,
        "now": sum(r["status"] == "now" for r in rows),
        "upcoming": sum(r["status"] == "upcoming" for r in rows),
    }}


def megabox_id(key: str) -> str:
    return "m" + hashlib.sha1(key.encode()).hexdigest()[:12]
