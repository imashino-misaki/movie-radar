"""Lotte Cinema public movie list (now showing + upcoming).

Two anonymous POSTs to the endpoint that backs lottecinema.co.kr's movie list
pages (MethodName GetMoviesToBe): moviePlayYN Y returns the films now showing,
N the upcoming ones (checked 2026-11-01: 57 and 39 rows, each response's
ItemCount equal to its rows). Each list carries one placeholder row whose
code is "AD" (a banner slot, not a film); it is dropped.

Only list-level public fields are kept: title, release date (or only the
planned month while Lotte has no date), age grade, now/upcoming status, the
KOBIS movie code Lotte lists for the film, Lotte's programme badges and
whether tickets are on sale. No
synopsis, images or anything booking-related is stored.
"""
from __future__ import annotations

import json

from .http import FetchError, check_block
from .text import clean

LIST_URL = "https://www.lottecinema.co.kr/LCWS/Movie/MovieData.aspx"
REFERER = "https://www.lottecinema.co.kr/NLCHS/Movie/List?flag=1"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")
# one page holds the whole list: a full page means rows may be missing
BLOCK_SIZE = 1000

LISTS = {"now": "Y", "upcoming": "N"}
FILM_DIVISION = "M"  # MovieDivisionCode of a film row; the "AD" banner row has none

# ViewGradeCode -> the KMRB grade name used by the other sources (codes seen
# 2026-11-01: 0, 12, 15, 19, 100; 100 is shown as no grade yet)
GRADES = {0: "전체관람가", 12: "12세이상관람가", 15: "15세이상관람가", 18: "청소년관람불가",
          19: "청소년관람불가", 100: "등급미정"}
# TagDivCd badges on the list (badge images checked 2026-11-01): 2 = 롯시픽
# (Lotte's own programme pick, incl. its re-releases), 1 = 롯시플 (concerts,
# live viewings). 롯시픽 is Lotte's exclusive line-up: its November 2026 press
# release ("11월 단독 개봉 콘텐츠") named 7 films, and the 6 already listed all
# carried badge 2 (checked 2026-11-02). 롯시플 is not exclusive (the same live
# viewings play at Megabox).
BADGES = {"2": "pick", "1": "plus"}


def body(play_yn: str) -> dict:
    return {"MethodName": "GetMoviesToBe", "channelType": "HO", "osType": "W",
            "osVersion": USER_AGENT, "multiLanguageID": "KR", "division": 1,
            "moviePlayYN": play_yn, "orderType": "1" if play_yn == "Y" else "5",
            "blockSize": BLOCK_SIZE, "pageNo": 1, "memberOnNo": ""}


def parse_row(m: dict, status: str) -> dict:
    planned = str(m.get("PlanedRelsYN") or "") == "1"
    code = m.get("ViewGradeCode")
    code = code if isinstance(code, int) else -1
    return {
        "movie_code": str(m.get("RepresentationMovieCode") or ""),
        "title": clean(m.get("MovieNameKR") or ""),
        # ReleaseDate is "YYYY-MM-DD 오전 12:00:00"; without a planned date it is
        # a month-end placeholder (2027-01-31 for 2027-01), so only the month is kept
        "release_date": (m.get("ReleaseDate") or "")[:10] if planned else "",
        "release_month": "" if planned else (m.get("PlanedRelsMnth") or ""),
        "grade": GRADES.get(code, clean(str(m.get("ViewGradeNameKR") or ""))),
        "status": status,
        # tickets on sale (BookingYN): checked 2026-11-03, Y on all 46 now-showing rows and on
        # 14 of 38 upcoming ones (presale)
        "bookable": str(m.get("BookingYN") or "") == "Y",
        "kobis_cd": str(m.get("KOFMovieCd") or "").strip(),
        "badge": BADGES.get(str(m.get("TagDivCd") or ""), ""),
        # name of the film festival a screening belongs to (empty on all 39 rows of the
        # 2026-11-02 list; the field exists for festival line-ups)
        "festival": clean(m.get("MovieFestivalName") or ""),
        # Lotte's genre, e.g. "애니메이션", "드라마" (MovieGenreNameKR / US are null);
        # decides the site's 애니메이션 category (site.is_anime)
        "genre": clean(m.get("MovieGenreName") or ""),
        # 단독 개봉 (Lotte-only release); every row was N on 2026-11-01 (programs.py)
        "exclusive": str(m.get("SoloOpenYN") or "") == "Y" or str(m.get("TagDivCd") or "") == "2",
        "poster": str(m.get("PosterURL") or "").strip(),
    }


def parse_page(text: str, status: str) -> list[dict]:
    """Film rows of one list. Raises BlockedError / FetchError on a bad page."""
    try:
        data = json.loads(text)
    except ValueError:
        check_block(text, markup_ok=False)
        raise FetchError("lotte list: response is not JSON") from None
    movies = data.get("Movies") if isinstance(data, dict) else None
    if str(data.get("IsOK") if isinstance(data, dict) else "").lower() != "true" \
            or not isinstance(movies, dict) or not isinstance(movies.get("Items"), list):
        raise FetchError(f"lotte list {status}: IsOK/Movies missing")
    items = movies["Items"]
    if len(items) >= BLOCK_SIZE or len(items) < int(movies.get("ItemCount") or 0):
        raise FetchError(f"lotte list {status}: {len(items)} rows, ItemCount {movies.get('ItemCount')}")
    return [parse_row(m, status) for m in items
            if m.get("MovieDivisionCode") == FILM_DIVISION and m.get("RepresentationMovieCode")]


def fetch_list(client) -> dict:
    """Fetch both lists. Returns {"rows": [...], "stats": {...}}."""
    rows: list[dict] = []
    for status, play_yn in LISTS.items():
        resp = client.post(LIST_URL, files={"paramList": (None, json.dumps(body(play_yn),
                                                                           ensure_ascii=False))},
                           headers={"Referer": REFERER, "Origin": "https://www.lottecinema.co.kr"})
        got = parse_page(resp.text, status)
        if not got:
            # an empty list would mark its films as no longer listed: treat as a failure
            raise FetchError(f"lotte list {status}: empty")
        rows += got
    seen: set[str] = set()
    rows = [r for r in rows if not (r["movie_code"] in seen or seen.add(r["movie_code"]))]
    return {"rows": rows, "stats": {
        "rows": len(rows),
        "now": sum(r["status"] == "now" for r in rows),
        "upcoming": sum(r["status"] == "upcoming" for r in rows),
        "with_kobis_cd": sum(bool(r["kobis_cd"]) for r in rows),
    }}


def lotte_id(movie_code: str) -> str:
    return "l" + movie_code
