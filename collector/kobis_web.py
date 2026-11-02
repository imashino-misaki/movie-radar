"""KOBIS website: film list by status, release schedule, real-time booking, detail page.

Upcoming list: the KOBIS film search (searchMovieList.do, anonymous POST)
filtered by 제작상태 개봉예정 / 개봉준비 lists films a distributor registered
for release, also before an opening date is set (on 2026-11-01: 178 and 87
films, 10 per page). The Open API list is queried by opening year, so it
misses those. Only the code, title, production year, genres and status are
read; the caller decides what to publish (see kobis.publishable).

Release schedule: the KOBIS 개봉스케줄 calendar (findOpenScheduleList.do,
anonymous POST per month) lists films with a planned opening date, or only
the planned month while the day is not set, with a poster thumbnail. On
2026-11-01 it reached 2027-07 (Nov 52 films, Dec 20, then 0-2 a month with
empty months in between), so SCHEDULE_MONTHS months from the current one are
read. Unlike the 개봉예정 status list it holds no films registered years ago
and never updated.

Detail page: first opening date, re-release dates and the film's details.

The KOBIS Open API gives a film's first opening date only (openDt stays the
original date when the film returns to cinemas). The public detail page on
kobis.or.kr (searchMovieDtl.do, anonymous GET) also lists the re-release
dates under 재개봉일, e.g. for code 20138661 (checked 2026-11-01):
개봉일 2013-06-13, 재개봉일 2025-03-19, 2026-03-18, 2026-10-14. Films never
re-released have no 재개봉일 entry; an unknown code answers HTTP 420. The
same page gives 제작상태, grade, running time, 상영타입, the companies with
their KOBIS company codes, 최종수정, the registered poster and the synopsis
(see parse). Staff and cast are loaded by a separate request and not read.

Pages are fetched for the KOBIS code of every record (grouped per-format
records share their lead's) and cached in data/kobis_detail.json
(code -> parse() fields + "checked"):
  - a code not in the cache, or cached before the detail fields existed, is fetched;
  - a cached code is fetched again (at most once a day) while a chain lists
    the film for a release date the page does not show yet, since KOBIS adds
    a re-release date by hand, sometimes only days before it;
  - other pages are read again after REFRESH_DAYS, oldest first, at most
    ceil(codes / REFRESH_DAYS) a run.
"""
from __future__ import annotations

import datetime as dt
import math
import re

from .applicants import TRUST_WINDOW_MONTHS
from .http import BlockedError, FetchError, check_block
from .kmrb import months_back
from .text import clean

DETAIL_URL = "https://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieDtl.do"
LIST_URL = "https://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do"
# 제작상태 codes of the list form's status filter (KOBIS code list 2202000000,
# from findCommCode.do on 2026-11-01): 220220 개봉예정, 220230 개봉준비
UPCOMING_STATUS_CODES = {"220220": "개봉예정", "220230": "개봉준비"}
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_STATUS_LABEL = "제작상태"  # present on every film's page, absent on the error page


def _field(html: str, label: str) -> str | None:
    m = re.search(r"<dt>\s*" + label + r"\s*</dt>\s*<dd>(.*?)</dd>", html, re.S)
    return m.group(1) if m else None


def _text(fragment: str) -> str:
    return clean(re.sub(r"<[^>]+>", " ", fragment.replace("&nbsp;", " ")))


# a film not yet open labels its date 개봉(예정)일 (듄: 파트 3, checked 2026-11-01)
_OPEN_LABEL = r"개봉(?:\(예정\))?일"
_FORMAT = re.compile(r"<strong>\s*([^<:]+?)\s*:\s*</strong>\s*([^<(]+)")
_RUNTIME = re.compile(r"(\d+)분")
_MODIFIED = re.compile(r"최종수정\s*:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_MAIN_POSTER = re.compile(r'<a class="fl thumb" href="(/common/mast/movie/[^"]+)"')
_SYNOPSIS = re.compile(r'시놉시스</strong>\s*<p class="desc_info">(.*?)</p>', re.S)
_COMPANY_ROLE = re.compile(r"<dt[^>]*>\s*([^<]+?)\s*</dt>(.*?)(?=<dt|</dl>)", re.S)
_COMPANY = re.compile(r"mstView\('company','(\d+)'\);return false;\">([^<]+)</a>")
NO_INFO = "해당정보없음"


def parse(html: str) -> dict:
    """The fields read from one detail page. Raises on a bad page.

    open_dt, rereleases; status (제작상태); grade and runtime (minutes) from
    요약정보; genres (요약정보's third field, e.g. ["스릴러", "미스터리"]: it is
    the genre on all four recorded pages, also on one without runtime and
    grade); formats (상영타입, e.g. "2D 디지털"); companies [{role, name,
    code}] from 영화사 (배급사, 수입사, 제작사, ...); modified (최종수정, KST);
    poster (the registered main poster, full size); synopsis."""
    status = _field(html, _STATUS_LABEL)
    if status is None:
        check_block(html, markup_ok=False)
        raise FetchError("kobis detail: film fields missing")
    opened = _DATE.findall(_field(html, _OPEN_LABEL) or "")
    summary = [_text(x) for x in (_field(html, "요약정보") or "").split("|")]
    runtime = next((int(m.group(1)) for x in summary if (m := _RUNTIME.match(x))), 0)
    companies = []
    section = html.split(">영화사<", 1)[1].split("</dl>", 1)[0] + "</dl>" if ">영화사<" in html else ""
    for role, rest in _COMPANY_ROLE.findall(section):
        companies += [{"role": clean(role), "name": clean(name), "code": code}
                      for code, name in _COMPANY.findall(rest)]
    modified, poster, synopsis = (_MODIFIED.search(html), _MAIN_POSTER.search(html),
                                  _SYNOPSIS.search(html))
    return {"open_dt": opened[0] if opened else "",
            "rereleases": sorted(set(_DATE.findall(_field(html, "재개봉일") or ""))),
            "status": _text(status),
            "grade": next((x for x in summary if x.endswith(("관람가", "관람불가"))), ""),
            "runtime": runtime,
            "genres": [g for g in (clean(x) for x in summary[2].split(",")) if g] if len(summary) > 2 else [],
            "formats": [a if a == b else f"{a} {b}" for a, b in
                        ((clean(x), clean(y)) for x, y in _FORMAT.findall(_field(html, "상영타입") or ""))],
            "companies": companies,
            "modified": modified.group(1) if modified else "",
            "poster": KOBIS_ORIGIN + poster.group(1) if poster else "",
            "synopsis": _text(synopsis.group(1)) if synopsis else ""}


def listed_code(rec: dict) -> str:
    """The KOBIS code of a record: Lotte's code for the film, else the KOBIS stage's."""
    st = rec["stages"]
    return st.get("lotte", {}).get("kobis_cd") or st.get("kobis", {}).get("movie_cd", "")


def chain_dates(rec: dict) -> set[str]:
    """Release dates the chains currently list the film for."""
    return {s["release_date"] for name in ("megabox", "lotte", "cgv")
            if (s := rec["stages"].get(name)) and s.get("status") != "ended"
            and s.get("release_date")}


def explained(entry: dict, dates: set[str]) -> bool:
    known = {entry.get("open_dt", "")} | set(entry.get("rereleases", []))
    return dates <= known


# a page is read again after this many days; with ~180 codes (2026-11-01)
# that is ~26 requests a day, and at most ceil(codes / REFRESH_DAYS) per run
REFRESH_DAYS = 7
DETAIL_FIELDS = ("status", "grade", "runtime", "genres", "formats", "companies", "modified",
                 "poster", "synopsis")


def detail_code(rec: dict, rating_codes: dict | None = None) -> str:
    """The KOBIS code whose page describes the record (Lotte's, else any,
    else the code release_lags gives for the record's own rating number:
    `rating_codes` {rating_no: KOBIS code}, see run.py). Without that last
    one a rated film that opened before the KOBIS list first showed it, or a
    screening-format rating KOBIS does not list, has no code at all."""
    return (listed_code(rec) or min(record_codes(rec), default="")
            or (rating_codes or {}).get(rec["stages"].get("rating", {}).get("rating_no", ""), ""))


def _stale(entry: dict, stamp: str) -> bool:
    checked = entry.get("checked", "")
    return not checked or (dt.date.fromisoformat(stamp) - dt.date.fromisoformat(checked)).days >= REFRESH_DAYS


def fetch(client, films: list[dict], cache: dict, stamp: str,
          rating_codes: dict | None = None) -> dict:
    """Fetch detail pages per the rules above. Updates `cache` in place; codes
    no record refers to any more are dropped. One failing page is retried next
    run; a block stops the source."""
    codes: dict[str, set[str]] = {}
    for rec in films:
        if rec.get("variant_of"):
            continue
        code = detail_code(rec, rating_codes)
        if code:
            codes.setdefault(code, set()).update(chain_dates(rec))
    new = [cd for cd in sorted(codes) if cd not in cache or "modified" not in cache[cd]]
    unexplained = [cd for cd, dates in sorted(codes.items()) if cd not in new
                   and not explained(cache[cd], dates) and cache[cd].get("checked", "") < stamp]
    stale = sorted((cd for cd in codes if cd not in new and cd not in unexplained
                    and _stale(cache[cd], stamp)), key=lambda cd: cache[cd]["checked"])
    stale = stale[:math.ceil(len(codes) / REFRESH_DAYS)]
    fetched = errors = 0
    for cd in new + unexplained + stale:
        try:
            page = parse(client.get(DETAIL_URL, params={"code": cd}).text)
        except BlockedError:
            raise
        except Exception:  # noqa: BLE001 - one bad page must not fail the source
            errors += 1
            continue
        cache[cd] = {**page, "checked": stamp}
        fetched += 1
    for cd in [cd for cd in cache if cd not in codes]:
        del cache[cd]
    return {"codes": len(codes), "fetched": fetched, "new": len(new), "refreshed": len(stale),
            "errors": errors,
            "unexplained": sum(cd in cache and not explained(cache[cd], d) for cd, d in codes.items())}


def apply(films: list[dict], cache: dict, rating_codes: dict | None = None) -> dict:
    """Attach stages "release_history" {kobis_cd, open_dt, rereleases} and
    "kobis_detail" {kobis_cd, DETAIL_FIELDS...} (in place)."""
    n = 0
    for rec in films:
        code = detail_code(rec, rating_codes)
        entry = cache.get(code) if code else None
        if not entry:
            rec["stages"].pop("release_history", None)
            rec["stages"].pop("kobis_detail", None)
            continue
        rec["stages"]["release_history"] = {"kobis_cd": code, "open_dt": entry["open_dt"],
                                            "rereleases": list(entry["rereleases"])}
        if "modified" in entry:
            # pages cached before "genres" existed lack it until their next refresh
            rec["stages"]["kobis_detail"] = {"kobis_cd": code,
                                             **{k: entry[k] for k in DETAIL_FIELDS if k in entry}}
        else:
            rec["stages"].pop("kobis_detail", None)
        n += 1
    return {"with_history": n,
            "with_rereleases": sum(bool(f["stages"].get("release_history", {}).get("rereleases"))
                                   for f in films),
            "with_detail": sum("kobis_detail" in f["stages"] for f in films)}


_TOTAL = re.compile(r'class="total">\s*총\s*<em[^>]*>\s*([\d,]+)\s*</em>')
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_CODE = re.compile(r"mstView\('movie','([^']+)'")
_TITLE_ATTR = re.compile(r'title="([^"]*)"')


def parse_list_page(html: str) -> tuple[list[dict], int]:
    """Rows of one search result page and the declared total. Raises on a bad page."""
    m = _TOTAL.search(html)
    if not m:
        check_block(html, markup_ok=False)
        raise FetchError("kobis list: result count missing")
    body = html[html.find("<tbody"):html.find("</tbody>")]
    rows = []
    for tr in _ROW.findall(body):
        code = _CODE.search(tr)
        tds = _TD.findall(tr)
        if not code or len(tds) < 8:
            continue
        genre = _TITLE_ATTR.search(tds[6])
        rows.append({"movie_cd": code.group(1), "title": clean(tds[0]),
                     "prdt_year": clean(tds[3]),
                     "genres": clean(genre.group(1) if genre else tds[6]),
                     "status": clean(tds[7])})
    return rows, int(m.group(1).replace(",", ""))


def fetch_upcoming(client) -> dict:
    """Every film KOBIS lists as 개봉예정 or 개봉준비. Returns {"rows", "stats"};
    a list shorter than its declared total is a failure."""
    rows: list[dict] = []
    stats = {}
    for code, name in UPCOMING_STATUS_CODES.items():
        got: list[dict] = []
        page = 1
        while True:
            resp = client.post(LIST_URL, data={"sPrdtStatCd": code, "curPage": str(page),
                                               "useYn": "Y", "searchType": "search"})
            page_rows, total = parse_list_page(resp.text)
            got += page_rows
            if len(got) >= total:
                break
            if not page_rows:
                raise FetchError(f"kobis list {name}: page {page} empty, {len(got)}/{total} rows")
            page += 1
        rows += got
        stats[name] = len(got)
    seen: set[str] = set()
    rows = [r for r in rows if not (r["movie_cd"] in seen or seen.add(r["movie_cd"]))]
    return {"rows": rows, "stats": stats}


SCHEDULE_URL = "https://www.kobis.or.kr/kobis/business/mast/mvie/findOpenScheduleList.do"
# months read from the current one: the calendar reached 8 months ahead on
# 2026-11-01 (2027-07) with empty months in between, so a year is read
SCHEDULE_MONTHS = 12
KOBIS_ORIGIN = "https://www.kobis.or.kr"
_CSRF = re.compile(r'name="CSRFToken" value="([^"]+)"')
_ITEM = re.compile(r"<li class=\"item[^\"]*\">\s*<!--.*?-->\s*<a[^>]*mstView\('movie','([^']+)'\)[^>]*>"
                   r"\s*<img\s+src=\"([^\"]*)\"(.*?)</dl>", re.S)
_ITEM_TITLE = re.compile(r'class="tit[^"]*" title="([^"]*)"')
_ITEM_INFO = re.compile(r'<li title="([^"]*)"')
_ITEM_DATE = re.compile(r"<dd>(.*?)</dd>", re.S)
_MONTH = re.compile(r"\d{4}-\d{2}")


def parse_schedule(html: str) -> list[dict]:
    """Films of one month of the calendar. Raises on a page without the calendar."""
    if 'id="listArea"' not in html:
        # a month without films shows the calendar tabs and a no-result image
        if 'id="codeMovie"' in html and "search_n.png" in html:
            return []
        check_block(html, markup_ok=False)
        raise FetchError("kobis schedule: calendar missing")
    rows = []
    for code, img, body in _ITEM.findall(html):
        title = _ITEM_TITLE.search(body)
        when = clean((_ITEM_DATE.findall(body) or [""])[0])
        info = _ITEM_INFO.findall(body)
        day = _DATE.search(when)
        month = _MONTH.search(when)
        rows.append({"movie_cd": code, "title": clean(title.group(1)) if title else "",
                     "date": day.group(0) if day else "",
                     "month": (day or month).group(0)[:7] if (day or month) else "",
                     "genres": clean(info[0]) if info else "",
                     "prdt_year": clean(info[3]).removeprefix("제작").strip() if len(info) > 3 else "",
                     "poster": KOBIS_ORIGIN + img if img.startswith("/") else img})
    return rows


def schedule_months(today) -> list[str]:
    y, m = today.year, today.month
    out = []
    for _ in range(SCHEDULE_MONTHS):
        out.append(f"{y:04d}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_schedule(client, today) -> dict:
    """Every film on the calendar from the current month on. Returns {"rows", "stats"}."""
    page = client.get(SCHEDULE_URL).text
    token = _CSRF.search(page)
    if not token:
        check_block(page, markup_ok=False)
        raise FetchError("kobis schedule: form token missing")
    rows: list[dict] = []
    stats = {}
    for ym in schedule_months(today):
        resp = client.post(SCHEDULE_URL, data={
            "CSRFToken": token.group(1), "pageIndex": "1", "sDate": ym, "sNationCd": "00",
            "sMultiMovie": "01", "viewEa": "", "openDt": "", "selOpenDt": ""})
        got = parse_schedule(resp.text)
        stats[f"{ym[:4]}-{ym[4:]}"] = len(got)
        rows += got
    seen: set[str] = set()
    rows = [r for r in rows if not (r["movie_cd"] in seen or seen.add(r["movie_cd"]))]
    return {"rows": rows, "stats": stats}


def record_codes(rec: dict) -> set[str]:
    """Every KOBIS code a record carries."""
    st = rec["stages"]
    codes = {st.get("lotte", {}).get("kobis_cd", ""), st.get("kobis", {}).get("movie_cd", ""),
             st.get("release_history", {}).get("kobis_cd", ""),
             rec["id"][1:] if rec["id"].startswith("k") else ""}
    return codes - {""}


# stages that show a film is moving now: a filing or rating in the KMRB
# window, a release-schedule entry, tickets on sale, a chain listing, a
# trailer or poster filing
ACTIVE_STAGES = frozenset({"intake", "rating", "schedule", "booking", "megabox", "lotte", "cgv", "promo"})
# the same year the applicant trust rule looks back (applicants.py)
DORMANT_MONTHS = TRUST_WINDOW_MONTHS


def dormant(rec: dict, today: dt.date) -> bool:
    """A KOBIS 개봉예정 / 개봉준비 entry nobody has touched for a year: no
    active stage, and the KOBIS page's last edit, any opening or re-release
    date and any past rating all older than DORMANT_MONTHS (or missing). Some
    were registered a decade ago and never opened, or kept 개봉예정 next to an
    opening date that passed years ago."""
    st = rec["stages"]
    if "kobis" not in st or ACTIVE_STAGES & set(st):
        return False
    modified = st.get("kobis_detail", {}).get("modified", "")[:10]
    if not modified:
        return False
    hist = st.get("release_history", {})
    latest = max([modified, st["kobis"].get("open_dt", ""), hist.get("open_dt", ""),
                  *hist.get("rereleases", []), st.get("past_rating", {}).get("date", "")])
    return latest < months_back(today, DORMANT_MONTHS).isoformat()


def mark_dormant(films: list[dict], today: dt.date) -> int:
    """Tag dormant records "dormant" (in place); returns how many."""
    n = 0
    for rec in films:
        tags = set(rec.get("tags", [])) - {"dormant"}
        if dormant(rec, today):
            tags.add("dormant")
            n += 1
        rec["tags"] = sorted(tags)
    return n


def apply_schedule(films: list[dict], rows: list[dict]) -> dict:
    """Attach stage "schedule" {movie_cd, date, month, poster, genres when known} (in place) to the
    records of films on the calendar; records no longer on it lose the stage."""
    by_cd = {r["movie_cd"]: r for r in rows}
    n = 0
    for rec in films:
        hit = next((by_cd[c] for c in sorted(record_codes(rec)) if c in by_cd), None)
        if hit is None:
            rec["stages"].pop("schedule", None)
            continue
        rec["stages"]["schedule"] = {k: hit[k] for k in ("movie_cd", "date", "month", "poster")}
        genres = [g for g in (clean(x) for x in hit.get("genres", "").split(",")) if g]
        if genres:  # the calendar's genre text, e.g. "애니메이션"
            rec["stages"]["schedule"]["genres"] = genres
        n += 1
    return {"with_schedule": n, "schedule_unmatched": len(by_cd) - len(
        {c for rec in films for c in record_codes(rec)} & set(by_cd))}


# ---------------------------------------------------------------- real-time booking

BOOKING_URL = "https://www.kobis.or.kr/kobis/business/stat/boxs/findRealTicketList.do"
# the form's region checkboxes (16 on 2026-11-01), sent all at once like "전체"
_AREA = re.compile(r'name="wideareaCd" value="([^"]+)"')
_AS_OF = re.compile(r"조회일시\s*:\s*(\d{4})/(\d{2})/(\d{2})\s+(\d{2}:\d{2})")
_REREL = re.compile(r'<span class="layer_desc">\s*재개봉일\s*:\s*(\d{4}-\d{2}-\d{2})\s*</span>')
_TAGS = re.compile(r"<[^>]+>")


def _num(cell: str) -> int:
    return int(clean(_TAGS.sub("", cell)).replace(",", "") or 0)


def parse_booking(html: str) -> dict:
    """The 실시간 예매율 table: {"as_of": "YYYY-MM-DD HH:MM" (KST), "rows": [...]}.
    Each row: movie_cd, title, rank, open_dt (the first opening date, "" for
    undated screenings), rerelease_dt (KOBIS's 재개봉일 label, else ""), rate
    (% of all advance tickets), presale_sales, presale_audience, total_sales,
    total_audience (cumulative, KRW and people)."""
    total = _TOTAL.search(html)
    if not total or "<tbody" not in html:
        check_block(html, markup_ok=False)
        raise FetchError("kobis booking: result table missing")
    as_of = _AS_OF.search(html)
    body = re.sub(r"<!--.*?-->", "", html.split("<tbody", 1)[1].split("</tbody>", 1)[0], flags=re.S)
    rows = []
    for tr in _ROW.findall(body):
        tds = _TD.findall(tr)
        code = _CODE.search(tr)
        if len(tds) != 8 or not code:
            raise FetchError("kobis booking: unexpected row markup")
        title = _TITLE_ATTR.search(tds[1])
        rerel = _REREL.search(tds[2])
        opened = _DATE.search(_REREL.sub("", tds[2]))
        rows.append({
            "movie_cd": code.group(1), "title": clean(title.group(1)) if title else "",
            "rank": _num(tds[0]), "open_dt": opened.group(0) if opened else "",
            "rerelease_dt": rerel.group(1) if rerel else "",
            "rate": float(clean(_TAGS.sub("", tds[3])).rstrip("%") or 0),
            "presale_sales": _num(tds[4]), "total_sales": _num(tds[5]),
            "presale_audience": _num(tds[6]), "total_audience": _num(tds[7])})
    if len(rows) != int(total.group(1).replace(",", "")):
        raise FetchError(f"kobis booking: {len(rows)} rows, page says {total.group(1)}")
    stamp = f"{as_of.group(1)}-{as_of.group(2)}-{as_of.group(3)} {as_of.group(4)}" if as_of else ""
    return {"as_of": stamp, "rows": rows}


def fetch_booking(client) -> dict:
    """Every film with advance tickets on sale, all regions and film kinds.
    Returns {"as_of", "rows", "stats"}."""
    page = client.get(BOOKING_URL).text
    token, areas = _CSRF.search(page), _AREA.findall(page)
    if not token or not areas:
        check_block(page, markup_ok=False)
        raise FetchError("kobis booking: form token or regions missing")
    # all of loadEnd/allMovieYn/areaCd/wideareaAll are needed: without them the
    # page answers with an empty table (checked 2026-11-01)
    resp = client.post(BOOKING_URL, data={
        "CSRFToken": token.group(1), "loadEnd": "0", "dmlMode": "search", "repNationCd": "",
        "allMovieYn": "Y", "sMultiChk": "YYY", "sNomal": "Y", "sMulti": "Y", "sIndie": "Y",
        "areaCd": "", "wideareaCd": areas, "wideareaAll": "ALL"})
    got = parse_booking(resp.text)
    return {**got, "stats": {"rows": len(got["rows"]), "as_of": got["as_of"],
                             "undated": sum(not r["open_dt"] for r in got["rows"])}}


BOOKING_FIELDS = ("movie_cd", "rank", "rate", "presale_audience", "presale_sales",
                  "total_audience", "total_sales", "open_dt", "rerelease_dt")


def apply_booking(films: list[dict], booking: dict) -> dict:
    """Attach stage "booking" {BOOKING_FIELDS..., as_of} (in place) to the
    records of films with tickets on sale; the others lose the stage."""
    by_cd = {r["movie_cd"]: r for r in booking["rows"]}
    n = 0
    for rec in films:
        hit = next((by_cd[c] for c in sorted(record_codes(rec)) if c in by_cd), None)
        if hit is None:
            rec["stages"].pop("booking", None)
            continue
        rec["stages"]["booking"] = {**{k: hit[k] for k in BOOKING_FIELDS}, "as_of": booking["as_of"]}
        n += 1
    return {"with_booking": n, "booking_unmatched": len(by_cd) - len(
        {c for rec in films for c in record_codes(rec)} & set(by_cd))}
