"""KMRB (영상물등급위원회) rating-decision list, www.kmrb.or.kr.

Recipe: GET the list page once (session cookie), then POST one request per page
with the search form. The list covers the last 3 months. It is fetched twice:
adult_yn='N' (general) and adult_yn='Y' (a superset that also includes adult
titles). A row whose receipt id appears only in the 'Y' set is adult.

Each <tbody> <tr> has 8 <td>: type, title cell (Korean title <br> [original
title], link selectPageInfo('<rcv_no>')), grade icon (alt text), applicant,
runtime, rating date, rating number, country.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re

from .http import PoliteClient, check_block
from .text import clean

LIST_URL = "https://www.kmrb.or.kr/main/gd/grade/selectGradeList.do"
POST_URL = LIST_URL + "?category=MV&mi=1090"
# rows per page (the form's pageIndex). The page itself sends 20; on 2026-11-01
# a 100-row page answered in 0.86 s against 0.77 s for 20 rows, so 100 cuts the
# requests about fivefold for nearly the same time per request
PAGE_SIZE = 100
MEDIA_CODES = "17501,17502"  # 국내영화, 국외영화


class ParseError(ValueError):
    pass


_TOTAL_PAGES = re.compile(r'name\s*=\s*"totalPages"\s+value="(\d+)"')
_TOTAL_COUNT = re.compile(r"전체\(([\d,]+)건\)")
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_RCV = re.compile(r"selectPageInfo\('(\d+)'")
_ALT = re.compile(r'alt="([^"]*)"')
_BR = re.compile(r"<br\s*/?>")


def parse_page(page_html: str) -> tuple[list[dict], int, int]:
    """Return (rows, total_pages, total_count). Raises ParseError on unexpected markup."""
    m_pages = _TOTAL_PAGES.search(page_html)
    m_total = _TOTAL_COUNT.search(page_html)
    markup_ok = bool(m_pages and m_total and "<tbody" in page_html)
    check_block(page_html, markup_ok)
    if not markup_ok:
        raise ParseError("KMRB page: missing totalPages / 전체(N건) / tbody")
    body = page_html.split("<tbody", 1)[1].split("</tbody>", 1)[0]
    rows = []
    for tr in _ROW.findall(body):
        tds = _CELL.findall(tr)
        if len(tds) < 8:
            continue
        title_cell = tds[1]
        m_rcv = _RCV.search(title_cell)
        parts = _BR.split(title_cell, maxsplit=1)
        ko_part, orig_part = (parts[0], parts[1]) if len(parts) == 2 else (title_cell, "")
        original = clean(orig_part)
        if original.startswith("[") and original.endswith("]"):
            original = original[1:-1].strip()
        m_alt = _ALT.search(tds[2])
        rows.append({
            "rcv_no": m_rcv.group(1) if m_rcv else "",
            "kind": clean(tds[0]),
            "title": clean(ko_part),
            "original_title": original,
            "grade": m_alt.group(1).strip() if m_alt else clean(tds[2]),
            "applicant": clean(tds[3]),
            "runtime": clean(tds[4]),
            "date": clean(tds[5]),
            "rating_no": clean(tds[6]),
            "country": clean(tds[7]),
        })
    return rows, int(m_pages.group(1)), int(m_total.group(1).replace(",", ""))


def months_back(day: dt.date, months: int) -> dt.date:
    y, m = divmod(day.month - 1 - months, 12)
    year, month = day.year + y, m + 1
    return day.replace(year=year, month=month,
                       day=min(day.day, calendar.monthrange(year, month)[1]))


# radio values of the site's period selector
SEARCH_DATE = {3: "3months", 12: "1year"}


def fetch_list(client: PoliteClient, adult_yn: str, today_kst: dt.date, months: int = 3) -> dict:
    """Fetch every page for one adult_yn value. Raises on any failure."""
    form = {
        "pageIndex": PAGE_SIZE, "mediCode": MEDIA_CODES, "advancedOpen": "N",
        "searchDate": SEARCH_DATE[months],
        "searchDateStart": months_back(today_kst, months).isoformat(),
        "searchDateEnd": today_kst.isoformat(),
        "adult_yn": adult_yn, "useTitle": "",
    }
    rows: list[dict] = []
    page, total_pages, declared = 1, None, None
    while True:
        resp = client.post(POST_URL, data={**form, "currPage": page},
                           headers={"Referer": LIST_URL})
        page_rows, pages, total = parse_page(resp.text)
        if total_pages is None:
            total_pages, declared = pages, total
        if not page_rows and declared:
            raise ParseError(f"KMRB adult_yn={adult_yn}: page {page}/{total_pages} had no rows")
        rows.extend(page_rows)
        if page >= total_pages:
            break
        page += 1
    # the film list's 전체(N건) matches its rows (checked 2026-11-01: 322/322, 830/830)
    if declared and len(rows) < declared:
        raise ParseError(f"KMRB adult_yn={adult_yn}: {len(rows)}/{declared} rows")
    return {"rows": rows, "pages": total_pages, "declared_total": declared}


def fetch_ratings(client: PoliteClient, today_kst: dt.date, months: int = 3) -> dict:
    """Fetch the N and Y lists; return general rows, adult rows and stats."""
    client.get(LIST_URL)  # session cookie
    general = fetch_list(client, "N", today_kst, months)
    everything = fetch_list(client, "Y", today_kst, months)
    result = split_adult(general["rows"], everything["rows"])
    result["stats"] = {
        "general_pages": general["pages"], "general_declared": general["declared_total"],
        "general_rows": len(general["rows"]),
        "all_pages": everything["pages"], "all_declared": everything["declared_total"],
        "all_rows": len(everything["rows"]),
    }
    return result


def _row_key(row: dict) -> str:
    return row["rcv_no"] or f'{row["rating_no"]}|{row["title"]}'


def split_adult(general_rows: list[dict], all_rows: list[dict]) -> dict:
    """Rows present only in the adult_yn='Y' superset are adult."""
    general, general_keys = [], set()
    for r in general_rows:
        k = _row_key(r)
        if k not in general_keys:
            general_keys.add(k)
            general.append(r)
    adult, adult_keys = [], set()
    for r in all_rows:
        k = _row_key(r)
        if k not in general_keys and k not in adult_keys:
            adult_keys.add(k)
            adult.append(r)
    return {"general": general, "adult": adult}


# ---------------------------------------------------------------- representatives

EXCEL_URL = "https://www.kmrb.or.kr/main/gd/grade/selectGradeMVExcelDown.do"
EXCEL_PAGE = 20  # the page's own pageIndex; the export needs its totals to match


def parse_reps(xlsx: bytes) -> dict[str, list[str]]:
    """{applicant: [representatives]} from the list's Excel export (columns
    신청사, 신청사대표; checked 2026-11-01: 830 rows, 160 applicants, one
    representative each)."""
    import io
    import warnings

    from openpyxl import load_workbook

    if xlsx[:2] != b"PK":
        text = xlsx[:2000].decode("utf-8", "replace")
        check_block(text, markup_ok=False)
        raise ParseError("KMRB export: not a workbook")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "Workbook contains no default style"
        rows = load_workbook(io.BytesIO(xlsx), read_only=True).active.iter_rows(values_only=True)
        header = [clean(str(c or "")) for c in next(rows, ())]
        if "신청사" not in header or "신청사대표" not in header:
            raise ParseError(f"KMRB export: unexpected columns {header[:10]}")
        a, r = header.index("신청사"), header.index("신청사대표")
        out: dict[str, set[str]] = {}
        for row in rows:
            applicant, rep = clean(str(row[a] or "")), clean(str(row[r] or ""))
            if applicant and rep:
                out.setdefault(applicant, set()).add(rep)
    return {k: sorted(v) for k, v in out.items()}


def fetch_reps(client: PoliteClient, today_kst: dt.date, total: int, months: int = 3) -> dict:
    """Representatives of every applicant in the adult_yn='Y' list (a superset
    of the general one); `total` is that list's row count."""
    form = {
        "currPage": "1", "pageIndex": str(EXCEL_PAGE), "totalCoComma": str(total),
        "totalPages": str(-(-total // EXCEL_PAGE)), "mediCode": MEDIA_CODES,
        "advancedOpen": "N", "fileType": "xls", "useTitle": "", "searchDate": SEARCH_DATE[months],
        "searchDateStart": months_back(today_kst, months).isoformat(),
        "searchDateEnd": today_kst.isoformat(), "aplcName": "", "direName": "", "rcvNo": "",
        "leadaName": "", "rtNo": "", "oauthName": "", "prodcNatnlName": "", "coreHarmRsn": "",
        "mvAssoName": "", "adult_yn": "Y",
    }
    reps = parse_reps(client.post(EXCEL_URL, data=form, headers={"Referer": POST_URL}).content)
    return {"reps": reps, "stats": {"applicants": len(reps)}}
