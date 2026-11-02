"""KMRB ORS (온라인등급분류서비스) in-progress intake list, ors.kmrb.or.kr.

The page https://ors.kmrb.or.kr/rating/ongoing_all_list.do shows the list 5
rows per server-rendered page; the page loads the full list for its Excel
download from /rating/ongoing_mv_list_ajax.do (GET, the same "searchMv"
form fields, JSON {"data": [rows]}). On 2026-11-01 that answered each tab in
under 1 s with the same receipts as all its HTML pages (36 and 78 rows;
paging took 24 requests of about 7 s). Tabs are chosen with
mv_kind = MV_DOM (국내영화) | MV_FOR (국외영화) | ...; we read those two.

Of each row only the fields below are read, plus the applicant's
representative (REPR_NM), which is kept in memory for the manual block /
allow lists (lists.py) and never written; the other personal fields (staff,
cast) are not read.
RCV_NO is the same internal receipt id the KMRB list exposes in
selectPageInfo('<n>') - the join key between the two sources. (OLD_RCV_NO,
e.g. 2026-RMF03095, is the display-only 접수번호.)
"""
from __future__ import annotations

import json

from .http import PoliteClient, check_block
from .text import clean

URL = "https://ors.kmrb.or.kr/rating/ongoing_all_list.do"
JSON_URL = "https://ors.kmrb.or.kr/rating/ongoing_mv_list_ajax.do"
TABS = {"MV_DOM": "국내영화", "MV_FOR": "국외영화"}


class ParseError(ValueError):
    pass


def status_label(progress: str) -> str:
    """The list page's 진행상태 for a row's PROC_STATE_CODE_VALUE (its script:
    결과통보 -> 완료, 신청완료 stays, anything else -> 진행중)."""
    return {"결과통보": "완료", "신청완료": "신청완료"}.get(progress, "진행중")


def parse_rows(text: str, tab: str, reps: dict | None = None) -> list[dict]:
    """Rows of one tab's JSON answer. Raises on a block notice or bad payload.
    `reps`, if given, collects {applicant: {representative}}."""
    try:
        data = json.loads(text)["data"]
        if not isinstance(data, list):
            raise TypeError("data is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        check_block(text, markup_ok=False)
        raise ParseError(f"ORS {tab}: unexpected answer ({exc})") from exc
    rows = []
    for x in data:
        if x.get("KIND") != tab or not x.get("RCV_NO"):
            raise ParseError(f"ORS: asked for tab {tab}, got {x.get('KIND')!r}")
        title = clean(x.get("MV_USE_TITLE") or "")
        original = clean(x.get("MV_ORI_TITLE") or "")
        progress = clean(x.get("PROC_STATE_CODE_VALUE") or "")
        rep = clean(x.get("REPR_NM") or "")
        if reps is not None and rep and x.get("APLC_NAME"):
            reps.setdefault(clean(x["APLC_NAME"]), set()).add(rep)
        rows.append({
            "rcv_no": str(x["RCV_NO"]),
            "tab": tab,
            "kind": clean(x.get("MEDI_CODE_VALUE") or "") or TABS[tab],
            "title": title,
            "original_title": "" if original == title else original,
            "applicant": clean(x.get("APLC_NAME") or ""),
            "date": clean(x.get("RCV_DATE") or ""),
            "status": status_label(progress),
            "progress": progress,
            "receipt_no": clean(x.get("OLD_RCV_NO") or ""),
        })
    return rows


def _form(tab: str) -> dict:
    return {
        "loadingYN": "Y", "proc_state_code": "onGoing", "mv_kind": tab, "vd_kind": "",
        "cPage": "1", "cPage2": "", "mv_use_title": "", "aplc_name": "",
        "compl_date_start": "", "compl_date_end": "", "dire_name": "", "leada_name": "",
    }


def fetch_tab(client: PoliteClient, tab: str, reps: dict | None = None) -> list[dict]:
    resp = client.get(JSON_URL, params=_form(tab),
                      headers={"Referer": URL, "X-Requested-With": "XMLHttpRequest"})
    rows = parse_rows(resp.text, tab, reps)
    seen: set[str] = set()  # one row per receipt
    return [r for r in rows if not (r["rcv_no"] in seen or seen.add(r["rcv_no"]))]


def fetch_intake(client: PoliteClient) -> dict:
    """{"rows", "stats", "reps"}; reps ({applicant: [representatives]}) is for
    the manual lists only and must not be written."""
    rows, stats, reps = [], {}, {}
    for tab in TABS:
        got = fetch_tab(client, tab, reps)
        rows.extend(got)
        stats[tab] = {"rows": len(got)}
    return {"rows": rows, "stats": stats, "reps": {k: sorted(v) for k, v in reps.items()}}
