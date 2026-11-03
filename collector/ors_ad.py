"""KMRB ORS in-progress filings of trailers and advertising material.

The ORS intake page (ors.py) has three more film tabs, read from the same
JSON endpoint with mv_kind = MVTRAIL (예고편), MVAD (광고영화) and MVADVTM
(광고선전물: posters, online and other material, 청소년 유해성 확인). A film's
first poster or trailer filing is often its first public trace after the film
rating, and comes days before the KMRB list shows the decision (promo.py):
천년여우's second poster was received 2026-10-30 and still in progress on
2026-11-01.

Measured 2026-11-01: one request per tab, 0.7-1.0 s each; 35 trailer, 5
advertising-film and 48 advertising-material rows, all received the same
day or not yet received (신청완료, no receipt date): a filing leaves the list
when it is decided, about a day after receipt (천년여우's first poster:
received 2026-10-15, confirmed 2026-10-16). Runs every two hours see nearly
every filing; decided filings are covered by the KMRB list (promo.py). The
ORS list of decided filings is not read: it is server-rendered 10 rows a
page (405 advertising-material rows decided 2026-10-25..11-01 = 41 pages a
week) and its Excel export answered HTTP 502.

Filings of the trailer and advertising-material tabs create records (id "a"
+ hash of title and applicant, see ad_id) when no record of the title takes
them, also rows not yet received (신청완료: no receipt date; first_seen.ors_ad
is the run time that first saw a filing, the receipt date is kept apart in
`dates`). Rows of the 광고영화 tab (MVAD) never create one: measured
2026-11-02, its five rows were cinema / TV commercials (academies, shops),
not films. Nor do rows of masked individuals or of applicants with adult
KMRB ratings in the trust window (applicants.py; `withheld`): nothing adult
is written. A record created from a filing is folded into the film's record
once one turns up (settle: a KMRB rating, a KOBIS entry, a promo or chain
record of the same title); until a stronger signal joins it, theatrical.py
routes it by its applicants' opening history (`ad_only`). Of each row only
the fields below are read; the representative, staff and cast fields are not.
Applicants that look like a person's name are masked (privacy.py).
"""
from __future__ import annotations

import hashlib

from . import ors
from .companies import company_key
from .http import PoliteClient
from .privacy import MASK, mask_applicant
from .text import clean, norm_key

TABS = {"MVTRAIL": "예고편", "MVAD": "광고영화", "MVADVTM": "광고선전물"}
# MV_ADVTM_KIND_CODE_VALUE of an advertising-material row (seen 2026-11-01:
# 포스터 23, 기타 22, 온라인 3) -> kind; trailers and advertising films by tab
MATERIAL_KINDS = {"포스터": "poster", "온라인": "online", "기타": "other"}
TAB_KINDS = {"MVTRAIL": "trailer", "MVAD": "ad_film"}
# tabs whose rows may create a record (MVAD: commercials, see above)
CREATE_TABS = ("MVTRAIL", "MVADVTM")
AD_PREFIX = "a"
# stages a record created from a filing may hold and still be known only
# from marketing material (a past rating is a lookup by title, not evidence
# of this release)
AD_ONLY_STAGES = frozenset({"ors_ad", "promo", "past_rating"})


def kind_of(tab: str, material: str) -> str:
    return TAB_KINDS.get(tab) or MATERIAL_KINDS.get(material, "other")


def parse_rows(text: str, tab: str) -> list[dict]:
    """Rows of one tab's JSON answer (ors.parse_rows' checks, other fields)."""
    import json

    try:
        data = json.loads(text)["data"]
        if not isinstance(data, list):
            raise TypeError("data is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        ors.check_block(text, markup_ok=False)
        raise ors.ParseError(f"ORS {tab}: unexpected answer ({exc})") from exc
    rows = []
    for x in data:
        if x.get("KIND") != tab or not x.get("RCV_NO"):
            raise ors.ParseError(f"ORS: asked for tab {tab}, got {x.get('KIND')!r}")
        title = clean(x.get("MV_USE_TITLE") or "")
        original = clean(x.get("MV_ORI_TITLE") or "")
        material = clean(x.get("MV_ADVTM_KIND_CODE_VALUE") or "")
        progress = clean(x.get("PROC_STATE_CODE_VALUE") or "")
        rows.append({
            "rcv_no": str(x["RCV_NO"]),
            "tab": tab,
            "kind": kind_of(tab, material),
            "media": clean(x.get("MEDI_CODE_VALUE") or "") or TABS[tab],
            "title": title,
            "original_title": "" if original == title else original,
            "applicant": mask_applicant(clean(x.get("APLC_NAME") or "")),
            "date": clean(x.get("RCV_DATE") or ""),          # receipt date ("" before receipt)
            "check_date": clean(x.get("COMPL_DATE2") or ""),  # decision date (empty while in progress)
            "status": ors.status_label(progress),
            "progress": progress,
            "receipt_no": clean(x.get("OLD_RCV_NO") or ""),
        })
    return rows


def fetch(client: PoliteClient) -> dict:
    """{"rows", "stats"}: every in-progress filing of the three tabs."""
    rows, stats, seen = [], {}, set()
    for tab in TABS:
        resp = client.get(ors.JSON_URL, params=ors._form(tab),
                          headers={"Referer": ors.URL, "X-Requested-With": "XMLHttpRequest"})
        got = [r for r in parse_rows(resp.text, tab) if not (r["rcv_no"] in seen or seen.add(r["rcv_no"]))]
        rows.extend(got)
        stats[tab] = {"rows": len(got), "received": sum(bool(r["date"]) for r in got)}
    return {"rows": rows, "stats": stats}


def ad_id(title: str, applicant: str) -> str:
    """Stable id of a record created from a filing: title key + applicant."""
    from .promo import title_key
    return AD_PREFIX + hashlib.sha1(f"{title_key(title)}|{applicant}".encode()).hexdigest()[:12]


def ad_only(rec: dict) -> bool:
    """A record created from a filing that no stronger signal has joined."""
    return rec["id"].startswith(AD_PREFIX) and set(rec.get("stages", {})) <= AD_ONLY_STAGES


def merge_stage(a: dict | None, b: dict | None) -> dict:
    """Union of two ors_ad stages (two records folded together)."""
    if not a or not b:
        return dict(a or b or {})
    return _stage(set(a.get("rcv_nos", [])) | set(b.get("rcv_nos", [])),
                  set(a.get("dates", [])) | set(b.get("dates", [])),
                  set(a.get("kinds", [])) | set(b.get("kinds", [])),
                  set(a.get("applicants", [])) | set(b.get("applicants", [])),
                  {**b.get("open", {}), **a.get("open", {})})


def _stage(rcv_nos: set, dates: set, kinds: set, applicants=frozenset(),
           open_now: dict | None = None) -> dict:
    dates = sorted(d for d in dates if d)
    return {"first_date": dates[0] if dates else "", "last_date": dates[-1] if dates else "",
            "kinds": sorted(kinds), "count": len(rcv_nos), "dates": dates,
            "rcv_nos": sorted(rcv_nos), "applicants": sorted(a for a in applicants if a),
            "open": dict(sorted((open_now or {}).items()))}


def _activity(rec: dict) -> str:
    """Latest dated activity of a record (promo._latest_activity, plus filing
    dates, Lotte / CGV dates and, for a record created from filings, the day
    it was first seen: a filing not yet received has no date)."""
    from .promo import _latest_activity
    st = rec["stages"]
    days = [_latest_activity(rec), st.get("ors_ad", {}).get("last_date", ""),
            st.get("lotte", {}).get("release_date", ""), st.get("cgv", {}).get("release_date", "")]
    if rec["id"].startswith(AD_PREFIX):
        days.append(rec.get("first_seen", {}).get("ors_ad", "")[:10])
    return max(days)


def pick(cands: list[dict], applicant: str, day: str) -> dict | None:
    """The record a filing (or a record created from filings) joins:
    promo.pick_record's rule (same applicant, else activity within
    promo.PRIOR_DAYS of `day`), applicants compared without legal form."""
    from .promo import _cutoff
    key = company_key(applicant) if applicant and applicant != MASK else ""
    same = [f for f in cands if f.get("applicant") == applicant
            or (key and company_key(f.get("applicant", "")) == key)]
    pool = same or [f for f in cands if _activity(f) >= _cutoff(day)]
    if not pool:
        return None
    # the most recently seen record; first_seen.ors_ad is left out because this
    # step sets it, so the rows of one run would follow whichever came first
    return max(pool, key=lambda f: (max((v[:10] for k, v in f["first_seen"].items() if k != "ors_ad"),
                                        default=""), f["id"]))


def can_create(row: dict, withheld) -> str:
    """"" if the row may create a record, else why not."""
    if row["tab"] not in CREATE_TABS:
        return "tab"
    if not (row["title"] or row["original_title"]):
        return "no_title"
    if not row["applicant"] or row["applicant"] == MASK:
        return "masked"
    if company_key(row["applicant"]) in withheld:
        return "adult_applicant"
    return ""


def _title_index(films: list[dict], wanted: list[str]):
    from .text import TitleIndex
    by_original: dict[str, list[dict]] = {}
    for f in films:
        if norm_key(f.get("original_title", "")):
            by_original.setdefault(norm_key(f["original_title"]), []).append(f)

    index = TitleIndex(films, wanted=wanted)

    def find(title: str, original: str) -> list[dict]:
        cands = index.find(title) if title else []
        if not cands and original:
            cands = by_original.get(norm_key(original), [])
        return cands
    return find


def apply(films: list[dict], rows: list[dict], stamp: str,
          withheld=frozenset()) -> tuple[list[dict], dict]:
    """Attach stage "ors_ad" (pure, no I/O). A row joins a record of the same
    title (or original title) under `pick`; otherwise it creates a record
    unless can_create says no (counted in `ors_ad_not_created` by reason).
    `stamp`: the run time with the KST offset (first_seen.ors_ad; its date part
    is the day); `withheld`: company keys of
    applicants with adult KMRB ratings. Filings accumulate across runs by
    receipt id; `open` is rewritten every run with the filings in today's
    lists ({receipt id: status}), so a decided filing drops out of it."""
    from .merge import _sort

    stats = {"ors_ad_rows": len(rows), "ors_ad_matched_rows": 0, "ors_ad_records": 0,
             "ors_ad_unmatched_titles": 0, "ors_ad_created": 0, "ors_ad_not_created": {}}
    by_id = {f["id"]: f for f in films}
    find = _title_index(films, [r["title"] for r in rows])
    created: list[dict] = []
    touched: dict[str, dict] = {}
    open_now: dict[str, dict] = {}
    unmatched: set[str] = set()
    day = stamp[:10]
    for r in sorted(rows, key=lambda r: r["rcv_no"]):
        keys = {norm_key(r["title"]), norm_key(r["original_title"])} - {""}
        cands = [f for f in find(r["title"], r["original_title"]) if f["id"] in by_id]
        cands += [f for f in created if f not in cands
                  and keys & {norm_key(f["title"]), norm_key(f.get("original_title", ""))}]
        # a masked applicant joins records masked the same way, else by recent activity
        rec = pick(cands, r["applicant"], r["date"] or day) if cands else None
        if rec is None:
            why = can_create(r, withheld)
            if why:
                stats["ors_ad_not_created"][why] = stats["ors_ad_not_created"].get(why, 0) + 1
                unmatched.add(norm_key(r["title"]) or norm_key(r["original_title"]))
                continue
            title = r["title"] or r["original_title"]
            aid = ad_id(title, r["applicant"])
            rec = by_id.get(aid)
            if rec is None:
                rec = {"id": aid, "rcv_no": "", "title": title, "original_title": r["original_title"],
                       "applicant": r["applicant"], "stages": {}, "first_seen": {},
                       "last_seen": day, "tags": []}
                by_id[aid] = rec
                created.append(rec)
                stats["ors_ad_created"] += 1
        else:
            stats["ors_ad_matched_rows"] += 1
        prev = rec["stages"].get("ors_ad") or {}
        open_now.setdefault(rec["id"], {})[r["rcv_no"]] = r["status"]
        rec["stages"]["ors_ad"] = _stage(set(prev.get("rcv_nos", [])) | {r["rcv_no"]},
                                         set(prev.get("dates", [])) | {r["date"]},
                                         set(prev.get("kinds", [])) | {r["kind"]},
                                         set(prev.get("applicants", [])) | {r["applicant"]})
        rec["first_seen"]["ors_ad"] = min(stamp, rec["first_seen"].get("ors_ad", stamp))
        rec["last_seen"] = max(rec.get("last_seen", ""), day)
        touched[rec["id"]] = rec
    for rec in by_id.values():
        st = rec["stages"].get("ors_ad")
        if st:
            st["open"] = dict(sorted(open_now.get(rec["id"], {}).items()))
    stats["ors_ad_records"] = len(touched)
    stats["ors_ad_unmatched_titles"] = len(unmatched - {""})
    return _sort(by_id.values()), stats


def settle(films: list[dict], withheld=frozenset()) -> tuple[list[dict], dict]:
    """Fold records created from filings into the film's own record once one
    exists (pure, no I/O). Runs every run before the chain lists, so a
    listing joins the film's record and not the filing's. A record created
    from filings that holds a rating or intake stage is the film's KMRB
    record (the rating joined it by title and applicant) and stays. The
    target is another record of the same title or original title under
    `pick` (same applicant, else activity within a year of the first filing,
    or of the day the record was first seen). A record still known only from
    filings whose applicant is masked or has gained adult KMRB ratings
    (`withheld`) is removed."""
    from .merge import _sort, fold_into

    stats = {"ors_ad_folded": 0, "ors_ad_removed": 0}
    ads = [f for f in films if f["id"].startswith(AD_PREFIX)
           and "rating" not in f["stages"] and "intake" not in f["stages"]]
    if not ads:
        return films, stats
    by_id = {f["id"]: f for f in films}
    others = [f for f in films if not f["id"].startswith(AD_PREFIX)]
    find = _title_index(others, [f["title"] for f in ads])
    for a in sorted(ads, key=lambda f: f["id"]):
        if ad_only(a) and (a.get("applicant", "") in ("", MASK)
                           or company_key(a["applicant"]) in withheld):
            del by_id[a["id"]]
            stats["ors_ad_removed"] += 1
            continue
        day = (a["stages"].get("ors_ad", {}).get("first_date")
               or min(a["first_seen"].values(), default="")[:10])
        cands = [f for f in find(a["title"], a.get("original_title", "")) if f["id"] in by_id]
        target = pick(cands, a.get("applicant", ""), day) if cands and day else None
        if target is None:
            continue
        fold_into(target, a)
        target["last_seen"] = max(target.get("last_seen", ""), a.get("last_seen", ""))
        del by_id[a["id"]]
        stats["ors_ad_folded"] += 1
    return _sort(by_id.values()), stats
