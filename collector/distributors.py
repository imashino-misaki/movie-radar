"""Per-company index of the published films (data/distributors.json).

Pure function over the final films list: for every distributor or applicant
company, the films it distributes (KOBIS 배급사) or filed for (the KMRB rating /
intake / trailer applicant), each with a short status.

Company identity: names from the KOBIS detail page's 배급사 entries (which
carry the KOBIS company code), the KOBIS Open API `distributors` and the
record's `applicant`. Names are compared by `companies.company_key` (legal
form, spacing and punctuation dropped). Two names are one company when they
share a key or a KOBIS code, or when on the same record one name's key starts
with the other's (the detail page writes "판씨네마(주) Pancinema Corp." where
the Open API writes "판씨네마(주)"). Masked applicants (개인) are skipped.

The applicant of a KOBIS-only record (`k` id) is a copy of its first KOBIS
distributor, not a filing, so it adds no applicant role. Per-format records
(`variant_of`) are listed under their lead's id, once.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

from .companies import company_key
from .privacy import MASK

# most advanced first; a format group takes the first status any member has
STATUSES = ("showing", "scheduled", "rated", "intake", "announced", "released", "dormant")
INACTIVE = frozenset({"released", "dormant"})
_CHAINS = ("megabox", "lotte")


def _opened(st: dict) -> str:
    """Latest opening or re-release date the record's stages give (ISO), or ""."""
    hist = st.get("release_history", {})
    book = st.get("booking", {})
    dates = [st.get("kobis", {}).get("open_dt", ""), hist.get("open_dt", ""),
             *hist.get("rereleases", []), book.get("open_dt", ""), book.get("rerelease_dt", ""),
             *(st.get(c, {}).get("release_date", "") for c in _CHAINS)]
    return max(dates, default="")


def film_status(rec: dict, today: dt.date) -> str:
    """One status per record, by precedence:

    dormant    tagged dormant (kobis_web.mark_dormant)
    showing    a chain lists it as now showing, or tickets are on sale and its
               latest opening / re-release date has passed
    scheduled  a coming date or month is known (calendar, chain, KOBIS), a
               chain lists it as upcoming, or advance tickets are on sale
    rated      KMRB rating newer than any past opening date
    intake     rating application in progress, newer than any past opening
    released   opened (a past opening date or KOBIS 제작상태 개봉), nothing current
    announced  known otherwise (KOBIS entry without a date, trailer filing, ...)
    """
    if "dormant" in rec.get("tags", []):
        return "dormant"
    st = rec["stages"]
    day, month = today.isoformat(), today.isoformat()[:7]
    opened = _opened(st)
    if any(st.get(c, {}).get("status") == "now" for c in _CHAINS):
        return "showing"
    if "booking" in st and opened and opened[:10] <= day:
        return "showing"
    future = [d for d in (st.get("schedule", {}).get("date", ""),
                          st.get("kobis", {}).get("open_dt", ""),
                          *(st.get(c, {}).get("release_date", "") for c in _CHAINS)) if d > day]
    months = [m for m in (st.get("schedule", {}).get("month", ""),
                          st.get("lotte", {}).get("release_month", "")) if m >= month]
    if (future or months or "schedule" in st or "booking" in st
            or any(st.get(c, {}).get("status") == "upcoming" for c in _CHAINS)):
        return "scheduled"
    past = opened if opened and opened[:10] <= day else ""
    if st.get("rating", {}).get("date", "") > past:
        return "rated"
    if "intake" in st and st["intake"].get("date", "") > past:
        return "intake"
    if past or st.get("kobis", {}).get("status") == "개봉" or \
            st.get("kobis_detail", {}).get("status") == "개봉":
        return "released"
    return "announced"


def _names(rec: dict) -> list[tuple[str, str, str]]:
    """(role, name, code) of the record's companies; role distributor / applicant."""
    st = rec["stages"]
    out = [("distributor", c["name"], c.get("code", ""))
           for c in st.get("kobis_detail", {}).get("companies", []) if c.get("role") == "배급사"]
    out += [("distributor", n, "") for n in st.get("kobis", {}).get("distributors", [])]
    if not rec["id"].startswith("k"):
        out.append(("applicant", rec.get("applicant", ""), ""))
    return [(r, n.strip(), c) for r, n, c in out if n and n.strip() != MASK and company_key(n)]


class _Union:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def build(films: list[dict], today: dt.date) -> dict:
    """{"statuses", "companies": [...]}; companies sorted by active films."""
    uf = _Union()
    for rec in films:
        names = _names(rec)
        keys = sorted({company_key(n) for _, n, _ in names}, key=len)
        for _, n, code in names:
            uf.find(("n", company_key(n)))
            if code:
                uf.union(("n", company_key(n)), ("c", code))
        for i, short in enumerate(keys):  # same record: one name extends the other
            for long in keys[i + 1:]:
                if long.startswith(short):
                    uf.union(("n", short), ("n", long))

    lead_of = {f["id"]: f.get("variant_of") or f["id"] for f in films}
    status: dict[str, str] = {}
    for rec in films:
        lid = lead_of[rec["id"]]
        s = film_status(rec, today)
        if lid not in status or STATUSES.index(s) < STATUSES.index(status[lid]):
            status[lid] = s

    spellings: dict = defaultdict(Counter)
    codes: dict = defaultdict(Counter)
    roles: dict = defaultdict(lambda: defaultdict(set))
    for rec in films:
        lid = lead_of[rec["id"]]
        for role, name, code in dict.fromkeys(_names(rec)):
            root = uf.find(("n", company_key(name)))
            spellings[root][name] += 1
            if code:
                codes[root][code] += 1
            roles[root][lid].add(role)

    companies = []
    for root, per_film in roles.items():
        spelled = sorted(spellings[root].items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
        name = spelled[0][0]
        code_list = [c for c, _ in codes[root].most_common()]
        cid = f"c{code_list[0]}" if code_list else f"n{company_key(name)}"
        entries = sorted(({"id": fid, "status": status[fid], "roles": sorted(r)}
                          for fid, r in per_film.items()),
                         key=lambda e: (STATUSES.index(e["status"]), e["id"]))
        counts = Counter(e["status"] for e in entries)
        companies.append({
            "id": cid, "name": name, "spellings": [s for s, _ in spelled[1:]],
            "kobis_codes": code_list,
            "counts": {"active": sum(v for k, v in counts.items() if k not in INACTIVE),
                       "total": len(entries),
                       "distributor": sum("distributor" in e["roles"] for e in entries),
                       "applicant": sum("applicant" in e["roles"] for e in entries),
                       **{s: counts[s] for s in STATUSES if counts[s]}},
            "films": entries})
    companies.sort(key=lambda c: (-c["counts"]["active"], -c["counts"]["total"], c["name"]))
    return {"statuses": list(STATUSES), "companies": companies}
