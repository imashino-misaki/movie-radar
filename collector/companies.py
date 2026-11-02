"""KOBIS company check for applicants and distributors KMRB has no record of.

An applicant is trusted when it has at least one non-adult and no adult KMRB
rating in the last 12 months (applicants.py). Companies that filed no rating
themselves in that window are unknown to that rule, and their films are
withheld; on 2026-11-01 that was 11 ORS applicants (e.g. (주)콘텐츠판다,
더핑크퐁컴퍼니 주식회사) and 40 distributors of withheld KOBIS films (e.g.
(주)쇼박스, (주)넥스트엔터테인먼트월드(NEW)).

For those names the KOBIS company register is read (anonymous website pages):
  1. searchCompanyList.do (POST sCompanyNm): the company whose name equals the
     applicant's apart from the legal form ((주), 주식회사, ...) and spacing;
     exactly one such company, else the name stays unknown;
  2. searchCompanyDtl.do?code=..&sType=filmo: the company's most recent films
     (10 a page) with their genres.
A company with at least one film and none in the 성인물 genre is trusted, the
same test the KMRB rule applies to ratings. Checked 2026-11-01: (주)콘텐츠판다
9 films, 0 adult; (주)엠디에스스피어 10/0; (주)코빈커뮤니케이션즈 10/10;
(주)도키엔터테인먼트 10/10; 유한회사 스튜디오루나 10/10. Names KMRB does know
are never checked here, so a company with adult KMRB ratings stays untrusted
whatever its KOBIS films show ((주)제이앤씨미디어그룹: 27 adult ratings,
no 성인물 among its 9 latest KOBIS films).

The search result rows also show the representative (대표자, public in the
register) and an address; the code, name and representative are read, the
representative for the manual block / allow lists (lists.py). Results are
cached in data/companies.json (name -> {"code", "rep", "films", "adult",
"checked"}) and read again after RECHECK_DAYS.
"""
from __future__ import annotations

import datetime as dt
import re

from .http import BlockedError, FetchError, check_block
from .privacy import MASK
from .text import clean, norm_key

SEARCH_URL = "https://www.kobis.or.kr/kobis/business/mast/comp/searchCompanyList.do"
DETAIL_URL = "https://www.kobis.or.kr/kobis/business/mast/comp/searchCompanyDtl.do"
ADULT_GENRE = "성인물"
# a company's register changes slowly (its first page holds its 10 latest
# films); a month keeps a new adult film from going unnoticed for long
RECHECK_DAYS = 30
_LEGAL = re.compile(r"\((?:주|유|사|재)\)|㈜|주식회사|유한회사|유한책임회사|사단법인|재단법인|"
                    r"합자회사|합명회사|협동조합")
_PAREN = re.compile(r"\([^()]*\)")
_COMPANY = re.compile(r"mstView\('company','(\d+)'\);return false;\"[^>]*>([^<]*)<")
_FILM = re.compile(r"<dt[^>]*>\s*<a[^>]*mstView\('movie','(\d+)'\)[^>]*>.*?</dt>.*?"
                   r'<dd class="minfo">(.*?)</dd>', re.S)


def company_key(name: str) -> str:
    """A company name without its legal form, spacing and punctuation."""
    return norm_key(_LEGAL.sub(" ", name or ""))


def same_company(a: str, b: str) -> bool:
    """Equal names apart from legal form and spacing, also when one of them
    adds a parenthesised alias ("무호흡필름(Apnea film)")."""
    return company_key(a) == company_key(b) or (
        bool(company_key(_PAREN.sub(" ", a))) and company_key(_PAREN.sub(" ", a)) == company_key(_PAREN.sub(" ", b)))


def search_terms(name: str) -> list[str]:
    """Pieces of the name to search for, longest first (the register search
    matches a part of the stored name, which may carry the legal form
    elsewhere: "롯데컬처웍스(주)롯데엔터테인먼트")."""
    pieces = [clean(p) for p in _PAREN.split(_LEGAL.sub("|", name)) for p in p.split("|")]
    pieces += [clean(p) for p in re.findall(r"\(([^()]*)\)", _LEGAL.sub("", name))]
    return sorted(dict.fromkeys(p for p in pieces if len(norm_key(p)) >= 2), key=len, reverse=True)


def parse_search(html: str) -> list[tuple[str, str]]:
    """(code, name) of the companies on one search result page."""
    return [(code, name) for code, name, _ in parse_search_rows(html)]


_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
REP_COLUMN = 3  # 영화사명, 영화사명(영문), 분류, 대표자, ... (checked 2026-11-01)


def parse_search_rows(html: str) -> list[tuple[str, str, str]]:
    """(code, name, representative) per name link on one search result page
    (the Korean and the English name of a company are both links)."""
    if "<tbody" not in html:
        check_block(html, markup_ok=False)
        raise FetchError("kobis company search: result table missing")
    body = html.split("<tbody", 1)[1].split("</tbody>", 1)[0]
    out = []
    for tr in _ROW.findall(body):
        tds = _TD.findall(tr)
        who = clean(re.sub(r"<[^>]+>", " ", tds[REP_COLUMN])) if len(tds) > REP_COLUMN else ""
        out += [(code, clean(name), who) for code, name in _COMPANY.findall(tr) if clean(name)]
    return list(dict.fromkeys(out))


def parse_filmo(html: str) -> dict:
    """{"films", "adult"}: films on the company's filmography page, and how
    many of them have the 성인물 genre."""
    if 'class="fmList"' not in html and "fmList" not in html:
        check_block(html, markup_ok=False)
        raise FetchError("kobis company filmography: list missing")
    films = {}
    for code, info in _FILM.findall(html):
        films[code] = ADULT_GENRE in info
    return {"films": len(films), "adult": sum(films.values())}


def verified(entry: dict | None) -> bool:
    return bool(entry) and entry.get("films", 0) >= 1 and entry.get("adult", 0) == 0


def _due(entry: dict | None, stamp: str) -> bool:
    if not entry or not entry.get("checked"):
        return True
    age = dt.date.fromisoformat(stamp) - dt.date.fromisoformat(entry["checked"])
    return age.days >= RECHECK_DAYS


def lookup(client, name: str) -> dict:
    """The register entry for one name: {"code", "rep", "films", "adult"}; code ""
    when no single company matches."""
    for term in search_terms(name)[:2]:
        rows = parse_search_rows(client.post(SEARCH_URL, data={
            "sCompanyNm": term, "searchType": "search", "curPage": "1", "useYn": "Y"}).text)
        exact = {code: who for code, found, who in rows if same_company(found, name)}
        if len(exact) == 1:
            code, who = exact.popitem()
            page = client.get(DETAIL_URL, params={"code": code, "sType": "filmo"}).text
            return {"code": code, "rep": who, **parse_filmo(page)}
        if exact:
            break  # several companies of that name: leave it unknown
    return {"code": "", "rep": "", "films": 0, "adult": 0}


def check(client, names, known: set[str], cache: dict, stamp: str) -> dict:
    """Look up (or reuse) the register entry of every name KMRB does not know.
    `known` = names with KMRB ratings in the trust window. Updates `cache` in
    place, dropping names no longer asked about. One failing name is retried
    next run; a block stops the source."""
    known_keys = {company_key(n) for n in known}
    wanted = sorted({n for n in names if n and n != MASK and company_key(n)
                     and company_key(n) not in known_keys})
    looked = errors = 0
    for name in wanted:
        if not _due(cache.get(name), stamp):
            continue
        try:
            cache[name] = {**lookup(client, name), "checked": stamp}
            looked += 1
        except BlockedError:
            raise
        except Exception:  # noqa: BLE001 - one bad page must not fail the source
            errors += 1
    for name in [n for n in cache if n not in wanted]:
        del cache[name]
    return {"names": len(wanted), "looked_up": looked, "errors": errors,
            "found": sum(bool(cache.get(n, {}).get("code")) for n in wanted),
            "verified": sum(verified(cache.get(n)) for n in wanted)}


def trusted_names(cache: dict) -> set[str]:
    return {name for name, entry in cache.items() if verified(entry)}
