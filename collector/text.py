"""Small HTML/text helpers shared by the parsers."""
from __future__ import annotations

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean(fragment) -> str:
    """Strip tags, unescape entities, collapse whitespace. JSON sources give
    some fields as numbers or null: None becomes "", other non-strings str()."""
    if fragment is None:
        return ""
    text = _TAG.sub(" ", fragment if isinstance(fragment, str) else str(fragment))
    text = html.unescape(text).replace("\xa0", " ")
    return _WS.sub(" ", text).strip()


# A parenthesised Hanja spelling of a Korean title ("공무도하(公無渡河)"): some
# sources add it, others list the plain title (checked 2026-11-01: Megabox and
# KOBIS write "공무도하(公無渡河)", KMRB "공무도하"). Only a group of Han
# ideographs is dropped; Latin, kana or Hangul in parentheses stays part of
# the title.
_HAN = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f"
_HANJA_PAREN = re.compile(rf"\(\s*[{_HAN}][{_HAN}\s·・,]*\)")


def norm_key(text: str) -> str:
    """Normalisation used for fuzzy joins: NFKC, lowercase, drop a parenthesised
    Hanja spelling (unless nothing else is left), spaces and punctuation."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    plain = _HANJA_PAREN.sub(" ", text)
    if any(ch.isalnum() for ch in plain):
        text = plain
    return "".join(ch for ch in text if ch.isalnum())


# Event series labels Megabox puts in front of a title ("[오페라] 라 보엠 @The
# Met", "[2026 NT Live] 골든 보이", "[RBO] 호두까기 인형"). KMRB rates films
# under the plain title: a KMRB title search for the labels ("NT Live", "RBO",
# "시네도슨트") found no film rating on 2026-11-01.
_LABEL_PREFIX = re.compile(r"^\s*\[[^\[\]]*\]\s*")


def strip_label(title: str) -> str:
    """Title without leading "[...]" series labels (unless nothing is left)."""
    out = title or ""
    while (m := _LABEL_PREFIX.match(out)) and out[m.end():].strip():
        out = out[m.end():]
    return out.strip()


# Grade names used before the current four (전체관람가, 12세이상관람가,
# 15세이상관람가, 청소년관람불가), read from the KMRB film-rating list on
# 2026-11-01: 1990-1997 연소자관람가 / 중학생이상(연령층)관람가 / 고등학생이상관람가
# / 연소자관람불가; 1997-1999 "...미만인자는관람할수없는등급"; 1999-2006 15세관람가 /
# 18세관람가 (with 전체관람가 / 12세이상관람가). The old names appear on ratings up
# to 2006-10-26 and the current ones from 2006-10-30. KOBIS keeps the old name
# on old audits (트레인스포팅 96-437 연소자관람불가, 마틸다 96-443 연소자관람가).
OLD_GRADES = {
    "연소자관람가": "전체관람가",
    "모든관람객이관람할수있는등급": "전체관람가",
    "중학생이상관람가": "12세이상관람가",
    "중학생이상(연령층)관람가": "12세이상관람가",
    "12세미만인자는관람할수없는등급": "12세이상관람가",
    "고등학생이상관람가": "15세이상관람가",
    "15세미만인자는관람할수없는등급": "15세이상관람가",
    "15세관람가": "15세이상관람가",
    "연소자관람불가": "청소년관람불가",
    "18세미만인자는관람할수없는등급": "청소년관람불가",
    "18세관람가": "청소년관람불가",
}


def grade_key(grade: str) -> str:
    """A grade under its current name (old names mapped, others unchanged)."""
    g = "".join((grade or "").split())
    return OLD_GRADES.get(g, g)


# Screening formats KMRB rates separately and appends to the film's title
# ("하이브 치어링 파티 (3D)", "인턴 (스크린엑스)"). Seen in the rating list on
# 2026-11-01: 2D, 3D, 4DX, SX, 스크린엑스, dolby; the rest are the other format
# names cinemas use for the same kinds of screens.
FORMATS = ("2D", "3D", "4D", "4DX", "MX4D", "SX", "ScreenX", "스크린엑스", "IMAX", "아이맥스",
           "Dolby", "돌비", "Dolby Cinema", "돌비시네마", "Dolby Atmos", "돌비애트모스", "ATMOS")
_FORMAT_KEYS = {"".join(ch for ch in unicodedata.normalize("NFKC", f).lower() if ch.isalnum())
                for f in FORMATS}
_TRAILING_PAREN = re.compile(r"\s*\(\s*([^()]*?)\s*\)\s*$")


def split_format(title: str) -> tuple[str, str]:
    """(title without a trailing screening-format label, the label or "")."""
    m = _TRAILING_PAREN.search(title or "")
    if m and norm_key(m.group(1)) in _FORMAT_KEYS and norm_key(title[:m.start()]):
        return title[:m.start()].rstrip(), m.group(1)
    return title or "", ""


def format_rank(rec: dict) -> tuple:
    """Order of the records of one film rated per format: the one a chain or
    KOBIS already lists first, then the 2D rating, then by id."""
    st = rec.get("stages", {})
    fmt = split_format(rec.get("title", ""))[1]
    return (not any(k in st for k in ("megabox", "lotte", "cgv", "kobis")),
            norm_key(fmt) != "2d", rec.get("id", ""))


def is_truncated(title: str) -> bool:
    """KMRB lists cut long titles to a fixed width and end them with "..."."""
    return (title or "").rstrip().endswith(("...", "…"))


def same_title(a: str, b: str) -> bool:
    """Exact normalised match; a truncated title ("거짓말 : 4K 무...") matches a
    title that starts with the same text."""
    ka, kb = norm_key(a), norm_key(b)
    if ka == kb:
        return True
    return bool((is_truncated(a) and ka and kb.startswith(ka))
                or (is_truncated(b) and kb and ka.startswith(kb)))


def fuller(old: str, new: str) -> str:
    """Of two spellings of the same title, the one that is not truncated."""
    if new and is_truncated(new) and old and not is_truncated(old) and same_title(old, new):
        return old
    return new or old


class TitleIndex:
    """Records by normalised title; a lookup also finds records whose stored
    title is a truncated prefix of the wanted title, and (for a truncated wanted
    title) records whose title starts with it. A title no record has exactly
    finds the records of the same title rated per screening format ("… (3D)"),
    only the first of them by format_rank."""

    def __init__(self, films, skip=lambda f: False, wanted=()):
        """`wanted`: every title that will be looked up this pass; a truncated
        record that is a prefix of more than one of them is ambiguous and is
        not matched by prefix."""
        self.exact: dict[str, list[dict]] = {}
        self.truncated: list[tuple[str, dict]] = []
        self.by_base: dict[str, list[dict]] = {}
        self.wanted = {norm_key(t) for t in wanted} - {""}
        for f in films:
            if skip(f):
                continue
            title = f.get("title", "")
            self.exact.setdefault(norm_key(title), []).append(f)
            if is_truncated(title) and norm_key(title):
                self.truncated.append((norm_key(title), f))
            base, fmt = split_format(title)
            if fmt:
                self.by_base.setdefault(norm_key(base), []).append(f)

    def find(self, title: str) -> list[dict]:
        k = norm_key(title)
        if not k:
            return []
        found = list(self.exact.get(k, []))
        if not found and self.by_base.get(k):
            return [min(self.by_base[k], key=format_rank)]
        prefix = [(pk, f) for pk, f in self.truncated if pk != k and k.startswith(pk)]
        prefix = [f for pk, f in prefix
                  if sum(w.startswith(pk) for w in self.wanted | {k}) == 1]
        if len(prefix) == 1:  # only an unambiguous prefix match counts
            found += prefix
        if is_truncated(title):
            longer = [f for key, fs in self.exact.items() if key != k and key.startswith(k) for f in fs]
            if len({norm_key(f.get("title", "")) for f in longer}) == 1:
                found += longer
        out, seen = [], set()
        for f in found:
            if id(f) not in seen:
                seen.add(id(f))
                out.append(f)
        return out
