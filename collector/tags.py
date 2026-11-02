"""Tag heuristics. All keyword lists live here.

Tags are computed from the published fields only (title, original title,
applicant, country) and recomputed on every run, so editing a list below
re-tags the whole archive.

  chain_cgv / chain_megabox / chain_lotte
      The applicant is the exhibitor itself (or its affiliate). A theatre chain
      applying for a rating on its own usually means a release on that chain
      only (단독 유력) - concerts, events, re-releases, special formats.
  anime      극장판 / 애니메이션 keywords, 劇場版, or a known anime franchise name
             in the title. Only a fallback: the site's 애니메이션 category takes the
             official genre (Lotte / KOBIS) when any source gives one (site.is_anime).
             Kana in the original title is not used: Japanese live-action films have
             it too (날조, 꽃밥, 열정 ...).
  concert    concert / tour / live-capture keywords (콘서트, 투어, 라이브, 인 시네마 ...).
  rerelease  4K, 리마스터, 재개봉, 앙코르 ...
  format     special-format or alternate-cut versions (SCREENX, 4DX, IMAX, Dolby,
             무삭제, 감독판, 확장판 ...).
Matching is case-insensitive on title + original title: Hangul/CJK keywords
are substrings, Latin keywords must match on word boundaries (so "live" does
not hit "alive" or "olive").
"""
from __future__ import annotations

import re

CHAIN_APPLICANT_MARKERS = {
    "chain_cgv": ("씨제이씨지브이", "cjcgv", "씨지브이", "씨제이포디플렉스", "cj4dplex", "포디플렉스"),
    "chain_megabox": ("메가박스", "megabox"),
    "chain_lotte": ("롯데컬처웍스", "롯데시네마", "lottecultureworks", "lottecinema"),
}

ANIME_KEYWORDS = (
    "극장판", "애니메이션", "劇場版", "gekijouban", "gekijoban", "anime",
)
# bare "애니" also counts, except in common non-anime words (애니멀, 애니웨이 ...)
_ANI_BARE = re.compile(r"애니(?!멀|웨이|팡|콜)")
# Long-running anime franchises that regularly get Korean theatrical releases.
ANIME_FRANCHISES = (
    "명탐정 코난", "원피스", "귀멸의 칼날", "주술회전", "하이큐", "도라에몽", "짱구는 못말려",
    "포켓몬", "건담", "에반게리온", "나루토", "드래곤볼", "슬램덩크", "블루 록", "블루록",
    "체인소 맨", "체인소맨", "스파이 패밀리", "마크로스", "프리큐어", "진격의 거인",
    "나의 히어로 아카데미아", "신카이 마코토", "지브리", "소드 아트 온라인", "은혼",
    "러브 라이브", "러브라이브", "아이돌마스터", "걸즈 앤 판처", "헬로키티", "산리오",
)
CONCERT_KEYWORDS = (
    "콘서트", "투어", "라이브", "인 시네마", "인시네마", "공연실황", "실황", "팬미팅",
    "concert", "tour", "live", "in cinema", "in cinemas",
)
RERELEASE_KEYWORDS = (
    "4k", "리마스터", "재개봉", "앙코르", "remaster", "re-release", "encore",
)
FORMAT_KEYWORDS = (
    "스크린엑스", "screenx", "4dx", "아이맥스", "imax", "돌비", "dolby",
    "무삭제", "감독판", "확장판", "uncut", "director's cut", "extended",
)

SUBCULTURE_TAGS = ("anime", "concert", "rerelease")

# franchise names that contain a concert keyword but are films
NOT_CONCERT = ("러브 라이브", "러브라이브", "love live", "lovelive")

# a live sports broadcast in a theater (league names, match rounds); bare sport words
# ("축구", "야구") and "생중계" also appear in film titles, so they are not used
SPORTS = re.compile(
    r"(?<![a-z0-9])(kbo|mlb|nba|ufc|lck|epl|k\s*리그|k league)(?![a-z0-9])"
    r"|한국시리즈|포스트시즌|준플레이오프|플레이오프\s*\d|와일드카드\s*결정전|월드시리즈"
    r"|챔피언스리그|롤드컵|\d+\s*차전",
    re.I)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _matches(haystack: str, needle: str) -> bool:
    needle = needle.lower()
    if re.fullmatch(r"[a-z0-9' .-]+", needle):
        return re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", haystack) is not None
    return needle in haystack


def _has_any(haystack: str, needles) -> bool:
    return any(_matches(haystack, n) for n in needles)


def tag_film(title: str, original_title: str = "", applicant: str = "", country: str = "") -> list[str]:
    tags: list[str] = []
    applicant_c = _compact(applicant)
    for tag, markers in CHAIN_APPLICANT_MARKERS.items():
        if any(m in applicant_c for m in markers):
            tags.append(tag)

    text = f"{title or ''} | {original_title or ''}".lower()
    if _has_any(text, ANIME_KEYWORDS) or _ANI_BARE.search(text) or _has_any(text, ANIME_FRANCHISES):
        tags.append("anime")
    concert_text = text
    for name in NOT_CONCERT:
        concert_text = concert_text.replace(name, " ")
    if SPORTS.search(text):
        tags.append("sports")
    elif _has_any(concert_text, CONCERT_KEYWORDS):
        tags.append("concert")
    if _has_any(text, RERELEASE_KEYWORDS):
        tags.append("rerelease")
    if _has_any(text, FORMAT_KEYWORDS):
        tags.append("format")
    return tags
