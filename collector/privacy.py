"""Applicant masking.

Applicants that look like an individual's name are published as "개인".
Rule: the whole applicant string (spaces removed) is 2-4 Hangul syllables
(^[가-힣]{2,4}$) and contains none of the corporate markers below.
Anything else (company forms, Latin text, longer names) is kept, because
company names are needed for chain detection.
This deliberately errs toward masking: a short company name with no marker
(e.g. a two-syllable distributor name) is also masked.
"""
from __future__ import annotations

import re

MASK = "개인"

_PERSON_LIKE = re.compile(r"^[가-힣]{2,4}$")

# Substrings that mark a short Hangul applicant as an organisation, not a person.
CORPORATE_MARKERS = (
    "주식회사", "유한회사", "회사", "재단", "협회", "법인", "조합", "센터", "학교", "대학",
    "극단", "방송", "필름", "픽쳐스", "픽처스", "시네마", "영화사", "미디어", "컴퍼니",
    "스튜디오", "엔터", "메가박스", "씨지브이", "롯데",
)


def is_person_like(applicant: str) -> bool:
    compact = re.sub(r"\s+", "", applicant or "")
    if not _PERSON_LIKE.match(compact):
        return False
    return not any(marker in compact for marker in CORPORATE_MARKERS)


def mask_applicant(applicant: str) -> str:
    applicant = (applicant or "").strip()
    return MASK if is_person_like(applicant) else applicant
