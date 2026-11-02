"""Manual allow / block lists of companies and company representatives.

The automatic trust rules (applicants.py, companies.py) miss two cases: a
legitimate company they cannot vouch for, and a company that files films for
a token theatrical release under a fresh name or representative. The lists
let the maintainer decide those by hand:

  {"block_companies": [...], "allow_companies": [...],
   "block_reps": [...], "allow_reps": [...]}

Company names are compared without legal form, spacing and punctuation;
representative names without spacing. A block always wins over an allow,
and both win over the automatic rules: a blocked company's films are not
published at all (not even with a general rating or a cinema listing), an
allowed company is trusted.

The lists are not part of the repository: the collector reads them from the
RADAR_LISTS environment variable (an Actions secret), or from the git-ignored
file .private/lists.json for local runs. Only counts are written to meta.json.

A company's representative comes from the KMRB rating list's Excel export
(every rated applicant, kmrb.fetch_reps), the ORS intake list and the KOBIS
company register (companies.json).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .companies import company_key
from .text import norm_key

ENV = "RADAR_LISTS"
LOCAL_FILE = Path(".private/lists.json")
KEYS = ("block_companies", "allow_companies", "block_reps", "allow_reps")


@dataclass(frozen=True)
class Lists:
    block_companies: frozenset = field(default_factory=frozenset)
    allow_companies: frozenset = field(default_factory=frozenset)
    block_reps: frozenset = field(default_factory=frozenset)
    allow_reps: frozenset = field(default_factory=frozenset)

    def __bool__(self) -> bool:
        return any((self.block_companies, self.allow_companies, self.block_reps, self.allow_reps))

    def decide(self, name: str, reps=()) -> str | None:
        """"block", "allow" or None for a company and its representatives."""
        key, rep_keys = company_key(name), {norm_key(r) for r in reps if r}
        if key in self.block_companies or rep_keys & self.block_reps:
            return "block"
        if key in self.allow_companies or rep_keys & self.allow_reps:
            return "allow"
        return None


def parse(data: dict) -> Lists:
    def keys(name: str, fn) -> frozenset:
        values = data.get(name) or []
        if not isinstance(values, list):
            raise ValueError(f"{name} must be a list")
        return frozenset(k for k in (fn(str(v)) for v in values) if k)

    unknown = set(data) - set(KEYS)
    if unknown:
        raise ValueError(f"unknown list names: {sorted(unknown)}")
    return Lists(keys("block_companies", company_key), keys("allow_companies", company_key),
                 keys("block_reps", norm_key), keys("allow_reps", norm_key))


def load(environ=None, local_file: Path | None = None) -> Lists:
    environ = os.environ if environ is None else environ
    text = environ.get(ENV, "").strip()
    path = LOCAL_FILE if local_file is None else local_file
    if not text and path.exists():
        text = path.read_text(encoding="utf-8").strip()
    return parse(json.loads(text)) if text else Lists()
