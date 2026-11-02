"""Join intake (ORS) and rating (KMRB) rows and merge them into the archive.

Join key: ORS RCV_NO == KMRB selectPageInfo id (same internal receipt id,
checked against the ORS detail page). When a row lacks it, the fallback key is
normalised title + (masked) applicant.

Adult filtering (nothing adult is ever written):
  * KMRB rows that appear only in the adult_yn='Y' list are adult; dropped.
  * ORS intake rows have no adult flag. An intake row is published only if
    (a) its applicant is trusted (see applicants.py: >=1 non-adult and 0 adult
    KMRB ratings in the last 12 months; "개인" never), or
    (b) it joins a film that already has a non-adult KMRB rating, or
    (c) a cinema chain lists a film of the same title (stage megabox / lotte:
        a chain's theatrical programme is not adult content); the row joins
        that record.
    Every other intake row is withheld: not written anywhere, only counted.
  * A KMRB general (adult_yn='N') row graded 청소년관람불가 is published only
    if its applicant is trusted; otherwise it is withheld and treated exactly
    like an adult row (KMRB's own adult flag misses some explicit titles).
    Non-청불 general rows are always published.
  * An archived record that now matches an adult or withheld rating, or is intake-only
    and its applicant is no longer trusted, is removed.
All applicant comparisons use the masked applicant, the same value the
archive stores.

first_seen.<stage> is set once, the first run a stage is seen, and never
changed afterwards. last_seen is the last run date the film was listed by
either source.
"""
from __future__ import annotations

import hashlib

from .privacy import MASK, mask_applicant
from .tags import tag_film
from .text import TitleIndex, format_rank, fuller, grade_key, norm_key, split_format


RESTRICTED_GRADE = "청소년관람불가"


def restricted_withheld(general_rows: list[dict], trusted: set[str]) -> list[dict]:
    """General (adult_yn='N') rows graded 청소년관람불가 whose applicant is not trusted."""
    out = []
    for r in general_rows:
        applicant = mask_applicant(r["applicant"])
        if RESTRICTED_GRADE in grade_key(r.get("grade", "")) and (applicant == MASK or applicant not in trusted):
            out.append(r)
    return out


def film_id(rcv_no: str, title: str, applicant: str) -> str:
    if rcv_no:
        return f"r{rcv_no}"
    digest = hashlib.sha1(f"{norm_key(title)}|{norm_key(applicant)}".encode()).hexdigest()
    return f"t{digest[:12]}"


def fallback_key(title: str, applicant: str) -> str | None:
    t = norm_key(title)
    return f"{t}|{norm_key(applicant)}" if t else None


def chain_listed(rec: dict) -> bool:
    """A cinema chain lists (or listed) the film."""
    return any(k in rec.get("stages", {}) for k in ("megabox", "lotte", "cgv"))


def merge(existing: list[dict], intake_rows: list[dict], general_rows: list[dict],
          adult_rows: list[dict], today: str, trusted: set[str]) -> tuple[list[dict], dict]:
    """Return (films, stats). Pure function: no I/O. `trusted` = masked applicant names."""
    def is_trusted(applicant: str) -> bool:
        return applicant != MASK and applicant in trusted

    withheld = restricted_withheld(general_rows, trusted)
    withheld_ids = {id(r) for r in withheld}
    rating_rows = len(general_rows)
    general_rows = [r for r in general_rows if id(r) not in withheld_ids]
    adult_rows = list(adult_rows) + withheld

    stats = {"intake_rows": len(intake_rows), "intake_published": 0, "intake_withheld": 0,
             "rating_rows": rating_rows, "rating_published": len(general_rows),
             "rating_withheld_restricted": len(withheld),
             "adult_rating_rows": len(adult_rows) - len(withheld),
             "archive_removed": 0, "join_by_rcv": 0, "join_by_title": 0, "intake_by_chain": 0,
             "masked_applicants": 0}
    adult_ids = {film_id(r["rcv_no"], r["title"], mask_applicant(r["applicant"])) for r in adult_rows}
    adult_keys = {k for r in adult_rows
                  if (k := fallback_key(r["title"], mask_applicant(r["applicant"])))}

    films: dict[str, dict] = {}
    chain_index = TitleIndex([f for f in existing if chain_listed(f)])
    for rec in existing:
        rec = {**rec, "stages": dict(rec.get("stages", {})),
               "first_seen": dict(rec.get("first_seen", {}))}
        # a record known only from trailer/poster ratings is published on its
        # applicant's trust alone (like an intake row): it goes when trust goes
        # (an ORS filing joined to it does not change that: ors_ad.py). A
        # record created from ORS filings (id "a…") follows ors_ad.settle's
        # rule instead and is routed by theatrical.py
        promo_only = (not rec["id"].startswith("a")
                      and set(rec["stages"]) <= {"promo", "past_rating", "ors_ad", "program"})
        is_adult = (rec["id"] in adult_ids
                    or (promo_only and not is_trusted(rec.get("applicant", "")))
                    or fallback_key(rec.get("title", ""), rec.get("applicant", "")) in adult_keys
                    or ("intake" in rec["stages"] and "rating" not in rec["stages"]
                        and not is_trusted(rec.get("applicant", "")) and not chain_listed(rec))
                    or (RESTRICTED_GRADE in grade_key(rec["stages"].get("rating", {}).get("grade", ""))
                        and not is_trusted(rec.get("applicant", ""))))
        if is_adult:
            stats["archive_removed"] += 1
            continue
        films[rec["id"]] = rec

    by_key: dict[str, str] = {}

    def index_key(rec: dict) -> None:
        k = fallback_key(rec.get("title", ""), rec.get("applicant", ""))
        if k:
            by_key.setdefault(k, rec["id"])

    for rec in films.values():
        index_key(rec)

    def locate(rcv_no: str, title: str, applicant: str) -> tuple[dict | None, str]:
        fid = film_id(rcv_no, title, applicant)
        if fid in films:
            return films[fid], "rcv" if rcv_no else "title"
        k = fallback_key(title, applicant)
        if k and k in by_key:
            cand = films[by_key[k]]
            # never merge two different receipts (e.g. a later re-release with the same title)
            if not (rcv_no and cand.get("rcv_no") and cand["rcv_no"] != rcv_no):
                return cand, "title"
        return None, ""

    def new_record(rcv_no: str, title: str, original: str, applicant: str) -> dict:
        rec = {"id": film_id(rcv_no, title, applicant), "rcv_no": rcv_no, "title": title,
               "original_title": original, "applicant": applicant, "stages": {},
               "first_seen": {}, "last_seen": today, "tags": []}
        films[rec["id"]] = rec
        index_key(rec)
        return rec

    masked_seen: set[str] = set()

    # --- ratings (authoritative for title / applicant) ---
    for r in general_rows:
        applicant = mask_applicant(r["applicant"])
        if applicant != r["applicant"].strip():
            masked_seen.add(r["applicant"].strip())
        rec, _ = locate(r["rcv_no"], r["title"], applicant)
        if rec is None:
            rec = new_record(r["rcv_no"], r["title"], r["original_title"], applicant)
        # KMRB truncates long titles: keep a full spelling another source gave
        rec["title"] = fuller(rec.get("title", ""), r["title"])
        rec["original_title"] = fuller(rec.get("original_title", ""), r["original_title"])
        rec["applicant"] = applicant
        if r["rcv_no"] and not rec.get("rcv_no"):
            rec["rcv_no"] = r["rcv_no"]
        rec["stages"]["rating"] = {
            "date": r["date"], "grade": r["grade"], "rating_no": r["rating_no"],
            "runtime": r["runtime"], "country": r["country"], "kind": r["kind"],
        }
        rec["first_seen"].setdefault("rating", today)
        rec["last_seen"] = today

    # --- intake ---
    for r in intake_rows:
        applicant = mask_applicant(r["applicant"])
        if film_id(r["rcv_no"], r["title"], applicant) in adult_ids:
            stats["intake_withheld"] += 1
            continue
        rec, how = locate(r["rcv_no"], r["title"], applicant)
        rated = rec is not None and "rating" in rec["stages"]
        if not rated and not is_trusted(applicant) and not (rec is not None and chain_listed(rec)):
            listed = [f for f in chain_index.find(r["title"]) if f["id"] in films]
            if len(listed) != 1:
                stats["intake_withheld"] += 1
                continue
            rec = films[listed[0]["id"]]  # (c): a chain lists the film (films holds copies)
            stats["intake_by_chain"] += 1
        stats["intake_published"] += 1
        if applicant != r["applicant"].strip():
            masked_seen.add(r["applicant"].strip())
        if rec is None:
            rec = new_record(r["rcv_no"], r["title"], r["original_title"], applicant)
        elif rated:
            stats["join_by_rcv" if how == "rcv" else "join_by_title"] += 1
        if "rating" not in rec["stages"]:
            rec["title"] = r["title"] or rec.get("title", "")
            rec["original_title"] = r["original_title"] or rec.get("original_title", "")
            rec["applicant"] = applicant
        if r["rcv_no"] and not rec.get("rcv_no"):
            rec["rcv_no"] = r["rcv_no"]
        rec["stages"]["intake"] = {
            "receipt_no": r["receipt_no"], "date": r["date"], "status": r["status"],
            "progress": r.get("progress", ""), "tab": r["tab"],
        }
        rec["first_seen"].setdefault("intake", today)
        rec["last_seen"] = today

    for rec in films.values():
        country = rec["stages"].get("rating", {}).get("country", "")
        rec["tags"] = tag_film(rec.get("title", ""), rec.get("original_title", ""),
                               rec.get("applicant", ""), country)

    stats["masked_applicants"] = len(masked_seen)
    out = sorted(films.values(), key=lambda f: (max(f["first_seen"].values(), default=""),
                                                f["id"]), reverse=True)
    return out, stats


def upgrade_title(rec: dict, title: str, original: str = "") -> None:
    """Replace a truncated title (or original title) of `rec` with the full
    spelling of the same title, in place."""
    rec["title"] = fuller(title, rec.get("title", ""))
    if original:
        rec["original_title"] = fuller(original, rec.get("original_title", ""))


def fold_into(dst: dict, src: dict) -> None:
    """Fold record `src` into `dst` (in place): every stage `dst` lacks is
    carried over, promo stages are united, and each first_seen keeps the
    earliest date."""
    from .ors_ad import merge_stage as merge_ors_ad
    from .promo import merge_stage  # local import: promo imports from here

    upgrade_title(dst, src.get("title", ""), src.get("original_title", ""))
    for k, v in src.get("first_seen", {}).items():
        dst["first_seen"][k] = min(v, dst["first_seen"].get(k, v))
    for k, v in src.get("stages", {}).items():
        if k == "promo":
            dst["stages"]["promo"] = merge_stage(dst["stages"].get("promo"), v)
        elif k == "ors_ad":
            dst["stages"]["ors_ad"] = merge_ors_ad(dst["stages"].get("ors_ad"), v)
        else:
            dst["stages"].setdefault(k, v)


def _sort(films):
    return sorted(films, key=lambda f: (max(f["first_seen"].values(), default=""), f["id"]),
                  reverse=True)


def apply_kobis(films: list[dict], cache: dict, kept_ids: list[str], trusted: set[str],
                withheld_audits: set[str], today: str,
                allow_new: bool = True,
                blocked: frozenset | set = frozenset()) -> tuple[list[dict], dict]:
    """Attach KOBIS data (pure, no I/O).

    Join: KOBIS audits[].auditNo == KMRB rating number -> stages["kobis"] on the
    existing record. Unmatched (KOBIS-only) titles become records with only a
    "kobis" stage when kobis.publishable() allows it; otherwise they are withheld
    (counted only). first_seen.kobis is the date KOBIS first listed the title.
    """
    from .kobis import ADULT_GENRE, audit_no, iso_date, publishable  # local import: kobis imports nothing from here

    stats = {"kobis_matched": 0, "kobis_only_published": 0, "kobis_only_withheld": 0,
             "kobis_pending_info": 0, "kobis_adult_removed": 0, "kobis_only_deferred": 0}
    by_id = {f["id"]: f for f in films}
    by_rating_no = {f["stages"]["rating"]["rating_no"]: f for f in films
                    if f["stages"].get("rating", {}).get("rating_no")}
    for cd in kept_ids:
        entry = cache.get(cd)
        if not entry or not entry.get("fetched_at"):
            stats["kobis_pending_info"] += 1
            continue
        stage = {"movie_cd": cd, "open_dt": iso_date(entry.get("openDt", "")),
                 "status": entry.get("prdtStatNm", ""),
                 "distributors": [mask_applicant(d) for d in entry.get("distributors", [])],
                 "genres": [g for g in entry.get("genres", []) if g],
                 "grades": sorted({a.get("watchGradeNm", "") for a in entry.get("audits", [])} - {""}),
                 "audit_nos": sorted({audit_no(a) for a in entry.get("audits", [])} - {""})}
        seen = entry.get("first_seen", today)
        kid = f"k{cd}"
        match = next((by_rating_no[audit_no(a)] for a in entry.get("audits", [])
                      if audit_no(a) in by_rating_no), None)
        denied = (any(ADULT_GENRE in g for g in entry.get("genres", []))
                  or any(audit_no(a) in withheld_audits for a in entry.get("audits", [])))
        if match is not None and denied:
            # KOBIS calls it 성인물 (or it shares a withheld rating): fail closed
            by_id.pop(match["id"], None)
            by_id.pop(kid, None)
            stats["kobis_adult_removed"] += 1
            continue
        if match is not None:
            stats["kobis_matched"] += 1
            old = by_id.pop(kid, None) if kid != match["id"] else None
            if old:  # a KOBIS-only record found its KMRB rating: fold it in
                seen = min(seen, old["first_seen"].get("kobis", seen))
                fold_into(match, old)
            match["stages"]["kobis"] = stage
            match["first_seen"]["kobis"] = min(seen, match["first_seen"].get("kobis", seen))
            continue
        if entry.get("withheld") or not publishable(entry, trusted, withheld_audits, blocked):
            stats["kobis_only_withheld"] += 1
            by_id.pop(kid, None)
            continue
        if kid not in by_id and not allow_new:
            # KMRB failed this run: withheld ratings are unknown, so no new KOBIS-only title
            stats["kobis_only_deferred"] += 1
            continue
        stats["kobis_only_published"] += 1
        names = entry.get("names", {})
        rec = by_id.get(kid) or {"id": kid, "rcv_no": "", "stages": {}, "first_seen": {}}
        rec.update(title=names.get("ko", ""), original_title=names.get("og") or names.get("en", ""),
                   applicant=stage["distributors"][0] if stage["distributors"] else "",
                   last_seen=today)
        rec["stages"]["kobis"] = stage
        rec["first_seen"].setdefault("kobis", seen)
        nations = entry.get("nations", [])
        rec["tags"] = tag_film(rec["title"], rec["original_title"], rec["applicant"],
                               nations[0] if nations else "")
        by_id[kid] = rec
    for fid in [i for i in by_id if i.startswith("k")]:
        if cache.get(fid[1:], {}).get("withheld"):  # titles no longer publishable
            del by_id[fid]
            stats["kobis_only_withheld"] += 1
    return _sort(by_id.values()), stats


def apply_megabox(films: list[dict], rows: list[dict], today: str) -> tuple[list[dict], dict]:
    """Attach the Megabox list (pure, no I/O).

    Rows are grouped by base title (Megabox lists 자막/더빙 versions as separate
    rows). A group joins an existing record whose normalised title matches;
    among several candidates the one whose KOBIS open date equals the Megabox
    release date wins, otherwise the most recently seen record. Unmatched groups
    become Megabox-only records (id "m" + hash of the title key); when such a
    title later gains a KMRB/KOBIS record, the Megabox-only record is folded in.
    Records that held a Megabox stage but are not in today's list are kept with
    status "ended".

    Megabox-only records are published without the trusted-applicant rule: the
    chain's public listing is a curated theatrical programme, not an applicant
    queue, and has no applicant to judge.
    """
    from .megabox import base_title, megabox_id, title_key  # local import, as in apply_kobis

    stats = {"megabox_groups": 0, "megabox_matched": 0, "megabox_only": 0,
             "megabox_ended": 0}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        k = title_key(r["title"])
        if k:
            groups.setdefault(k, []).append(r)

    by_id = {f["id"]: f for f in films}
    # chain-only records (m Megabox, l Lotte, c CGV) are not join targets: their
    # own chain step folds them into the record this listing joins
    index = TitleIndex(films, skip=lambda f: f["id"][:1] in ("m", "l", "c"),
                       wanted=[base_title(g[0]["title"]) for g in groups.values()])

    def pick(cands: list[dict], release: str) -> dict:
        exact = [f for f in cands if release and f["stages"].get("kobis", {}).get("open_dt") == release]
        pool = exact or cands
        return max(pool, key=lambda f: (max(f["first_seen"].values(), default=""), f["id"]))

    assigned: set[str] = set()
    for k, grp in sorted(groups.items()):
        stats["megabox_groups"] += 1
        dates = sorted(r["release_date"] for r in grp if r["release_date"])
        statuses = {r["status"] for r in grp}
        stage = {"movie_nos": sorted(r["movie_no"] for r in grp),
                 "status": "now" if "now" in statuses else ("upcoming" if "upcoming" in statuses
                                                             else sorted(statuses)[0]),
                 "release_date": dates[0] if dates else "",
                 "grade": next((r["grade"] for r in grp if r["grade"]), ""),
                 "exclusive": any(r["exclusive"] for r in grp),
                 "poster": next((r["poster"] for r in grp if r.get("poster")), "")}
        if any(r.get("classic") for r in grp):
            stage["classic"] = True
        mid = megabox_id(k)
        if mid not in by_id:  # a Megabox-only record stored under an earlier title key keeps its id
            mid = next((i for i, f in by_id.items()
                        if i.startswith("m") and title_key(f["title"]) == k), mid)
        seen = today
        cands = [f for f in index.find(base_title(grp[0]["title"])) if f["id"] in by_id]
        if cands:
            rec = pick(cands, stage["release_date"])
            upgrade_title(rec, base_title(grp[0]["title"]))
            old = by_id.pop(mid, None)
            if old:  # a Megabox-only record found its KMRB/KOBIS record: fold it in
                seen = min(seen, old["first_seen"].get("megabox", seen))
                fold_into(rec, old)
            stats["megabox_matched"] += 1
        else:
            rec = by_id.get(mid) or {"id": mid, "rcv_no": "", "title": base_title(grp[0]["title"]),
                                     "original_title": "", "applicant": "", "stages": {},
                                     "first_seen": {}, "tags": []}
            rec["tags"] = tag_film(rec["title"], "", "", "")
            by_id[mid] = rec
            stats["megabox_only"] += 1
        rec["first_seen"]["megabox"] = min(seen, rec["first_seen"].get("megabox", seen))
        old_stage = rec["stages"].get("megabox")
        if old_stage and "runs" not in old_stage:
            old_stage = {**old_stage, "first_seen": rec["first_seen"]["megabox"]}
        stage["runs"] = track_runs(old_stage, today, stage["release_date"])
        rec["stages"]["megabox"] = stage
        rec["last_seen"] = today
        assigned.add(rec["id"])

    for rec in by_id.values():
        st = rec["stages"].get("megabox")
        if st and rec["id"] not in assigned and st.get("status") != "ended":
            rec["stages"]["megabox"] = {**st, "status": "ended"}
            stats["megabox_ended"] += 1
    return _sort(by_id.values()), stats


def group_variants(films: list[dict]) -> dict:
    """Link the records of one film rated per screening format (in place).

    KMRB rates each format of a film separately and appends the format to the
    title ("하이브 치어링 파티 (2D)" / "(3D)", "인턴 (스크린엑스)"), often under
    another applicant (the format's own distributor). Records whose titles
    differ only by such a label are one film: the record with the plain title
    leads the group (if exactly one has it: two plain records mean two films
    of that title, which are left alone), else the first by format_rank. The
    lead gets `formats` [{"format", "id"}] (the per-format records, itself
    included); every other member gets `variant_of` (the lead's id). Recomputed
    every run."""
    plain: dict[str, list[dict]] = {}
    variants: dict[str, list[dict]] = {}
    for f in films:
        f.pop("formats", None)
        f.pop("variant_of", None)
        base, fmt = split_format(f.get("title", ""))
        (variants if fmt else plain).setdefault(norm_key(base), []).append(f)
    stats = {"format_groups": 0, "format_variants": 0, "format_ambiguous": 0}
    for key, members in sorted(variants.items()):
        heads = plain.get(key, [])
        if len(heads) > 1:
            stats["format_ambiguous"] += 1
            continue
        members = sorted(members, key=format_rank)
        lead = heads[0] if heads else members[0]
        if not heads and len(members) == 1:
            continue  # one format only: nothing to group
        lead["formats"] = [{"format": split_format(f["title"])[1], "id": f["id"]} for f in members]
        for f in members:
            if f is not lead:
                f["variant_of"] = lead["id"]
                stats["format_variants"] += 1
        stats["format_groups"] += 1
    return stats


def track_runs(old: dict | None, today: str, release_date: str) -> list[dict]:
    """Listing periods of a film on one chain: [{"from", "to", "release_date"}].
    A film listed again after it had ended starts a new period (a re-release);
    otherwise the open period is extended to today. A stage from before
    periods were kept starts with one period from its first sighting."""
    runs = [dict(r) for r in (old or {}).get("runs", [])]
    if old and not runs:
        runs = [{"from": old.get("first_seen", today), "to": today,
                 "release_date": old.get("release_date", "")}]
    if not runs or (old or {}).get("status") == "ended":
        runs.append({"from": today, "to": today, "release_date": release_date})
    else:
        runs[-1]["to"] = today
        runs[-1]["release_date"] = release_date or runs[-1].get("release_date", "")
    return runs


def apply_lotte(films: list[dict], rows: list[dict], today: str,
                kobis_cache: dict | None = None) -> tuple[list[dict], dict]:
    """Attach the Lotte Cinema list (pure, no I/O).

    A row joins the record of the KOBIS movie code Lotte lists for it (a
    KOBIS-only record or one whose KOBIS stage has that code), else a record
    whose normalised title matches (as for Megabox). Unmatched rows become
    Lotte-only records (id "l" + Lotte movie code), folded into a KMRB/KOBIS/
    Megabox record once one matches. Records that held a Lotte stage but are
    not in today's lists are kept with status "ended"; `runs` keeps every
    listing period (track_runs). Lotte-only records are published without the
    trusted-applicant rule, as Megabox-only ones are. `kobis_open_dt` is the
    first opening date KOBIS gives for Lotte's KOBIS code (from `kobis_cache`).
    """
    from .kobis import iso_date
    from .lotte import lotte_id  # local import, as in apply_kobis

    stats = {"lotte_matched": 0, "lotte_matched_kobis_cd": 0, "lotte_only": 0, "lotte_ended": 0}
    by_id = {f["id"]: f for f in films}
    index = TitleIndex(films, skip=lambda f: f["id"].startswith(("l", "c")),
                       wanted=[r["title"] for r in rows])
    by_cd: dict[str, list[dict]] = {}
    for f in films:
        cd = f["stages"].get("kobis", {}).get("movie_cd") or (f["id"][1:] if f["id"].startswith("k") else "")
        if cd and not f["id"].startswith("l"):
            by_cd.setdefault(cd, []).append(f)

    def pick(cands: list[dict], release: str) -> dict:
        exact = [f for f in cands if release and f["stages"].get("kobis", {}).get("open_dt") == release]
        return max(exact or cands, key=lambda f: (max(f["first_seen"].values(), default=""), f["id"]))

    assigned: set[str] = set()
    for r in sorted(rows, key=lambda r: r["movie_code"]):
        stage = {k: r.get(k, "") for k in ("movie_code", "status", "release_date", "release_month",
                                           "grade", "kobis_cd", "badge", "poster")}
        stage["exclusive"] = bool(r.get("exclusive"))
        if r.get("festival"):
            stage["festival"] = r["festival"]
        if r.get("genre"):
            stage["genre"] = r["genre"]
        stage["kobis_open_dt"] = iso_date((kobis_cache or {}).get(r["kobis_cd"], {}).get("openDt", ""))
        lid = lotte_id(r["movie_code"])
        seen = today
        cands = [f for f in by_cd.get(r["kobis_cd"], []) if f["id"] in by_id] if r["kobis_cd"] else []
        if cands:
            stats["lotte_matched_kobis_cd"] += 1
        else:
            cands = [f for f in index.find(r["title"]) if f["id"] in by_id]
        if cands:
            rec = pick(cands, r["release_date"])
            upgrade_title(rec, r["title"])
            old = by_id.pop(lid, None)
            if old:  # a Lotte-only record found its KMRB/KOBIS/Megabox record: fold it in
                seen = min(seen, old["first_seen"].get("lotte", seen))
                fold_into(rec, old)
            stats["lotte_matched"] += 1
        else:
            rec = by_id.get(lid) or {"id": lid, "rcv_no": "", "title": r["title"],
                                     "original_title": "", "applicant": "", "stages": {},
                                     "first_seen": {}, "tags": []}
            rec["tags"] = tag_film(rec["title"], "", "", "")
            by_id[lid] = rec
            stats["lotte_only"] += 1
        rec["first_seen"]["lotte"] = min(seen, rec["first_seen"].get("lotte", seen))
        old_stage = rec["stages"].get("lotte")
        if old_stage and "runs" not in old_stage:
            old_stage = {**old_stage, "first_seen": rec["first_seen"]["lotte"]}
        stage["runs"] = track_runs(old_stage, today, stage["release_date"])
        rec["stages"]["lotte"] = stage
        rec["last_seen"] = today
        assigned.add(rec["id"])

    for rec in by_id.values():
        st = rec["stages"].get("lotte")
        if st and rec["id"] not in assigned and st.get("status") != "ended":
            rec["stages"]["lotte"] = {**st, "status": "ended"}
            stats["lotte_ended"] += 1
    return _sort(by_id.values()), stats
