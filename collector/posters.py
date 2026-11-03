"""Poster images, stored as small WebP files in data/posters/.

Sources, in order of preference (sizes checked 2026-11-01):
  1. KOBIS: the image the distributor registered, from the detail page
     (main poster link) or the 개봉스케줄 calendar (its 192 px thumbnail; the
     original, about 1200 px wide and 1-1.6 MB, is the same path without
     "thumb_x192/thn_"). The thumb_x640 variant is only 349 px wide.
  2. Megabox list: imgPathNm, a 640 px ".large.jpg".
  3. Lotte list: PosterURL, 200 px.
  4. CGV list (from cgv.json): a 640 px JPEG on cdn.cgv.co.kr (served to
     a plain HTTP client on 2026-11-01). Last, so it is only fetched for films
     no other source has a poster for.
Each image is fetched once per source URL, scaled down to at most
WIDTH px wide and saved as WebP (quality: Pillow's default 80); on
2026-11-01 that gave 36-62 KB per poster. A record gets `poster`, the file's
path in the repository. Files no record uses any more are deleted. A source
that fails is retried on a later day; the next source is tried meanwhile,
and while every source fails a record keeps the file it had.

Per-format records of a grouped film (variant_of) share their lead's poster.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from .http import BlockedError

# 2x a 240 CSS px card; KOBIS thumbnails (192 px) are too small for that
WIDTH = 480
QUALITY = 80
KOBIS_THUMB = "thumb_x192/thn_"
REPO_DIR = "data/posters"  # rec["poster"] is the file's path in the repository


def candidates(rec: dict) -> list[str]:
    st = rec["stages"]
    thumb = st.get("schedule", {}).get("poster", "")
    urls = [st.get("kobis_detail", {}).get("poster", ""),
            thumb.replace(KOBIS_THUMB, "") if KOBIS_THUMB in thumb else thumb,
            st.get("megabox", {}).get("poster", ""), st.get("lotte", {}).get("poster", ""),
            st.get("cgv", {}).get("poster", "")]
    return list(dict.fromkeys(u for u in urls if u))


def file_name(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".webp"


def to_webp(data: bytes) -> tuple[bytes, tuple[int, int]]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        if im.width > WIDTH:
            im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        im.save(out, "WEBP", quality=QUALITY)
        return out.getvalue(), im.size


def fetch(client, films: list[dict], cache: dict, poster_dir: Path, stamp: str) -> dict:
    """Make sure every record with a poster source has a file. Updates `cache`
    (source URL -> {"file", "size", "bytes", "fetched"} or {"error", "checked"})
    and the files in place; sets rec["poster"]. A block stops the source."""
    poster_dir.mkdir(parents=True, exist_ok=True)
    fetched = errors = 0
    by_id = {f["id"]: f for f in films}
    for rec in films:
        if rec.get("variant_of"):
            continue
        prev, failed = rec.pop("poster", None), False
        for url in candidates(rec):
            entry = cache.get(url, {})
            if entry.get("file") and (poster_dir / entry["file"]).exists():
                rec["poster"] = f"{REPO_DIR}/{entry['file']}"
                break
            if entry.get("error") and entry.get("checked", "") >= stamp:
                failed = True
                continue  # failed today: next source, retried another day
            try:
                data, size = to_webp(client.get(url).content)
            except BlockedError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad image must not fail the source
                cache[url] = {"error": type(exc).__name__, "checked": stamp}
                errors += 1
                failed = True
                continue
            name = file_name(url)
            (poster_dir / name).write_bytes(data)
            cache[url] = {"file": name, "size": list(size), "bytes": len(data), "fetched": stamp}
            rec["poster"] = f"{REPO_DIR}/{name}"
            fetched += 1
            break
        if "poster" not in rec and failed and prev and (poster_dir / Path(prev).name).exists():
            rec["poster"] = prev  # a replacement failed: the previous image stays until one arrives
    for rec in films:  # per-format records show their lead's poster
        lead = by_id.get(rec.get("variant_of", ""))
        if lead is not None:
            if lead.get("poster"):
                rec["poster"] = lead["poster"]
            else:
                rec.pop("poster", None)
    used = {Path(f["poster"]).name for f in films if f.get("poster")}
    removed = 0
    for path in poster_dir.glob("*.webp"):
        if path.name not in used:
            path.unlink()
            removed += 1
    wanted = {u for f in films if not f.get("variant_of") for u in candidates(f)}
    for url in [u for u, e in cache.items()
                if (e.get("file") and e["file"] not in used) or (e.get("error") and u not in wanted)]:
        del cache[url]
    return {"with_poster": sum(bool(f.get("poster")) for f in films), "fetched": fetched,
            "errors": errors, "removed": removed, "files": len(used),
            "bytes": sum(e.get("bytes", 0) for e in cache.values())}
