"""Polite HTTP client.

Rules: requests are sequential (one client, no concurrency), at least
MIN_INTERVAL seconds between two requests to the same host, and any 403/429
stops the whole run (BlockedError) instead of retrying harder.

A connect timeout (TCP or TLS handshake never completed, so no request
reached the site) is retried up to CONNECT_RETRIES times. Measured
2026-11-01 from four GitHub Actions runners: TCP connects to
www.megabox.co.kr took ~0.3 s, but the first HTTPS request from three of
them stalled until a 15 s connect timeout, the second succeeded in 6.1-8.0 s
and the third in 0.27 s; with the old single 40 s attempt the collector lost
Megabox in two runs in a row.
"""
from __future__ import annotations

import time
from urllib.parse import urlsplit

import httpx

MIN_INTERVAL = 0.5  # seconds between requests to one host (project politeness rule)
USER_AGENT = "Mozilla/5.0 (compatible; movie-radar/0.1; public release tracker)"
# measured 2026-11-01 (see above): retries succeeded within 8.0 s, the stall
# lasts until the timeout; 15 s leaves margin over 8.0 s
CONNECT_TIMEOUT = 15.0
CONNECT_RETRIES = 2  # the third attempt succeeded on every measured runner


class FetchError(RuntimeError):
    """A request failed (network error or unexpected status)."""


class BlockedError(FetchError):
    """The site answered 403 or 429: stop, do not retry."""


class PoliteClient:
    def __init__(self, min_interval: float = MIN_INTERVAL, timeout: float = 40.0,
                 client: httpx.Client | None = None, sleep=time.sleep, clock=time.monotonic):
        self._client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT)),
            follow_redirects=True)
        self._min_interval = min_interval
        self._last: dict[str, float] = {}
        self._sleep = sleep
        self._clock = clock
        self.request_count = 0
        self.connect_retries = 0

    def _wait(self, url: str) -> None:
        host = urlsplit(url).hostname or ""
        last = self._last.get(host)
        if last is not None:
            delay = self._min_interval - (self._clock() - last)
            if delay > 0:
                self._sleep(delay)
        self._last[host] = self._clock()

    def request(self, method: str, url: str, **kw) -> httpx.Response:
        for attempt in range(CONNECT_RETRIES + 1):
            self._wait(url)
            self.request_count += 1
            try:
                resp = self._client.request(method, url, **kw)
                break
            except httpx.ConnectTimeout as exc:
                if attempt == CONNECT_RETRIES:
                    raise FetchError(f"{method} {url}: {exc!r} ({attempt + 1} attempts)") from exc
                self.connect_retries += 1
            except httpx.HTTPError as exc:
                raise FetchError(f"{method} {url}: {exc!r}") from exc
            finally:
                # measure the gap from the end of this request, not its start
                host = urlsplit(url).hostname or ""
                self._last[host] = self._clock()
        if resp.status_code in (403, 429):
            raise BlockedError(f"{method} {url}: HTTP {resp.status_code} - stopping")
        if resp.status_code != 200:
            raise FetchError(f"{method} {url}: HTTP {resp.status_code}")
        return resp

    def get(self, url: str, **kw) -> httpx.Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw) -> httpx.Response:
        return self.request("POST", url, **kw)

    def close(self) -> None:
        self._client.close()


# --- block detection --------------------------------------------------------
# A site that blocks us may answer 200 with a notice page instead of 403/429.
# "보안 정책에 위배" (security-policy violation notice) counts as a block anywhere
# in the body. "차단" (blocked) is a common word that can appear in a film
# title, so it only counts when the page is also missing its expected markup.
BLOCK_PHRASES_ALWAYS = ("보안 정책에 위배",)
BLOCK_PHRASES_IF_NO_MARKUP = ("차단",)


def check_block(text: str, markup_ok: bool) -> None:
    """Raise BlockedError if the body looks like a block notice."""
    for phrase in BLOCK_PHRASES_ALWAYS:
        if phrase in text:
            raise BlockedError(f"block notice in body ({phrase})")
    if not markup_ok:
        for phrase in BLOCK_PHRASES_IF_NO_MARKUP:
            if phrase in text:
                raise BlockedError(f"block notice in body ({phrase}), expected markup missing")
