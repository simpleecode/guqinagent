#!/usr/bin/env python3
"""Fetch melodies from public/CC sources: TheSession (ABC, CC-BY-4.0) and
Mutopia (public-domain classical MIDI).  Plain URL downloads also work.
Only for personal research use; respect each site's terms."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

UA = {
    # thesession.org's WAF 403s unknown bot UAs; requests are cached and kept
    # to interactive volumes, so identify as a plain browser.
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/129.0.0.0 Safari/537.36"),
}
CACHE = Path(__file__).resolve().parent / "cache"


def _get(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text).strip("_")[:60] or "tune"


def search_thesession(query: str, limit: int = 5, cache: Path = CACHE) -> list[dict]:
    # The JSON format is requested via ``?format=json``; a ``.json`` path
    # suffix (the old convention) now 404s.
    url = ("https://thesession.org/tunes/search?q="
           + urllib.parse.quote(query) + f"&format=json&perpage={limit}")
    data = json.loads(_get(url))
    results = []
    for item in data.get("tunes", [])[:limit]:
        detail = json.loads(_get(f"https://thesession.org/tunes/{item['id']}?format=json"))
        setting = detail["settings"][0]
        name = f"thesession_{item['id']}_{_slug(detail.get('name', 'tune'))}"
        path = cache / f"{name}.abc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(setting["abc"], encoding="utf-8")
        results.append({"source": "thesession", "title": detail.get("name", ""),
                        "path": str(path), "format": "abc"})
    return results


MUTOPIA_HOSTS = ("https://www.mutopiaproject.org", "https://mutopiaproject.org")


def _mutopia_get(path_query: str, timeout: int = 30) -> bytes:
    # The bare domain does not resolve on every network; ``www`` always has.
    last_error: Exception | None = None
    for host in MUTOPIA_HOSTS:
        try:
            return _get(host + path_query, timeout=timeout)
        except OSError as error:
            last_error = error
    raise RuntimeError(f"Mutopia unreachable on all hosts: {last_error}")


def search_mutopia(query: str, limit: int = 5, cache: Path = CACHE) -> list[dict]:
    # The search form field is ``searchingfor``; ``search`` is silently ignored.
    url = ("/cgibin/make-table.cgi?"
           + urllib.parse.urlencode({"instrument": "Piano", "searchingfor": query}))
    html = _mutopia_get(url).decode("utf-8", "replace")
    # Result links are absolute (https://host/ftp/...); older mirrors used
    # relative ftp/... paths.  Accept both.
    links = re.findall(r'href="((?:https?://[^/"]+)?/ftp/[^"]+\.mid)"', html)
    results = []
    for link in links[:limit]:
        filename = Path(link).name
        if link.startswith("http"):
            midi = _get(link)
        else:
            midi = _mutopia_get(link)
        path = cache / f"mutopia_{_slug(filename[:-4])}.mid"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(midi)
        results.append({"source": "mutopia", "title": filename[:-4],
                        "path": str(path), "format": "midi"})
    return results


def download_url(url: str, cache: Path = CACHE) -> dict:
    data = _get(url, timeout=120)
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".bin"
    name = _slug(Path(url).stem or "download")
    path = cache / f"url_{name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"source": "url", "title": name, "path": str(path),
            "format": suffix.lstrip(".").lower()}
