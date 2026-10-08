from pathlib import Path
from mitmproxy import http

OUT = Path(__file__).resolve().parents[1] / "evidence" / "mitm_flows"
OUT.mkdir(parents=True, exist_ok=True)


def response(flow: http.HTTPFlow) -> None:
    host = flow.request.pretty_host or ""
    if "sitongli.net" not in host:
        return
    idx = len(list(OUT.glob("*.txt"))) + 1
    base = OUT / f"{idx:04d}_{host}_{flow.request.method}_{flow.response.status_code}"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base.name)[:180]
    stem = OUT / safe
    req = [
        f"{flow.request.method} {flow.request.pretty_url}",
        "REQUEST_HEADERS",
        str(flow.request.headers),
        "",
        "REQUEST_BODY",
        flow.request.get_text(strict=False) or "",
    ]
    resp = [
        f"HTTP {flow.response.status_code}",
        "RESPONSE_HEADERS",
        str(flow.response.headers),
        "",
        "RESPONSE_BODY",
        flow.response.get_text(strict=False) or "",
    ]
    stem.with_suffix(".txt").write_text("\n".join(req + ["", "---", ""] + resp), encoding="utf-8", errors="replace")
    stem.with_suffix(".request.bin").write_bytes(flow.request.raw_content or b"")
    stem.with_suffix(".response.bin").write_bytes(flow.response.raw_content or b"")
