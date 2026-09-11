#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request


def request_json(url, token=None):
    headers = {
        "Accept": "application/json",
        "User-Agent": "Sitongli/2.3.0 export helper",
    }
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", "replace")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw": body}
        return e.code, parsed


def main():
    ap = argparse.ArgumentParser(description="Export a Sitongli score by score_key through the app API.")
    ap.add_argument("--base-url", default="https://s.sitongli.net")
    ap.add_argument("--score-key", required=True, help="Value observed in the app as score_key/key.")
    ap.add_argument("--token", default=None, help="Login token if the score is private.")
    ap.add_argument("--outdir", default="../output")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    url = args.base_url.rstrip("/") + "/scores/" + args.score_key
    status, data = request_json(url, args.token)
    raw_path = outdir / "raw_data.json"
    raw_path.write_text(json.dumps({"url": url, "status": status, "raw_data": data}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GET {url} -> HTTP {status}")
    print(f"wrote {raw_path}")
    if status >= 400:
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)

    tmp_capture = outdir / "_http_capture.jsonl"
    tmp_capture.write_text(json.dumps({"frida_message": {"payload": {"text": json.dumps(data, ensure_ascii=False)}}}, ensure_ascii=False) + "\n", encoding="utf-8")
    script = pathlib.Path(__file__).with_name("normalize_score_export.py")
    subprocess.check_call([sys.executable, str(script), "--capture", str(tmp_capture), "--outdir", str(outdir)])


if __name__ == "__main__":
    main()
