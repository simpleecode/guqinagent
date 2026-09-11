#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run(cmd, check=True, capture=True):
    p = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace",
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed {p.returncode}: {' '.join(cmd)}\n{p.stdout or ''}")
    return p.stdout or ""


def adb(args, *parts, check=True, capture=True):
    return run([args.adb, *parts], check=check, capture=capture)


def parse_bounds(raw):
    nums = [int(x) for x in re.findall(r"\d+", raw or "")]
    if len(nums) != 4:
        return None
    x1, y1, x2, y2 = nums
    return x1, y1, x2, y2


def center(bounds):
    x1, y1, x2, y2 = bounds
    return (x1 + x2) // 2, (y1 + y2) // 2


def dump_xml(args, out_dir, name):
    adb(args, "shell", "uiautomator", "dump", "/sdcard/window.xml", check=False)
    out = out_dir / f"{name}.xml"
    run([args.adb, "pull", "/sdcard/window.xml", str(out)], check=True)
    return out


def nodes(xml_path):
    root = ET.parse(xml_path).getroot()
    for node in root.iter("node"):
        text = node.attrib.get("text", "")
        desc = node.attrib.get("content-desc", "")
        package = node.attrib.get("package", "")
        bounds = parse_bounds(node.attrib.get("bounds", ""))
        if bounds:
            yield node, f"{text}\n{desc}", package, bounds


def click_contains(args, out_dir, needle, step, prefer_clickable=True):
    xml_path = dump_xml(args, out_dir, step)
    matches = []
    for node, haystack, package, bounds in nodes(xml_path):
        if needle in haystack:
            clickable = node.attrib.get("clickable") == "true"
            matches.append((0 if clickable else 1, bounds, package, haystack.strip()))
    if not matches:
        raise RuntimeError(f"cannot find {needle!r} in {xml_path}")
    matches.sort(key=lambda item: (item[0] if prefer_clickable else 0, item[1][1], item[1][0]))
    _, bounds, package, label = matches[0]
    x, y = center(bounds)
    print(f"[click] {needle!r} -> ({x},{y}) package={package} label={label[:80]!r}")
    adb(args, "shell", "input", "tap", str(x), str(y), capture=False)
    time.sleep(args.delay)
    return bounds


def foreground(args):
    out = adb(args, "shell", "dumpsys", "window", check=False)
    for line in out.splitlines():
        if "mCurrentFocus=" in line:
            return line.strip()
    return ""


def xml_text(xml_path):
    return xml_path.read_text(encoding="utf-8", errors="replace")


def require_markers(xml_path, markers, label):
    text = xml_text(xml_path)
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(f"{label} missing markers {missing!r} in {xml_path}")
    return text


def ensure_app_foreground(args, label):
    focus = foreground(args)
    if args.package not in focus:
        raise RuntimeError(f"{label} is not in app foreground: {focus}")
    return focus


def open_entry(args):
    if args.deeplink:
        adb(args, "shell", "am", "force-stop", args.package, capture=False)
        time.sleep(0.5)
        adb(args, "shell", "am", "start", "-W", "-a", "android.intent.action.VIEW",
            "-d", args.deeplink, args.package, capture=False)
    else:
        adb(args, "shell", "am", "start", "-W", "-a", "android.intent.action.MAIN",
            "-c", "android.intent.category.LAUNCHER", "-n", args.activity, capture=False)
    time.sleep(args.delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adb", default=r"C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe")
    ap.add_argument("--package", default="com.sitongli.app.gadget")
    ap.add_argument("--activity", default="com.sitongli.app.gadget/com.sitongli.app.MainActivity")
    ap.add_argument("--score-title", default="秋风词")
    ap.add_argument("--deeplink", default=None,
                    help="Open a score detail page directly, e.g. sitongli://app/scores/161749.")
    ap.add_argument("--out-dir", default="cases/sitongli-guanshanyue/batch/SWDFmDZX/evidence/adb_nav")
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--scrolls", type=int, default=0)
    ap.add_argument("--no-open-score", action="store_true",
                    help="Stop on the score detail page before tapping the score body entry.")
    ap.add_argument("--swipe", default="720,2100,720,1500,450",
                    help="Swipe x1,y1,x2,y2,duration_ms for score body scrolling.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    open_entry(args)
    ensure_app_foreground(args, "entry")
    if args.deeplink:
        detail_xml = dump_xml(args, out_dir, "01_detail_deeplink")
        require_markers(detail_xml, [args.score_title, "乐谱"], "deeplink detail")
    else:
        click_contains(args, out_dir, "我的", "01_home")
        click_contains(args, out_dir, "我的乐谱", "02_my")
        click_contains(args, out_dir, args.score_title, "03_scores")
    if args.no_open_score:
        dump_xml(args, out_dir, "04_detail")
        print("[focus]", foreground(args))
        return
    click_contains(args, out_dir, "乐谱", "04_detail")
    after_xml = dump_xml(args, out_dir, "05_after_score_click")
    focus = ensure_app_foreground(args, "after score click")
    require_markers(after_xml, ["正调定弦"], "score body")
    print("[focus]", focus)

    swipe = [part.strip() for part in args.swipe.split(",")]
    if len(swipe) != 5 or not all(part.lstrip("-").isdigit() for part in swipe):
        raise ValueError("--swipe must be x1,y1,x2,y2,duration_ms")

    for i in range(args.scrolls):
        adb(args, "shell", "input", "swipe", *swipe, capture=False)
        time.sleep(args.delay)
        dump_xml(args, out_dir, f"scroll_{i:02d}")
        print("[scroll]", i, ensure_app_foreground(args, f"scroll {i}"))


if __name__ == "__main__":
    main()
