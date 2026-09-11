#!/usr/bin/env python3
"""Upload only the public exported SFT bundle through the local SSH helper."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import paramiko


def credentials(helper: Path) -> dict[str, str]:
    tree = ast.parse(helper.read_text(encoding="utf-8"))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"HOST", "USER", "PASSWORD"}:
                values[target.id] = str(node.value.value)
    if set(values) != {"HOST", "USER", "PASSWORD"}:
        raise RuntimeError("SSH helper does not define HOST/USER/PASSWORD constants")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--helper", type=Path, default=Path(r"D:\new_cip\ops\.ssh_run.py"))
    args = parser.parse_args()
    files = [path for path in args.input_dir.iterdir() if path.is_file()]
    if not files or any("audit" in path.name for path in files):
        raise RuntimeError("input must be a non-empty public SFT directory without audit files")
    auth = credentials(args.helper)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(auth["HOST"], username=auth["USER"], password=auth["PASSWORD"], timeout=30)
    try:
        command = "mkdir -p " + repr(args.remote_dir)
        _, stdout, stderr = client.exec_command(command)
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(stderr.read().decode("utf-8", "replace"))
        sftp = client.open_sftp()
        try:
            for path in files:
                sftp.put(str(path), args.remote_dir.rstrip("/") + "/" + path.name)
                print(path.name, path.stat().st_size, flush=True)
        finally:
            sftp.close()
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
