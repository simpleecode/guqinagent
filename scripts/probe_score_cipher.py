#!/usr/bin/env python3
import argparse
import base64
import gzip
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path


def read_java_char_array(path: Path) -> str:
    data = path.read_bytes()
    # Java serialization for char[]: ac ed 00 05 75 72 ... xp <len:u32> <utf16be chars>
    marker = b"\x78\x70"
    idx = data.find(marker)
    if idx < 0 or idx + 6 > len(data):
        raise ValueError("not a simple Java serialized char[]")
    n = struct.unpack(">I", data[idx + 2:idx + 6])[0]
    raw = data[idx + 6:idx + 6 + n * 2]
    return raw.decode("utf-16-be")


def score_blob(buf: bytes) -> dict:
    out = {
        "len": len(buf),
        "head_hex": buf[:32].hex(" "),
        "printable": sum(32 <= b < 127 or b in (9, 10, 13) for b in buf[:512]),
    }
    for name, fn in [
        ("gzip", gzip.decompress),
        ("zlib", zlib.decompress),
        ("zlib_raw", lambda b: zlib.decompress(b, -15)),
    ]:
        try:
            dec = fn(buf)
            out[name] = {
                "ok": True,
                "len": len(dec),
                "head": dec[:80].decode("utf-8", errors="replace"),
                "has_score_fields": any(x in dec for x in [b"notes", b"jians", b"sections", b"lyric", b"dataops"]),
            }
        except Exception as e:
            out[name] = {"ok": False, "err": str(e)[:80]}
    try:
        txt = buf.decode("utf-8")
        out["utf8"] = {"ok": True, "head": txt[:200], "json": txt.lstrip().startswith(("{", "["))}
    except Exception as e:
        out["utf8"] = {"ok": False, "err": str(e)[:80]}
    return out


def try_crypto(data: bytes, key_text: str):
    static_texts = [
        key_text,
        "SlToLhsXjd46JFS7",
        "Zp3XjhgNyWoqogSh",
        "SlToLhsXjd46JFS7Zp3XjhgNyWoqogSh",
        "Zp3XjhgNyWoqogShSlToLhsXjd46JFS7",
        key_text + "SlToLhsXjd46JFS7",
        key_text + "Zp3XjhgNyWoqogSh",
        "SlToLhsXjd46JFS7" + key_text,
        "Zp3XjhgNyWoqogSh" + key_text,
    ]
    keys = []
    for text in static_texts:
        kb = text.encode("utf-8")
        prefix = text[:12]
        keys.append((f"{prefix}:raw16", kb[:16]))
        keys.append((f"{prefix}:raw24", kb[:24]))
        keys.append((f"{prefix}:raw32", kb[:32]))
        keys.append((f"{prefix}:md5", hashlib.md5(kb).digest()))
        keys.append((f"{prefix}:sha256", hashlib.sha256(kb).digest()))
        keys.append((f"{prefix}:sha1_16", hashlib.sha1(kb).digest()[:16]))
        keys.append((f"{prefix}:sha512_32", hashlib.sha512(kb).digest()[:32]))
    key_bytes = key_text.encode("utf-8")
    ivs = [
        ("zero", b"\x00" * 16),
        ("head16", data[:16]),
        ("tail16", data[-16:]),
        ("dk_key16", key_bytes[:16]),
        ("dk_md5", hashlib.md5(key_bytes).digest()),
        ("static1", b"SlToLhsXjd46JFS7"),
        ("static2", b"Zp3XjhgNyWoqogSh"),
    ]
    prefixes = [
        ("all", data),
        ("skip32", data[32:]),
        ("skip24", data[24:]),
        ("skip16", data[16:]),
        ("skip12", data[12:]),
        ("skip8", data[8:]),
        ("skip4", data[4:]),
    ]

    results = []
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except Exception as e:
        return [{"error": f"cryptography unavailable: {e}"}]

    def decrypt_aes(key: bytes, mode_name: str, iv, body: bytes) -> bytes:
        if mode_name == "ECB":
            mode = modes.ECB()
        elif mode_name == "CBC":
            mode = modes.CBC(iv)
        elif mode_name == "CTR":
            mode = modes.CTR(iv)
        else:
            raise ValueError(mode_name)
        decryptor = Cipher(algorithms.AES(key), mode).decryptor()
        return decryptor.update(body) + decryptor.finalize()

    def pkcs7_unpad(blob: bytes) -> bytes:
        if not blob:
            raise ValueError("empty")
        n = blob[-1]
        if n < 1 or n > 16 or blob[-n:] != bytes([n]) * n:
            raise ValueError("bad padding")
        return blob[:-n]

    def try_decompress_offsets(blob: bytes):
        for needle, name in [(b"\x1f\x8b", "gzip"), (b"\x78\x01", "zlib_7801"), (b"\x78\x9c", "zlib_789c"), (b"\x78\xda", "zlib_78da")]:
            start = 0
            while True:
                idx = blob.find(needle, start)
                if idx < 0:
                    break
                yield f"from_{name}_at_{idx}", blob[idx:]
                start = idx + 1

    for pk, body in prefixes:
        for kn, key in keys:
            if len(key) not in (16, 24, 32):
                continue
            for mode in ["ECB", "CBC", "CTR"]:
                if mode == "ECB" and len(body) % 16:
                    continue
                if mode == "ECB":
                    tries = [("none", None)]
                elif mode == "CBC":
                    tries = ivs
                    if len(body) % 16:
                        continue
                else:
                    tries = [("nonce0", b""), ("head8", data[:8]), ("head16nonce8", data[:8])]
                for ivn, iv in tries:
                    try:
                        if mode == "ECB":
                            dec = decrypt_aes(key, mode, None, body)
                        elif mode == "CBC":
                            dec = decrypt_aes(key, mode, iv, body)
                        else:
                            dec = decrypt_aes(key, mode, iv, body)
                        for variant, blob in [("raw", dec)]:
                            results.append({"alg": f"AES-{mode}", "prefix": pk, "key": kn, "iv": ivn, "variant": variant, "score": score_blob(blob)})
                            for dname, dblob in try_decompress_offsets(blob):
                                results.append({"alg": f"AES-{mode}", "prefix": pk, "key": kn, "iv": ivn, "variant": dname, "score": score_blob(dblob)})
                        try:
                            up = pkcs7_unpad(dec)
                            results.append({"alg": f"AES-{mode}", "prefix": pk, "key": kn, "iv": ivn, "variant": "pkcs7", "score": score_blob(up)})
                            for dname, dblob in try_decompress_offsets(up):
                                results.append({"alg": f"AES-{mode}", "prefix": pk, "key": kn, "iv": ivn, "variant": "pkcs7_" + dname, "score": score_blob(dblob)})
                        except Exception:
                            pass
                    except Exception:
                        pass
            if len(key) == 32:
                for nonce_name, nonce in [("head16", data[:16]), ("zero16", b"\0" * 16), ("dk16", key_bytes[:16])]:
                    try:
                        decryptor = Cipher(algorithms.ChaCha20(key, nonce), mode=None).decryptor()
                        dec = decryptor.update(body) + decryptor.finalize()
                        results.append({"alg": "ChaCha20", "prefix": pk, "key": kn, "iv": nonce_name, "variant": "raw", "score": score_blob(dec)})
                    except Exception:
                        pass
    return results


def rank(item):
    s = item.get("score", {})
    bonus = 0
    for name in ["gzip", "zlib", "zlib_raw"]:
        val = s.get(name, {})
        if val.get("ok"):
            bonus += 100
            if val.get("has_score_fields"):
                bonus += 1000
    utf = s.get("utf8", {})
    if utf.get("ok"):
        bonus += 20 + utf.get("head", "").count("{") * 2
        if utf.get("json"):
            bonus += 200
    bonus += s.get("printable", 0)
    return bonus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", type=Path, required=True)
    ap.add_argument("--dk", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = args.response.read_bytes()
    key = read_java_char_array(args.dk)
    results = try_crypto(data, key)
    ranked = sorted(results, key=rank, reverse=True)
    report = {
        "response": str(args.response),
        "response_len": len(data),
        "response_sha256": hashlib.sha256(data).hexdigest(),
        "dk": str(args.dk),
        "dk_key": key,
        "top": ranked[:50],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"key": key, "out": str(args.out), "top_scores": [rank(x) for x in ranked[:10]]}, ensure_ascii=False, indent=2))
    for item in ranked[:10]:
        if "error" in item:
            print(json.dumps(item, ensure_ascii=False))
            continue
        print(json.dumps({k: item[k] for k in ["alg", "prefix", "key", "iv", "variant"]}, ensure_ascii=False), item["score"].get("head_hex"), item["score"].get("utf8"))


if __name__ == "__main__":
    main()
