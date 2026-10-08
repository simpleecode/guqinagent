from pathlib import Path
import base64
import gzip
import hashlib
import lzma
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
ct = base64.b64decode((ROOT / "evidence/flutter_profile.b64").read_text().strip())
dk = b"1xkt78395mwuc1d05wp49094ye2n14gd1e2ng927"
OUT = ROOT / "work/profile_decrypt_candidates"
OUT.mkdir(parents=True, exist_ok=True)


raws = [
    dk,
    dk[:16],
    dk[:24],
    dk[:32],
    dk[-16:],
    dk[-24:],
    dk[-32:],
    hashlib.md5(dk).digest(),
    hashlib.sha1(dk).digest(),
    hashlib.sha256(dk).digest(),
    hashlib.sha512(dk).digest()[:32],
]
keys = []
for raw in raws:
    for n in (16, 24, 32):
        if len(raw) >= n:
            keys.append((f"raw{len(raw)}_{n}", raw[:n]))
for name, func in [
    ("md5", hashlib.md5),
    ("sha1", hashlib.sha1),
    ("sha256", hashlib.sha256),
    ("sha512", hashlib.sha512),
]:
    d = func(dk).digest()
    for n in (16, 24, 32):
        if len(d) >= n:
            keys.append((f"{name}_{n}", d[:n]))

seen = set()
keys2 = []
for name, key in keys:
    if key not in seen:
        seen.add(key)
        keys2.append((name, key))


def decode_layers(pt):
    outs = [pt]
    for fn in (
        lambda b: zlib.decompress(b),
        lambda b: zlib.decompress(b, -15),
        gzip.decompress,
        lzma.decompress,
    ):
        try:
            outs.append(fn(pt))
        except Exception:
            pass
    return outs


def looks_interesting(pt):
    for blob in decode_layers(pt):
        if not blob:
            continue
        head = blob[:512]
        if blob[:1] in (b"{", b"["):
            return True
        needles = [b"token", b"uid", b"nick", b"profile", b"Bearer", b"phone", b"avatar"]
        if any(x in blob for x in needles):
            return True
        printable = sum(32 <= c < 127 or c in (9, 10, 13) or c >= 0x80 for c in head)
        if len(head) and printable / len(head) > 0.85 and b"\x00" not in head:
            return True
    return False


def save_hit(label, pt):
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)[:160]
    for i, blob in enumerate(decode_layers(pt)):
        (OUT / f"{safe}.{i}.bin").write_bytes(blob)
        try:
            (OUT / f"{safe}.{i}.txt").write_text(blob.decode("utf-8"), encoding="utf-8")
        except Exception:
            pass
    print("HIT", label, "preview=", decode_layers(pt)[-1][:240])


try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except Exception as e:
    print("cryptography import failed", e)
    sys.exit(1)

layouts = []
for nn in (12, 16, 24):
    if len(ct) > nn + 16:
        layouts.append((f"prefix_nonce{nn}_tag_suffix16", ct[:nn], ct[nn:]))
for nn in (12, 16, 24):
    if len(ct) > nn + 16:
        layouts.append((f"suffix_nonce{nn}_tag_suffix16", ct[-nn:], ct[:-nn]))

for lname, nonce, data in layouts:
    for kname, key in keys2:
        if len(nonce) == 12:
            try:
                pt = AESGCM(key).decrypt(nonce, data, None)
                if looks_interesting(pt):
                    save_hit("AESGCM_" + lname + "_" + kname, pt)
                    sys.exit(0)
            except Exception:
                pass
        if len(key) == 32 and len(nonce) == 12:
            try:
                pt = ChaCha20Poly1305(key).decrypt(nonce, data, None)
                if looks_interesting(pt):
                    save_hit("CHACHA_" + lname + "_" + kname, pt)
                    sys.exit(0)
            except Exception:
                pass

ivs = [
    ("zero", b"\x00" * 16),
    ("prefix16", ct[:16]),
    ("suffix16", ct[-16:]),
    ("md5dk", hashlib.md5(dk).digest()),
    ("sha256dk16", hashlib.sha256(dk).digest()[:16]),
]


def unpad_variants(pt):
    outs = [pt]
    if pt:
        n = pt[-1]
        if 1 <= n <= 16 and pt.endswith(bytes([n]) * n):
            outs.append(pt[:-n])
    return outs


for kname, key in keys2:
    for ivname, iv in ivs:
        for mode_name, mode_factory in [
            ("CBC_all", lambda v: modes.CBC(v)),
            ("CBC_skip16", lambda v: modes.CBC(v)),
            ("CFB_all", lambda v: modes.CFB(v)),
            ("CFB_skip16", lambda v: modes.CFB(v)),
            ("OFB_all", lambda v: modes.OFB(v)),
            ("OFB_skip16", lambda v: modes.OFB(v)),
            ("CTR_all", lambda v: modes.CTR(v)),
            ("CTR_skip16", lambda v: modes.CTR(v)),
        ]:
            data = ct[16:] if mode_name.endswith("skip16") else ct
            if len(data) % 16 and mode_name.startswith("CBC"):
                continue
            try:
                dec = Cipher(algorithms.AES(key), mode_factory(iv)).decryptor()
                pt = dec.update(data) + dec.finalize()
                for upt in unpad_variants(pt):
                    if looks_interesting(upt):
                        save_hit(f"AES_{mode_name}_{kname}_{ivname}", upt)
                        sys.exit(0)
            except Exception:
                pass


def rc4(key, data):
    s = list(range(256))
    j = 0
    out = bytearray()
    key = bytearray(key)
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) & 255
        s[i], s[j] = s[j], s[i]
    i = j = 0
    for b in data:
        i = (i + 1) & 255
        j = (j + s[i]) & 255
        s[i], s[j] = s[j], s[i]
        out.append(b ^ s[(s[i] + s[j]) & 255])
    return bytes(out)


stream_keys = [(n, k) for n, k in keys2] + [
    ("dk_full", dk),
    ("sha256hex", hashlib.sha256(dk).hexdigest().encode()),
    ("md5hex", hashlib.md5(dk).hexdigest().encode()),
]
for kname, key in stream_keys:
    for data_name, data in [("all", ct), ("skip16", ct[16:]), ("skip12", ct[12:]), ("skip8", ct[8:])]:
        try:
            pt = rc4(key, data)
            if looks_interesting(pt):
                save_hit(f"RC4_{kname}_{data_name}", pt)
                sys.exit(0)
        except Exception:
            pass
        pt = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
        if looks_interesting(pt):
            save_hit(f"XOR_{kname}_{data_name}", pt)
            sys.exit(0)

print("no decrypt hit", len(keys2), "aes keys", len(layouts), "aead layouts", "candidates_dir=", OUT)
