"""Build <team>_submission.zip in the layout the challenge requires, then verify it.

    python tools/make_submission_zip.py [--team "NaN Sense"]

Layout (entry names use forward slashes so the archive extracts correctly on Linux/macOS;
Windows PowerShell 5.1's Compress-Archive stores backslashes, which breaks that):
    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/{src/*.py, README.md, requirements.txt}
    Documentation_template.md

Verification: CRC-checks the whole archive, confirms no backslash names, then extracts the two TSVs
and runs the official validate_submission.py on those extracted copies, i.e. on exactly what is in the zip.
The zip is written to a .partial file and renamed only when complete, so an interrupted run never
leaves a truncated zip behind.
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def entries():
    out = [("output/matching_results.tsv", os.path.join(ROOT, "output", "matching_results.tsv")),
           ("output/candidate_pairs.tsv", os.path.join(ROOT, "output", "candidate_pairs.tsv"))]
    src = os.path.join(ROOT, "src")
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            out.append((f"code/business_entity_resolution/src/{name}", os.path.join(src, name)))
    for name in ("README.md", "requirements.txt"):
        out.append((f"code/business_entity_resolution/{name}", os.path.join(ROOT, name)))
    out.append(("Documentation_template.md", os.path.join(ROOT, "Documentation_template.md")))
    missing = [s for _, s in out if not os.path.exists(s)]
    if missing:
        sys.exit(f"missing files: {missing}")
    return out


def build(path):
    tmp = path + ".partial"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for arc, src in entries():
            z.write(src, arc)
    os.replace(tmp, path)
    print(f"built {path} ({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)


def stored_names(path):
    """Entry names exactly as stored (Python's zipfile normalises backslashes on Windows, hiding them)."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 65557))
        tail = f.read()
        i = tail.rfind(b"PK\x05\x06")
        _, _, _, _, _, cdsize, cdoff, _ = struct.unpack("<4sHHHHIIH", tail[i:i + 22])
        if cdoff == 0xFFFFFFFF:
            return None  # zip64: cannot parse here
        f.seek(cdoff)
        cd = f.read(cdsize)
    pos, names = 0, []
    while pos < len(cd):
        fnlen, extralen, cmtlen = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        names.append(cd[pos + 46:pos + 46 + fnlen].decode("utf-8", "replace"))
        pos += 46 + fnlen + extralen + cmtlen
    return names


def verify(path):
    ok = True
    names = stored_names(path)
    if names is None:
        print("name check skipped (zip64)")
    else:
        bad = [n for n in names if "\\" in n]
        print(f"entries: {len(names)} | names with backslash: {len(bad)}")
        ok &= not bad
    with zipfile.ZipFile(path) as z:
        corrupt = z.testzip()
        print(f"CRC check of every entry: {'OK' if corrupt is None else 'CORRUPT: ' + corrupt}", flush=True)
        ok &= corrupt is None
        want = {arc for arc, _ in entries()}
        absent = want - set(z.namelist())
        print(f"required entries missing: {sorted(absent) if absent else 'none'}")
        ok &= not absent
        d = os.path.join(C.WORK_DIR, "_zipcheck")
        shutil.rmtree(d, ignore_errors=True)
        z.extract("output/matching_results.tsv", d)
        z.extract("output/candidate_pairs.tsv", d)
    print("running the official validator on the copies extracted from the zip ...", flush=True)
    r = subprocess.run(
        [sys.executable, C.VALIDATOR,
         "--matching", os.path.join(d, "output", "matching_results.tsv"),
         "--candidate", os.path.join(d, "output", "candidate_pairs.tsv"),
         "--test-dir", os.path.join(C.DATA_DIR, "test")],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout, r.stderr)
    shutil.rmtree(d, ignore_errors=True)
    ok &= r.returncode == 0
    print("RESULT:", "ZIP VERIFIED" if ok else "PROBLEMS FOUND")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="NaN Sense")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    zpath = os.path.join(ROOT, f"{a.team}_submission.zip")
    if not a.verify_only:
        build(zpath)
    sys.exit(0 if verify(zpath) else 1)
