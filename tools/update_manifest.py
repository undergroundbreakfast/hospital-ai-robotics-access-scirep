#!/usr/bin/env python3
"""Hash candidate release files, excluding Git-ignored local data and this manifest."""
import hashlib
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
lines = []
for name in sorted(set(paths)):
    path = ROOT / name
    if not name or name == "FILE_MANIFEST_SHA256.txt" or not path.is_file():
        continue
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{name}")
(ROOT / "FILE_MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n")
print(f"Hashed {len(lines)} release files.")
