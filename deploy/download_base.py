"""Download the ModelScope base mirror without proxies, verifying every SHA-256."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/Qwen3.5-4B-Base"
DEST.mkdir(parents=True, exist_ok=True)

def download(entry):
    name = entry["Path"]
    if entry["Type"] != "blob" or name.startswith("."):
        return
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-4B-Base/repo"
           f"?Revision={entry['Revision']}&FilePath={quote(name)}")
    partial = target.with_name(target.name + ".partial")
    if not target.exists():
        print(f"Downloading {name} ({entry['Size']} bytes)", flush=True)
        env = {k: v for k, v in os.environ.items() if k.lower() not in
               {"http_proxy", "https_proxy", "all_proxy"}}
        subprocess.run(["aria2c", "--all-proxy=", "--max-connection-per-server=16",
                        "--split=16", "--min-split-size=4M", "--continue=true",
                        "--file-allocation=none", "--summary-interval=30",
                        "--max-tries=5", "--retry-wait=3", "--auto-file-renaming=false",
                        "--dir=" + str(partial.parent), "--out=" + partial.name,
                        url], check=True, env=env)
        partial.rename(target)
    with target.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != entry["Sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {name}")
    print(f"Verified {name}", flush=True)

if __name__ == "__main__":
    entries = json.loads((ROOT / "deploy/base-model-files.json").read_text())["Data"]["Files"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, entries))
