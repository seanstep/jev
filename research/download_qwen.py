"""Download and verify a small Qwen base from ModelScope without proxies."""
import concurrent.futures, hashlib, json, os, subprocess
from pathlib import Path
from urllib.parse import quote
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen2.5-0.5B"
DEST = ROOT / "models/Qwen2.5-0.5B"
DEST.mkdir(parents=True, exist_ok=True)

def download(f):
    name = f["Path"]
    if f["Type"] != "blob" or name.startswith("."):
        return
    target = DEST / name
    if target.exists():
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        if digest == f["Sha256"]:
            return
        target.unlink()
    url = f"https://modelscope.cn/api/v1/models/{REPO}/repo?Revision={f['Revision']}&FilePath={quote(name)}"
    env = {k: v for k, v in os.environ.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    if f["Size"] > 50_000_000:
        subprocess.run(["aria2c", "--all-proxy=", "--max-connection-per-server=16", "--split=16", "--min-split-size=4M", "--file-allocation=none", "--max-tries=5", "--retry-wait=3", "--dir=" + str(DEST), "--out=" + name, url], check=True, env=env)
    else:
        subprocess.run(["curl", "--noproxy", "*", "-fLsS", "--retry", "4", url, "-o", str(target)], check=True, env=env)
    digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    if digest != f["Sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {name}")
    print("verified", name, flush=True)

if __name__ == "__main__":
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.load(op.open(f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?Recursive=true", timeout=30))
    (ROOT / "research/qwen-files.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, data["Data"]["Files"]))
