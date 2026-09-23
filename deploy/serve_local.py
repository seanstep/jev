"""Serve the verified local ModelScope base with the official Kev adapter."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kev.checkpoint import Checkpoint, LoadOptions
from kev.serve import Server, app
import torch
import uvicorn

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the selected GPU")
    checkpoint = Checkpoint(str(ROOT / "models/kev-4b"))
    checkpoint.meta.base = str(ROOT / "models/Qwen3.5-4B-Base")
    checkpoint.meta.base_revision = None
    tokenizer, model = checkpoint.load("cuda", LoadOptions.from_env())
    torch.cuda.empty_cache()  # Release temporary FP32 allocations used to merge LoRA.
    app.state.server = Server(checkpoint, tokenizer, model, "cuda")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("KEV_PORT", "8009")))
