"""Extra `embed` (D-041): 512-dim CLAP audio embeddings (`laion/larger_clap_music`, Apache-2.0 weights) via
transformers `ClapModel`, stored beside the index in `embeddings/clap.npy` (float32, one L2-normalised row
per sample in index order) with `embeddings/clap.json` naming the model and the audio digest of each row.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from grepchopflip.models import CLAP

SR = 48000
MAX_S = 10.0
DIM = 512
FILE = "embeddings/clap.npy"
META = "embeddings/clap.json"
HINT = "embeddings skipped: install the extra with `uv sync --extra embed` (or pip install grepchopflip[embed])"


def available() -> bool:
    return importlib.util.find_spec("transformers") is not None


class Embedder:
    def __init__(self, device: str = "cuda", batch: int = 16):
        import torch
        from transformers import ClapFeatureExtractor, ClapModel

        from grepchopflip.models import ensure_hf

        d = ensure_hf(CLAP)
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.fe = ClapFeatureExtractor.from_pretrained(str(d))
        self.model = ClapModel.from_pretrained(str(d)).to(self.device).eval()
        self.batch = batch

    def embed(self, clips: list[np.ndarray]) -> np.ndarray:
        """(n, 512) L2-normalised embeddings of mono 48 kHz clips (first 10 s; no random crop)."""
        import torch

        out = []
        for i in range(0, len(clips), self.batch):
            chunk = [np.asarray(c[: int(MAX_S * SR)], dtype=np.float32) for c in clips[i:i + self.batch]]
            feats = self.fe(chunk, sampling_rate=SR, return_tensors="pt")
            feats = {k: v.to(self.device) for k, v in feats.items() if hasattr(v, "to")}
            with torch.inference_mode():
                e = self.model.get_audio_features(**feats)
            if not isinstance(e, torch.Tensor):     # newer transformers return a model output
                e = getattr(e, "pooler_output", None)
                if e is None:
                    raise RuntimeError("ClapModel.get_audio_features returned no embedding")
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            out.append(e.cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, DIM), dtype=np.float32)
