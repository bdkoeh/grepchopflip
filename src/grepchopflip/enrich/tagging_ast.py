"""Extra `tag` (D-041): AudioSet tags from AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, BSD-3-Clause weights)
via transformers, and a drum-class verdict that checks the drum part a hit was separated into (D-021).

AudioSet has labels for kick ("Bass drum"), snare ("Snare drum", "Rimshot", "Drum roll"), hi-hat and cymbal,
but none for toms, so toms (and every non-drum class) get tags only, no verdict. Ride and crash are one
family ("Cymbal"): AudioSet does not tell them apart.

Verdict: AST's absolute probabilities for isolated, separated hits are tiny (median "Bass drum" probability
for kicks on the test tracks: 0.004), so the verdict compares the families with each other: each family
takes the highest logit among its labels and a softmax over the four gives P(family | one of these drums).
The classifier's class is the most likely family and `class_confidence` the source family's probability. A
hit is flagged for review only when the classifier is clearly confident in another family: that family has
probability >= `DISAGREE_MIN_SHARE`, the source family <= `DISAGREE_MAX_SOURCE`, and its plain sigmoid
probability is at least `DISAGREE_MIN_SCORE` (some absolute evidence, not just a relative ranking).
Zero-padding to AST's 10 s window was compared with tiling short clips to 1, 2 and 4 s on the test tracks;
zero-padding agreed best (kick/snare 97-100 %).
"""

from __future__ import annotations

import importlib.util

import numpy as np

from grepchopflip.models import AST

SR = 16000
MAX_S = 10.0
HINT = "tags skipped: install the extra with `uv sync --extra tag` (or pip install grepchopflip[tag])"
FAMILIES: dict[str, tuple[str, ...]] = {
    "Kick": ("Bass drum",),
    "Snare": ("Snare drum", "Rimshot", "Drum roll"),
    "Hat": ("Hi-hat",),
    "Cymbal": ("Cymbal",),
}
CLASS_FAMILY = {"Kick": "Kick", "Snare": "Snare", "Hat": "Hat", "Ride": "Cymbal", "Crash": "Cymbal"}
DISAGREE_MIN_SHARE = 0.9
DISAGREE_MAX_SOURCE = 0.05
DISAGREE_MIN_SCORE = 0.05
METAL = frozenset({"Hat", "Cymbal"})
TOP_K = 5
TAG_MIN_SCORE = 0.1
REASON_PREFIX = "classifier="


def available() -> bool:
    return importlib.util.find_spec("transformers") is not None


class Tagger:
    def __init__(self, device: str = "cuda", batch: int = 32):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        from grepchopflip.models import ensure_hf

        d = ensure_hf(AST)
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.fe = ASTFeatureExtractor.from_pretrained(str(d))
        self.model = ASTForAudioClassification.from_pretrained(str(d)).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        self.batch = batch

    def probs(self, clips: list[np.ndarray]) -> np.ndarray:
        """Sigmoid probabilities (n, 527) for mono 16 kHz clips (first 10 s)."""
        import torch

        out = []
        for i in range(0, len(clips), self.batch):
            chunk = [np.asarray(c[: int(MAX_S * SR)], dtype=np.float32) for c in clips[i:i + self.batch]]
            feats = self.fe(chunk, sampling_rate=SR, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(input_values=feats["input_values"].to(self.device)).logits
            out.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)


def summarize(p: np.ndarray, labels: list[str]) -> dict:
    """What is cached per clip: family scores and the top AudioSet tags."""
    idx = {lab: i for i, lab in enumerate(labels)}
    pc = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    logit = np.log(pc / (1 - pc))
    fam, fam_logit = {}, {}
    for f, labs in FAMILIES.items():
        ii = [idx[lab] for lab in labs if lab in idx]
        fam[f] = round(float(max(p[i] for i in ii)), 5)
        fam_logit[f] = round(float(max(logit[i] for i in ii)), 4)
    top = [(labels[i], float(p[i])) for i in np.argsort(-p)[:TOP_K] if p[i] >= TAG_MIN_SCORE]
    return {"families": fam, "logits": fam_logit, "tags": [[lab, round(s, 4)] for lab, s in top]}


def verdict(cls: str, summary: dict) -> dict | None:
    """None for classes without an AudioSet family; else the classifier's class, confidence and agreement."""
    src = CLASS_FAMILY.get(cls)
    if src is None:
        return None
    names = list(FAMILIES)
    lg = np.array([summary["logits"][f] for f in names])
    sm = np.exp(lg - lg.max())
    sm /= sm.sum()
    share = dict(zip(names, (float(x) for x in sm)))
    best = names[int(np.argmax(sm))]
    disagrees = (best != src and share[best] >= DISAGREE_MIN_SHARE and share[src] <= DISAGREE_MAX_SOURCE
                 and summary["families"][best] >= DISAGREE_MIN_SCORE)
    return {"class": best, "confidence": round(share[src], 3), "best_share": round(share[best], 3),
            "score": summary["families"][best], "agrees": best == src,
            "agrees_metal": best == src or {best, src} == METAL, "disagrees": disagrees}


def reason(v: dict) -> str:
    return f"{REASON_PREFIX}{v['class']}({v['best_share']:.2f})"
