"""A tiny synthetic song folder for the organize/index tests: fake chops.json files plus short WAVs, no models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from grepchopflip.chop.common import Chop, write_chops_json

SR = 44100
SHA = "ab" * 32


def wav(root: Path, rel: str, seconds: float, channels: int = 1, seed: int = 0, amp: float = 0.5) -> None:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = (rng.standard_normal((n, channels)) * amp * 0.3).astype(np.float32)
    y *= np.linspace(1.0, 0.1, n, dtype=np.float32)[:, None]
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(root / rel), y, SR, subtype="PCM_24")


def chop(root: Path, cid: str, stem: str, kind: str, rel: str, start: float, *, part=None, review=False,
         seconds=0.25, channels=1, seed=0, **kw) -> Chop:
    wav(root, rel, seconds, channels, seed)
    extra = {"review_reasons": ["poor_attack"] if review else []}
    extra.update(kw.pop("extra", {}))
    return Chop(id=cid, stem=stem, part=part, kind=kind, source_start_s=start, source_end_s=start + seconds,
                path=rel, channels=channels, sample_rate=SR, needs_review=review, bit_depth=24, bar=1,
                source_file=f"stems/{stem}.wav", source_sha256=SHA, extra=extra, **kw)


def make_song(root: Path, with_words: bool = True, drop: tuple[str, ...] = ()) -> Path:
    """Chops for every class. Positions are chosen so a review hit (kick_b, 2.0 s) sorts before a kept one."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.json").write_text(json.dumps({
        "path": "/music/My_Song.wav", "name": "My_Song.wav", "sha256": SHA, "duration_s": 30.0,
        "sample_rate": 44100, "channels": 2, "frames": 1323000, "format": "wav"}), encoding="utf-8")
    beats = [round(i * 60 / 92, 4) for i in range(40)]
    (root / "analysis.json").write_text(json.dumps({
        "bpm": 92.0, "bpm_confidence": 0.9, "tempo_reliable": True, "grid_source": "tracked",
        "grid_consistency": 0.95, "beats_s": beats, "downbeats_s": beats[::4], "beats_per_bar": 4,
        "key": "A", "mode": "minor", "key_reliable": True, "key_confidence": 0.4,
        "models": {"beat_tracker": {"name": "beat_this"}}, "gcf_version": "0.1.0.dev0"}), encoding="utf-8")
    (root / "stems").mkdir(exist_ok=True)
    (root / "chops").mkdir(exist_ok=True)
    (root / "chops" / "grid.json").write_text(json.dumps({"downbeats_s_corrected": beats[::4]}), encoding="utf-8")
    p = {"fingerprint": "f"}

    d = root / "chops" / "drums"
    drums = [
        chop(root, "drums_oneshot_0001", "drums", "oneshot", "chops/drums/oneshots/kick/kick_0001.wav", 5.0, part="kick", seed=1),
        chop(root, "drums_oneshot_0002", "drums", "oneshot", "chops/drums/review/kick/kick_0002.wav", 2.0, part="kick", review=True, seed=2),
        chop(root, "drums_oneshot_0003", "drums", "oneshot", "chops/drums/oneshots/snare/snare_0001.wav", 3.0, part="snare", seed=3),
        chop(root, "drums_oneshot_0004", "drums", "oneshot", "chops/drums/oneshots/toms/toms_0001.wav", 3.5, part="toms", seed=4),
        chop(root, "drums_oneshot_0005", "drums", "oneshot", "chops/drums/oneshots/hh/hh_0001.wav", 4.0, part="hh", seed=5, channels=2),
        chop(root, "drums_oneshot_0006", "drums", "oneshot", "chops/drums/oneshots/ride/ride_0001.wav", 4.5, part="ride", seed=6),
        chop(root, "drums_oneshot_0007", "drums", "oneshot", "chops/drums/review/crash/crash_0001.wav", 6.0, part="crash", review=True, seed=7),
        chop(root, "drums_loop_0001", "drums", "loop", "chops/drums/loops/drumloop_2bar_001.wav", 1.0, seed=8, bars=2, seconds=0.5, channels=2),
    ]
    write_chops_json(d / "chops.json", [c for c in drums if c.id not in drop], p, "drums")

    bass = [
        chop(root, "bass_0001", "bass", "oneshot", "chops/bass/oneshots/bass_0001.wav", 2.0, seed=10, root_midi=29, root_note="F1", root_confidence=0.8),
        chop(root, "bass_0002", "bass", "oneshot", "chops/bass/oneshots/bass_0002.wav", 3.0, seed=11, root_midi=30, root_note="F#1", root_confidence=0.3),
        chop(root, "bassloop_1bar_001", "bass", "loop", "chops/bass/loops/bassloop_1bar_001.wav", 0.5, seed=12, bars=1, seconds=0.5, channels=2),
    ]
    write_chops_json(root / "chops" / "bass" / "chops.json", [c for c in bass if c.id not in drop], p, "chop.bass")

    mel = [
        chop(root, "melloop_1bar_001", "other", "loop", "chops/melodic/loops/melloop_1bar_001.wav", 0.5, seed=20, bars=1, seconds=0.5, channels=2, review=True),
        chop(root, "phrase_0001", "other", "phrase", "chops/melodic/phrases/phrase_0001.wav", 1.0, seed=21, channels=2,
             extra={"key": "A minor", "key_reliable": True}),
        chop(root, "phrase_0002", "other", "phrase", "chops/melodic/phrases/phrase_0002.wav", 4.0, seed=22, channels=2,
             extra={"key": "C major", "key_reliable": False}),
    ]
    write_chops_json(root / "chops" / "melodic" / "chops.json", [c for c in mel if c.id not in drop], p, "chop.melodic")

    vox = [chop(root, "vox_0001", "vocals", "phrase", "chops/vocals/phrases/vox_0001.wav", 2.0, seed=30,
                root_midi=71, root_note="B4", root_confidence=0.9)]
    write_chops_json(root / "chops" / "vocals" / "chops.json", [c for c in vox if c.id not in drop], p, "chop.vocals")
    if with_words:
        words = [chop(root, "word_0001", "vocals", "word", "chops/vocals/words/word_0001.wav", 2.1, seed=31, review=True)]
        write_chops_json(root / "chops" / "vocals" / "chops_words.json", words, p, "chop.words")
    for c in [*drums, *bass, *mel, *vox]:       # a re-chop that dropped a hit leaves no file behind
        if c.id in drop:
            (root / c.path).unlink(missing_ok=True)
    return root


def make_split_song(root: Path, split: str = "vocals", many_loops: int = 2) -> Path:
    """Slice 13: a song folder of a non-default split. Every combined group gets loops (1, 2 and 4 bars) and
    phrases; isolated vocals a phrase; source.json and stems/.done.json record the split."""
    from grepchopflip.chop.melodic import group_tag
    from grepchopflip.split import parse_split

    sp = parse_split(split)
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.json").write_text(json.dumps({
        "path": "/music/My_Song.wav", "name": "My_Song.wav", "sha256": SHA, "duration_s": 30.0,
        "sample_rate": 44100, "channels": 2, "frames": 1323000, "format": "wav", **sp.record()}), encoding="utf-8")
    beat = 60 / 92
    beats = [round(i * beat, 4) for i in range(40)]
    (root / "analysis.json").write_text(json.dumps({
        "bpm": 92.0, "bpm_confidence": 0.9, "tempo_reliable": True, "grid_source": "tracked",
        "grid_consistency": 0.95, "beats_s": beats, "downbeats_s": beats[::4], "beats_per_bar": 4,
        "key": "A", "mode": "minor", "key_reliable": True, "key_confidence": 0.4,
        "models": {"beat_tracker": {"name": "beat_this"}}, "gcf_version": "0.1.0.dev0"}), encoding="utf-8")
    (root / "stems").mkdir(exist_ok=True)
    (root / "stems" / ".done.json").write_text(json.dumps({"params": {"split": sp.spec}, **sp.record()}))
    for n in sp.names:
        wav(root, f"stems/{n}.wav", 0.1)
    (root / "chops").mkdir(exist_ok=True)
    p = {"fingerprint": "f"}
    bar = 4 * beat
    seed = 100
    for g in sp.combined:
        t = group_tag(g)
        cs = []
        for bars in (1, 2, 4):
            for k in range(many_loops):
                seed += 1
                cs.append(chop(root, f"{t}loop_{bars}bar_{k + 1:03d}", g, "loop",
                               f"chops/{g}/loops/{t}loop_{bars}bar_{k + 1:03d}.wav", round(k * bar * bars, 4),
                               seed=seed, bars=bars, seconds=round(bars * bar, 4), channels=2,
                               review=(bars == 2 and k == 0)))
        cs.append(chop(root, f"{t}phrase_0001", g, "phrase", f"chops/{g}/phrases/{t}phrase_0001.wav", 0.0,
                       seed=seed + 1, seconds=round(2 * bar, 4), channels=2, length_beats=8.0,
                       extra={"key": "A minor", "key_reliable": True}))
        write_chops_json(root / "chops" / g / "chops.json", cs, p, f"chop.group.{g}")
    if sp.isolated("vocals"):
        vox = [chop(root, "vox_0001", "vocals", "phrase", "chops/vocals/phrases/vox_0001.wav", 0.0, seed=30,
                    seconds=round(2 * bar, 4), length_beats=8.0)]
        write_chops_json(root / "chops" / "vocals" / "chops.json", vox, p, "chop.vocals")
    return root


def tree(root: Path) -> list[str]:
    return sorted(q.relative_to(root).as_posix() for q in root.rglob("*.wav") if "stems" not in q.parts)
