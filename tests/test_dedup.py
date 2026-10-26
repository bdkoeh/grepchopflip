"""Duplicate grouping on duplicated fixtures."""

import numpy as np

from grepchopflip.enrich import dedup as dd

SR = 44100


def _hit(seed, f0=None, seconds=0.3, tau=0.05, amp=0.5):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * f0 * t) if f0 else rng.standard_normal(n)
    return amp * y * np.exp(-t / tau)


def _item(i, y, cls="Kick", kind="oneshot", review=False, rms=-20.0, start=None):
    return {"i": i, "class": cls, "kind": kind, "needs_review": review, "rms_dbfs": rms,
            "source_start_s": float(i) if start is None else start, "duration_s": len(y) / SR,
            "fp": dd.fingerprint(y, SR, kind)}


def test_identical_hits_group_and_different_hits_do_not():
    kick = _hit(1, f0=55)
    near = kick * 0.7 + np.random.default_rng(9).standard_normal(len(kick)) * 1e-4   # quieter copy, tiny noise
    snare_like = _hit(2)                                                              # noise burst
    items = [_item(0, kick, rms=-18), _item(1, near, rms=-21), _item(2, snare_like, rms=-15)]
    g = dd.group(items)
    assert g[0][0] == g[1][0] != g[2][0]
    assert g[0] == ("Kick-001", True, 2) and g[1] == ("Kick-001", False, 2)   # loudest is the representative
    assert g[2][1] is True and g[2][2] == 1


def test_distance_is_level_independent_and_zero_for_identical():
    y = _hit(3, f0=110)
    F = np.stack([dd.fingerprint(y, SR, "oneshot"), dd.fingerprint(0.1 * y, SR, "oneshot")])
    assert np.allclose(dd.distances(F), 0, atol=1e-3)


def test_never_groups_across_review_or_class():
    y = _hit(4, f0=60)
    items = [_item(0, y), _item(1, y, review=True), _item(2, y, cls="Tom")]
    g = dd.group(items)
    assert len({g[i][0] for i in range(3)}) == 3 and all(g[i][1] for i in range(3))


def test_leader_clustering_does_not_chain():
    """a~b and b~c but a!~c: c must not join a's group through b."""
    base = _hit(5, f0=80)
    fps = [dd.fingerprint(base, SR, "oneshot")]
    for shift in (4.0, 8.0):                       # fingerprints offset by 4 and 8 dB above the floor
        f = fps[0].copy()
        act = f > dd.FLOOR_DB + 1e-3
        f[act] = np.minimum(f[act] - shift, 0)
        fps.append(f)
    items = [{"i": i, "class": "Kick", "kind": "oneshot", "needs_review": False, "rms_dbfs": -10.0 - i,
              "source_start_s": float(i), "duration_s": 0.3, "fp": f} for i, f in enumerate(fps)]
    g = dd.group(items)
    assert g[0][0] == g[1][0] and g[2][0] != g[0][0]


def test_loops_need_similar_duration():
    y = _hit(6, f0=100, seconds=2.0, tau=1.0)
    items = [_item(0, y, cls="DrumLoop", kind="loop"), _item(1, y[: int(1.5 * SR)], cls="DrumLoop", kind="loop")]
    g = dd.group(items)
    assert g[0][0] != g[1][0]
    g2 = dd.group([_item(0, y, cls="DrumLoop", kind="loop"), _item(1, y.copy(), cls="DrumLoop", kind="loop")])
    assert g2[0][0] == g2[1][0]


def test_group_ids_unique_and_numbered_in_song_order():
    items = [_item(i, _hit(10 + i, f0=50 + 40 * i), start=10.0 - i) for i in range(3)]
    g = dd.group(items)
    ids = [g[i][0] for i in range(3)]
    assert sorted(ids) == ["Kick-001", "Kick-002", "Kick-003"] and g[2][0] == "Kick-001"   # earliest first


def test_short_hit_differs_from_full_hit():
    full = _hit(7, f0=200, seconds=0.35, tau=0.3)
    short = full[: int(0.08 * SR)]
    F = np.stack([dd.fingerprint(full, SR, "oneshot"), dd.fingerprint(short, SR, "oneshot")])
    assert dd.distances(F)[0, 1] > dd.threshold_for("Kick")
