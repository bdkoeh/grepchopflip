import numpy as np
import pytest
import soundfile as sf

SR = 44100


def synth_song(seconds: float = 8.0, sr: int = SR) -> np.ndarray:
    """Kick+snare pattern (sine / noise bursts), bass line and chord pad. Stereo float32."""
    rng = np.random.default_rng(7)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    mix = np.zeros(n)
    beat = 0.5  # 120 BPM
    for k in range(int(seconds / beat)):
        i = int(k * beat * sr)
        if k % 2 == 0:  # kick on 1 and 3: pitch-dropping sine
            m = int(0.25 * sr)
            tt = np.arange(m) / sr
            f = 50 + 100 * np.exp(-tt * 30)
            mix[i:i + m] += (0.9 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 12))[: n - i]
        else:  # snare on 2 and 4: noise burst + tone
            m = int(0.2 * sr)
            tt = np.arange(m) / sr
            burst = (rng.standard_normal(m) * 0.5 + 0.4 * np.sin(2 * np.pi * 190 * tt)) * np.exp(-tt * 20)
            mix[i:i + m] += burst[: n - i]
        # closed hat on every beat
        m = int(0.05 * sr)
        tt = np.arange(m) / sr
        mix[i:i + m] += (rng.standard_normal(m) * 0.15 * np.exp(-tt * 80))[: n - i]
    # bass line, one note per bar of 4 beats
    for bar, f in enumerate([55.0, 55.0, 65.4, 49.0]):
        a, b = int(bar * 2.0 * sr), int((bar + 1) * 2.0 * sr)
        if a >= n:
            break
        seg = t[a:b]
        mix[a:b] += 0.35 * np.sin(2 * np.pi * f * seg) * (0.8 + 0.2 * np.sin(2 * np.pi * 2 * seg))
    # chord pad (A minor)
    for f in (220.0, 261.63, 329.63):
        mix += 0.08 * np.sin(2 * np.pi * f * t)
    mix = mix / max(1.0, np.abs(mix).max() / 0.8)
    left = mix
    right = np.roll(mix, 12) * 0.95
    return np.stack([left, right], axis=1).astype(np.float32)


@pytest.fixture(scope="session")
def synth_wav(tmp_path_factory):
    p = tmp_path_factory.mktemp("synth") / "synth song.wav"
    sf.write(str(p), synth_song(), SR, subtype="PCM_24")
    return p
