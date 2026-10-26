# Analysis & Chopping Components — Reference (v1 research, 2026-10-24)

Scope: open-source building blocks for the automated sample chopper (onsets, beats/downbeats, pitch/key, loudness, I/O/DSP, tagging, dedup). Target: Windows 11, Python 3.12, RTX 5090 (Blackwell, sm_120 -> needs a CUDA 12.8+ PyTorch build: https://discuss.pytorch.org/t/support-for-nvidia-rtx-5090-cuda-sm-120/223683), project license MIT. Non-permissive or risky licenses are marked **FLAG**.

License legend: OK = permissive (MIT/BSD/ISC/Apache/CC0); **FLAG-LGPL** = usable as a dynamically linked dependency but not vendorable; **FLAG-GPL/AGPL/NC** = avoid as a dependency of an MIT project.

---

## 1. librosa (onsets, beats, tempo, chroma, silence, spectral features)

- **Install / version**: `pip install librosa`. PyPI latest is **1.0.0 (2026-08-11), requires Python >= 3.12**, ISC license (https://pypi.org/project/librosa/). 1.0 is "intended to stabilize the API from the long-running 0.x series" and expired all 0.x deprecations (https://github.com/librosa/librosa/releases/tag/1.0.0). 0.11.0 shipped 2025-03-11 (https://github.com/librosa/librosa/releases). **Decision**: pin `librosa>=1.0,<2` for Python 3.12; the APIs below are the keyword-only forms present in both 0.11 and 1.0 (source: `main` branch docstrings).
- `librosa.load(path, sr=22050, res_type="soxr_hq")` — default downsample to 22.05 kHz; pass `sr=None` to keep native rate (https://raw.githubusercontent.com/librosa/librosa/main/librosa/core/audio.py). `res_type` choices: `soxr_vhq/hq/mq/lq/qq`, `kaiser_best/fast` (resampy), `fft`, `polyphase`, `sinc_*`, `linear`, `zero_order_hold` (same file).

### 1.1 Onset detection (transient one-shots)
```python
onset_detect(*, y=None, sr=22050, onset_envelope=None, hop_length=512,
             backtrack=False, energy=None, units="frames", normalize=True,
             sparse=True, **peak_pick_kwargs)
```
Source: https://raw.githubusercontent.com/librosa/librosa/main/librosa/onset.py
- `backtrack=True` moves each onset back to the nearest preceding local minimum of `energy` (default: the onset envelope; pass `librosa.feature.rms(y=y)[0]` for amplitude-based backtracking). Requires `sparse=True`. This is what you want for slice *start* points so the attack is not clipped.
- `units`: `"frames" | "samples" | "time"`. `normalize=True` scales the envelope to [0,1] before peak picking so `delta` is relative.
- Default `peak_pick` kwargs (in frames, derived from sr/hop): `pre_max=0.03*sr//hop` (30 ms), `post_max=0.00*sr//hop+1`, `pre_avg=0.10*sr//hop` (100 ms), `post_avg=0.10*sr//hop+1`, `wait=0.03*sr//hop` (30 ms), `delta=0.07`.
- `peak_pick` rule (https://raw.githubusercontent.com/librosa/librosa/main/librosa/util/utils.py): n is a peak iff `x[n] == max(x[n-pre_max : n+post_max])` **and** `x[n] >= mean(x[n-pre_avg : n+post_avg]) + delta` **and** `n - previous_peak > wait`. Tuning: raise `delta` (0.1-0.3) to drop ghost hits; raise `wait` to enforce a minimum slice length (e.g. 50-100 ms for drums).
- `onset_backtrack(events, energy)` is exposed separately; searches leftward only, returns frame 0 if no minimum exists.
```python
onset_strength(*, y=None, sr=22050, S=None, lag=1, max_size=1, ref=None,
               detrend=False, center=True, feature=None, aggregate=None, **kw)
```
Spectral flux: mean over bins of the positive difference between log-mel frame t and reference at t-lag. `max_size>1` applies a local-max filter over frequency (SuperFlux-style vibrato suppression); `feature` defaults to `melspectrogram(fmax=sr/2)`; `aggregate` defaults to `np.mean` (use `np.median` for robustness). Underlying `n_fft=2048, hop_length=512` (same source). For sharper drum onsets use `hop_length=256` or lower and `n_fft=1024`.

### 1.2 Beats / tempo
```python
tempo, beats = librosa.beat.beat_track(*, y=None, sr=22050, onset_envelope=None,
    hop_length=512, start_bpm=120.0, tightness=100, trim=True, bpm=None,
    prior=None, units="frames", sparse=True)
```
Ellis (2007) dynamic-programming tracker: onset strength -> tempo via autocorrelation -> DP peak picking constrained to tempo. Returns `(tempo_ndarray, beats)`; tempo is an ndarray even for mono. Silent input -> 0 BPM and empty beats (https://raw.githubusercontent.com/librosa/librosa/main/librosa/beat.py). `prior` accepts a scipy distribution over BPM; `bpm` fixes tempo (use beat_this BPM and let librosa only phase-align, or skip librosa beats entirely). No downbeats. `librosa.beat.plp(..., tempo_min=30, tempo_max=300)` gives a predominant-local-pulse curve for variable-tempo material (same source).
- Accuracy: Ellis-DP is adequate for steady electronic material but has no downbeat output and is prone to octave errors; prefer a neural tracker (sec. 3) for bar-aligned loops.

### 1.3 Silence / splitting
```python
librosa.effects.split(y, *, top_db=60, ref=np.max, frame_length=2048, hop_length=512, aggregate=np.max) -> (m,2) sample intervals
librosa.effects.trim(y, *, top_db=60, ref=np.max, ...) -> (y_trimmed, [start,end])
```
Frames more than `top_db` below `ref` are silence; with default `ref=np.max` a uniformly silent file is *not* trimmed — pass an absolute `ref` (e.g. `ref=1.0` with `top_db=50`) for fixed-threshold gating (https://raw.githubusercontent.com/librosa/librosa/main/librosa/effects.py). `hpss(y, kernel_size=31, margin=1.0)` returns `(harmonic, percussive)` — useful pre-step before chroma/key on the "other" stem (same source).

### 1.4 Zero-crossings, time-stretch
- `librosa.zero_crossings(y, *, threshold=1e-10, ref_magnitude=None, pad=True, zero_pos=True, axis=-1)` -> boolean indicator array (https://raw.githubusercontent.com/librosa/librosa/main/librosa/core/audio.py). Use `np.flatnonzero(...)` and snap each cut to the nearest crossing.
- `effects.time_stretch(y, *, rate, n_fft=2048, ...)` phase vocoder (`rate>1` faster); `effects.pitch_shift(y, *, sr, n_steps, bins_per_octave=12, res_type="soxr_hq")` = stretch + resample (https://raw.githubusercontent.com/librosa/librosa/main/librosa/effects.py). Phase-vocoder quality is mediocre on transients (phasiness); see sec. 6 for python-stretch.

### 1.5 Spectral features for kick/snare/hat/tonal heuristics
All return `(..., 1, t)` unless noted; defaults `n_fft=2048, hop_length=512` (https://raw.githubusercontent.com/librosa/librosa/main/librosa/feature/spectral.py):
- `spectral_centroid` (brightness: kick low, hat high), `spectral_bandwidth(p=2)`, `spectral_flatness(amin=1e-10, power=2.0)` (near 1 = noise-like/hat/snare body, near 0 = tonal), `spectral_rolloff(roll_percent=0.85)`, `rms(frame_length=2048)`, `zero_crossing_rate`, `spectral_contrast(n_bands=6)` -> `(n_bands+1, t)`, `mfcc(n_mfcc=20)` -> `(20, t)`.
- Chroma: `chroma_stft(n_chroma=12, tuning=None)` (auto tuning estimate), `chroma_cqt(bins_per_octave=36, n_octaves=7)`, `chroma_cens` (smoothed, `win_len_smooth=41`), `tonnetz` -> `(6, t)`.
- `librosa.estimate_tuning(y=..., sr=..., bins_per_octave=12)` -> deviation in [-0.5, 0.5) bins; feed into `chroma_cqt(tuning=...)` (https://raw.githubusercontent.com/librosa/librosa/main/librosa/core/pitch.py).

### 1.6 Key estimation (no built-in in librosa)
Krumhansl-Kessler profiles: major `[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]`, minor `[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]`; algorithm = mean chroma over time, Pearson-correlate against 24 rotations, argmax (profiles and procedure as implemented in partitura: https://partitura.readthedocs.io/en/latest/_modules/partitura/musicanalysis/key_identification.html; Temperley / Kostka-Payne alternatives listed there). About 15 lines of numpy on top of `chroma_cqt` of the harmonic (HPSS) component. Reference-only alternatives: Essentia `KeyExtractor` (**FLAG-AGPL**, no Windows Python bindings: https://essentia.upf.edu/installing.html, https://essentia.upf.edu/licensing_information.html); libkeyfinder/keyfinder-cli (**FLAG-GPL-3**: https://git-old.alpinelinux.org/aports/tree/community/keyfinder-cli/APKBUILD).

---

## 2. allin1 (All-In-One Music Structure Analyzer) — BPM, beats, downbeats, segments

- **Repo/license**: https://github.com/mir-aidj/all-in-one — MIT. PyPI 1.1.0 (2023-10-10), classifiers Python 3.8-3.11 (https://pypi.org/project/allin1/). Paper: Kim & Nam, WASPAA 2023 (arXiv 2307.16425).
- **Install (README)**: 1) PyTorch; 2) NATTEN — Linux: wheel; **Windows: build from source** (clone SHI-Labs/NATTEN, `make`; macOS auto-installs); 3) `pip install git+https://github.com/CPJKU/madmom` then `pip install allin1`; ffmpeg optional for MP3 (README). `pyproject.toml` deps: numpy, librosa, demucs, hydra-core, omegaconf, huggingface_hub, matplotlib; `natten` only auto-installed on Darwin (https://raw.githubusercontent.com/mir-aidj/all-in-one/main/pyproject.toml).
- **madmom dependency (FLAG-NC)**: `allin1/postprocessing/metrical.py` imports `madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], threshold=..., fps=...)` (https://raw.githubusercontent.com/mir-aidj/all-in-one/main/src/allin1/postprocessing/metrical.py). madmom code is BSD-2 but its model/data files are CC BY-NC-SA 4.0 (https://raw.githubusercontent.com/CPJKU/madmom/main/LICENSE). The DBN processor is algorithmic code (no `.npy/.pkl` model), so the NC clause arguably does not attach, but madmom must be installed from git (PyPI build fails on new Python/numpy: https://github.com/CPJKU/beat_this).
- **NATTEN status (critical gotcha)**: allin1 1.1.0 targets the old NATTEN API; with NATTEN >= 0.17.5 you get `cannot import name 'natten1dav'` / "Expected attention dim was 5, got 9" (https://github.com/mir-aidj/all-in-one/issues/30). Community PR #27 patches imports; PR #37 (closed unmerged, Nov 2025) documents a working recipe: `torch==2.7.0` + `pip install natten==0.21.0+torch270cu128 -f https://whl.natten.org` + madmom from git + `pip install git+https://github.com/not-matt/all-in-one` (https://github.com/mir-aidj/all-in-one/pull/37). PR #39 (open, Sep 2026) proposes making NATTEN optional with a plain-PyTorch fallback (https://github.com/mir-aidj/all-in-one/issues?q=natten).
- **NATTEN on Windows**: official wheels (`pip install natten==<ver>+torch<ver>cu<ver> -f https://whl.natten.org`) are **Linux only**; Windows = experimental MSVC source build via `WindowsBuilder.bat install` from the VS Native Tools prompt, needs CUDA Toolkit 12.0+, may hit a missing-`nvToolsExt` error requiring edits to PyTorch's CMake files; Blackwell kernels need CUDA 12.8+ (not in cu126 builds); manual builds take `CUDA_ARCH="10.0;10.3"` for SM100/SM103 — sm_120 (RTX 50 consumer) is not listed (https://natten.org/install/). Current NATTEN requires torch >= 2.7 (officially >= 2.8), MIT license (https://github.com/SHI-Labs/NATTEN). **Risk: high on Win11 + 5090.** Mitigations: (a) WSL2 + Linux wheel; (b) the not-matt fork; (c) replace with beat_this (sec. 3) for beats/downbeats and accept no segment labels in v1.
- **CLI**: `allin1 file1.wav file2.mp3` -> `./struct/<name>.json`; flags `-o/--out-dir`, `-v/--visualize` (`--viz-dir`), `-s/--sonify` (`--sonif-dir`), `-a/--activ`, `-e/--embed`, `-m/--model harmonix-all|harmonix-foldN`, `-d/--device`, `-k/--keep-byproducts`, `--demix-dir ./demix`, `--spec-dir ./spec` (README).
- **Python API**:
```python
import allin1
r = allin1.analyze("song.wav", out_dir=None, visualize=False, sonify=False,
                   model="harmonix-all", device="cuda", include_activations=False,
                   include_embeddings=False, demix_dir="./demix", spec_dir="./spec",
                   keep_byproducts=True, multiprocess=True)
r.bpm; r.beats; r.downbeats; r.beat_positions; r.segments  # [{start,end,label}]
```
Labels: `start, end, intro, outro, break, bridge, inst, solo, verse, chorus`. Activations/embeddings at 100 fps; embeddings shape `[stems=4, T, 24]` (bass, drums, other, vocals) (README).
- **Pre-separated stems — YES**: `demix.py` runs `python -m demucs.separate --out <demix_dir> --name htdemucs --device <dev> <files>` and **skips** a track when `demix_dir/htdemucs/<track_stem>/{bass,drums,other,vocals}.wav` all exist (https://raw.githubusercontent.com/mir-aidj/all-in-one/main/src/allin1/demix.py). So place your own Demucs `htdemucs` stems (4-stem model, 44.1 kHz stereo WAV as Demucs writes them: https://github.com/facebookresearch/demucs) in that layout and pass `demix_dir`. Results are skipped per track if `out_dir/<name>.json` exists (https://raw.githubusercontent.com/mir-aidj/all-in-one/main/src/allin1/analyze.py).
- **Speed**: RTX 4090 + i9-10940X: 10 songs (33 min) in 73 s with `harmonix-all` (README). MP3 decoding can shift beats 20-40 ms; convert to WAV first (README).
- Demucs itself is MIT, archived 2025-01-01, maintained fork at github.com/adefossez/demucs (https://github.com/facebookresearch/demucs).

---

## 3. Beat / downbeat alternatives

| Tool | License | Windows | Notes |
|---|---|---|---|
| **beat_this** (CPJKU, ISMIR 2024) | MIT code **and** weights (https://github.com/CPJKU/beat_this) | pure PyTorch, no custom CUDA ops | **Recommended for v1.** |
| BeatNet | CC-BY-4.0 (https://github.com/mjhydri/BeatNet) | needs madmom + PyAudio wheel | madmom 0.16.1 breaks on Py 3.10+/NumPy 1.24+; 22.05 kHz input; output `(n,2)` beats/downbeats. **FLAG**: pulls madmom (NC models). Skip. |
| madmom | BSD-2 code, CC BY-NC-SA 4.0 models (https://raw.githubusercontent.com/CPJKU/madmom/main/LICENSE) | git install only | **FLAG-NC** — avoid its pretrained RNN beat/downbeat models. |
| Essentia | AGPLv3 non-commercial; models CC BY-NC-ND (https://essentia.upf.edu/licensing_information.html) | "Python bindings are not supported yet" (https://essentia.upf.edu/installing.html) | **FLAG-AGPL**, unusable here. |
| librosa `beat_track` | ISC | yes | no downbeats; fallback only. |

**beat_this details** (https://github.com/CPJKU/beat_this): `pip install beat-this` after torch >= 2.0; deps tqdm, einops, soxr, rotary-embedding-torch; ffmpeg/torchaudio backend for non-WAV. CLI `beat_this input.wav -o out.beats` (`--gpu=-1` CPU, `--float16`, `--touch-first/--skip-existing` for parallel batches). Checkpoints `final0` (~78 MB, default), `small0` (~8.1 MB). Optional `--dbn` uses madmom (not needed; default "minimal" postprocessor). API:
```python
from beat_this.inference import File2Beats, Audio2Beats
f2b = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
beats, downbeats = f2b("song.wav")        # seconds; verify order (File2File unpacks as downbeats, beats)
a2b = Audio2Beats("final0", device="cuda", float16=True); beats, downbeats = a2b(signal, sr)
```
Internally: mono-sum, soxr resample to 22050 Hz, processes spectrograms in 1500-frame chunks with 6-frame borders (https://github.com/CPJKU/beat_this/blob/main/beat_this/inference.py). No BPM field — derive `bpm = 60 / median(np.diff(beats))` and `beats_per_bar` from downbeat spacing. Caveat: results on training-set songs are unrealistically good (README).

---

## 4. Pitch / note / key detection

### 4.1 Basic Pitch (Spotify) — polyphonic note transcription
- Apache-2.0, copyright 2022 Spotify (https://github.com/spotify/basic-pitch). PyPI 0.4.0 (2024-08-16); classifiers Python 3.8-3.11 only — **FLAG: no declared 3.12 support**; test in a 3.11 side-venv or accept risk (https://pypi.org/project/basic-pitch/). Default runtime on Windows = **ONNX** (CoreML on macOS, TFLite on Linux); TensorFlow not installed unless `pip install basic-pitch[tf]` (README). Extras: `onnx`, `coreml`, `tf`.
- Model constants: input resampled to **22050 Hz mono**, window `AUDIO_N_SAMPLES=43844` (~2 s), `FFT_HOP=256`, 86 fps annotations, 264 contour bins (3/semitone over 88 keys from 27.5 Hz) (https://raw.githubusercontent.com/spotify/basic-pitch/main/basic_pitch/constants.py). Windows overlap by 30 frames and are stitched (https://raw.githubusercontent.com/spotify/basic-pitch/main/basic_pitch/inference.py).
- CLI: `basic-pitch <out_dir> <audio>... [--sonify-midi] [--save-model-outputs] [--save-note-events] [--model-serialization ...]` (README).
- API (https://raw.githubusercontent.com/spotify/basic-pitch/main/basic_pitch/inference.py):
```python
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
model = Model(ICASSP_2022_MODEL_PATH)          # load once for batches
model_output, midi, note_events = predict("other.wav", model,
    onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=127.7,  # ms
    minimum_frequency=None, maximum_frequency=None, multiple_pitch_bends=False,
    melodia_trick=True, midi_tempo=120)
# note_events: [(start_s, end_s, pitch_midi, amplitude_0to1, pitch_bend_list|None)]
# model_output keys: "note", "onset", "contour"
```
Use note onsets as phrase/note chop points on melodic stems, and the pitch histogram of notes for a key vote.

### 4.2 torchcrepe — monophonic f0 (vocals, bass)
- MIT; `pip install torchcrepe` after PyTorch (https://github.com/maxrmorrison/torchcrepe). `predict(audio, sample_rate, hop_length=None, fmin=50., fmax=2006., model='full'|'tiny', decoder=torchcrepe.decode.viterbi, return_periodicity=False, batch_size=None, device='cpu', pad=True)`; internally resamples to 16 kHz via resampy if needed (https://raw.githubusercontent.com/maxrmorrison/torchcrepe/master/torchcrepe/core.py). Clean-up: `torchcrepe.filter.median(periodicity, 3)`, `torchcrepe.threshold.At(0.21)(pitch, periodicity)`, `torchcrepe.threshold.Silence(-60.)(periodicity, audio, sr, hop)` (README). Hop `int(sr/200)` = 5 ms (README).

### 4.3 librosa pyin (CPU fallback, no model download)
`librosa.pyin(y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None, n_thresholds=100, resolution=0.1, switch_prob=0.01, no_trough_prob=0.01, fill_na=np.nan)` -> `(f0, voiced_flag, voiced_prob)`; Mauch & Dixon 2014 (https://raw.githubusercontent.com/librosa/librosa/main/librosa/core/pitch.py). Good enough for root-note tagging of bass one-shots.

### 4.4 Key: see 1.6 (chroma + Krumhansl). Combine stem-level key (other+bass+vocals) with per-slice root from pyin/Basic Pitch notes.

---

## 5. Loudness & export levels

- **pyloudnorm** — MIT, `pip install pyloudnorm` (https://github.com/csteinmetz1/pyloudnorm):
```python
import pyloudnorm as pyln
meter = pyln.Meter(sr)                     # BS.1770, 400 ms blocks; block_size=0.200 possible
lufs = meter.integrated_loudness(x)        # x shape (samples,) or (samples, channels)
lra  = meter.loudness_range(x)
y = pyln.normalize.loudness(x, lufs, -14.0); y = pyln.normalize.peak(x, -1.0)
```
Gotchas (https://raw.githubusercontent.com/csteinmetz1/pyloudnorm/master/pyloudnorm/meter.py): input shorter than `block_size` fails `util.valid_audio` / yields nonsense block counts — for one-shots < 400 ms **zero-pad to >= 400 ms or use `block_size=0.100`**, and treat the result as "short-term loudness"; signals under the -70 LUFS absolute gate return `-inf` (write `null` in JSON). Filter classes: default BS.1770, `"DeMan"`, `"Fenton/Lee 1/2"`, `"Dash et al."`, `"custom"` (README).
- **Peak normalization**: `pyln.normalize.peak(x, -1.0)` or `x * (10**(target_db/20) / np.abs(x).max())`. True-peak is not implemented in pyloudnorm; leave >= 1 dB headroom.
- **Dither for 16-bit**: converting float to 16-bit without dither turns quantization error into distortion; add TPDF dither of 2 LSB amplitude when the file may be processed further (lowest-power ideal dither, no noise modulation), reserve noise-shaped dither for final masters (https://en.wikipedia.org/wiki/Dither). libsndfile/soundfile do **not** dither; soundfile only enables `SFC_SET_CLIPPING` on every opened file so out-of-range floats clip instead of wrapping (https://raw.githubusercontent.com/bastibe/python-soundfile/master/soundfile.py). **v1 decision**: export 24-bit PCM (`PCM_24`) or `FLOAT` and skip dither; offer `--bit-depth 16` with `rng.triangular(-1,0,1)/32768` TPDF added before `PCM_16` write.

---

## 6. Audio I/O & DSP

### 6.1 soundfile (BSD-3; bundled libsndfile is LGPL — dynamic, **FLAG-LGPL (ok)**)
`pip install soundfile` ships libsndfile for Win 64/32 (https://python-soundfile.readthedocs.io/).
```python
import soundfile as sf
x, sr = sf.read("in.wav", dtype="float32", always_2d=True, start=0, stop=None)
sf.write("out.wav", x, sr, subtype="PCM_24")   # WAV default subtype is PCM_16
with sf.SoundFile("in.wav") as f: f.seek(start); seg = f.read(n_frames, dtype="float32", always_2d=True)
```
`sf.available_subtypes("WAV")`, `sf.default_subtype("WAV") == "PCM_16"`; array dtype does not pick file type — `subtype` does; data is converted (https://python-soundfile.readthedocs.io/). Subtypes incl. `PCM_S8, PCM_16, PCM_24, PCM_32, PCM_U8, FLOAT, DOUBLE` (https://raw.githubusercontent.com/bastibe/python-soundfile/master/soundfile.py). Partial reads via `seek/read` avoid loading long stems twice.

### 6.2 Resampling
- **python-soxr** `pip install soxr` — **LGPL-2.1-or-later (FLAG-LGPL, dynamic dep ok, do not vendor)**, `soxr.resample(x, in_rate, out_rate)` (1-D or frames x channels; float32/float64/int16/int32), `ResampleStream` for chunks; benchmark 10 s 48k->44.1k: soxr HQ 10.8 ms vs scipy 21.3 ms (https://github.com/dofuuz/python-soxr). librosa already depends on it (`res_type="soxr_hq"`).
- **resampy** — ISC, `resampy.resample(x, sr_orig, sr_new, axis=-1, filter='kaiser_best', parallel=...)` (https://github.com/bmcfee/resampy, https://resampy.readthedocs.io/en/stable/api.html). Pure-permissive alternative if LGPL is a concern.

### 6.3 Time-stretch / pitch-shift
| Option | License | Verdict |
|---|---|---|
| **python-stretch** (Signalsmith Stretch binding) 0.3.1 | MIT; Windows wheels cp3.8-3.12 (https://pypi.org/project/python-stretch/); upstream C++ MIT (https://github.com/Signalsmith-Audio/signalsmith-stretch) | **Use.** |
| librosa `time_stretch`/`pitch_shift` | ISC | phase vocoder; fallback. |
| pyrubberband / Rubber Band | GPL-2+ or commercial (https://github.com/spotify/pedalboard) | **FLAG-GPL** |
| Spotify pedalboard | GPL-3.0 (statically links JUCE, Rubber Band, LAME) (https://github.com/spotify/pedalboard) | **FLAG-GPL** |
```python
import python_stretch as ps
st = ps.Signalsmith.Stretch(); st.preset(n_channels, sr)   # audio: (channels, samples)
st.setTransposeSemitones(0); st.timeFactor = target_bpm / src_bpm   # verify direction on a test loop
y = st.process(audio)
```
Quality sweet spot 0.75x-1.5x; formant options exist (https://github.com/Signalsmith-Audio/signalsmith-stretch). Use only for optional "conform loops to target BPM"; v1 default exports untouched audio + tempo metadata.

### 6.4 Click-free chops (practice)
- Snapping cut points to zero crossings reduces but does not eliminate clicks (the following half-cycle can still be loud); always add short fades/crossfades too (https://community.ardour.org/node/1067, https://www.elektronauts.com/t/digitakt-os-1-50/189531?page=40). BeatCleaver manual: place slice points near zero crossings to reduce pops (https://www.oscillicious.com/beatcleaver/BeatCleaver%20Manual.pdf). Cut on integer sample boundaries (fractional cuts cause single-sample gaps: https://zynewave.com/topic/clicks-when-cutting-audio-parts).
- v1 rule: `start = backtracked onset` -> move to nearest zero crossing within +-2 ms (`librosa.zero_crossings`), **never later than the onset peak minus 1 ms**; apply 1-2 ms linear fade-in and 5-10 ms fade-out (equal-power for loop seams). For loops, cut exactly on beat-grid samples (no zero snapping, which would break tempo), then fade 2 ms both ends.

---

## 7. Tagging / classifying one-shots

| Model | License (code / weights) | Input | Output | Fit |
|---|---|---|---|---|
| **PANNs Cnn14** via `panns-inference` | MIT / checkpoint on Zenodo (https://github.com/qiuqiangkong/panns_inference, https://github.com/qiuqiangkong/audioset_tagging_cnn) | 32 kHz mono, `(batch, samples)` | 527 AudioSet probs + 2048-d embedding; mAP 0.431 | AudioSet has Bass drum, Snare drum, Hi-hat, Cymbal, Tom-tom, Clapping etc. Batch-able on GPU: practical for thousands of slices. |
| YAMNet | Apache-2.0 (https://github.com/tensorflow/models) | 16 kHz, 0.96 s frames, needs >= 975 ms | 521 classes, 1024-d emb | Requires TF + Keras 2 (incompatible with Keras 3 / TF >= 2.16) (https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md). Skip. |
| LAION-CLAP | code CC0-1.0; weights HF `lukewys/laion_clap` tagged cc0-1.0 (https://github.com/LAION-AI/CLAP, https://huggingface.co/lukewys/laion_clap) | 48 kHz, `(N, T)` | 512-d audio/text embeddings (README example shape) | Zero-shot: rank `audio_emb @ text_emb("This is a sound of a kick drum")`. Checkpoints `630k-audioset-best.pt`, `music_audioset_epoch_15_esc_90.14.pt` (HTSAT-base). Heavier than PANNs but gives free-text tags and dedup embeddings in one pass. |
| Oneshot_Drum_Classification | no license file; no weights (https://github.com/adrienchaton/Oneshot_Drum_Classification) | — | — | Not usable. |
| DLDC (alecames) | unverified; ships `model_v0.12.pt` for kick/snare/hat/tom/cymbal (https://github.com/alecames/DLDC) | — | 5 classes | Check LICENSE before use; small student project. |
| AudioStellar | desktop app, "free, open source", license not listed on Flathub (https://flathub.org/apps/xyz.audiostellar.AudioStellar) | — | — | App, not a library. Ignore. |

```python
from panns_inference import AudioTagging, labels
at = AudioTagging(checkpoint_path=None, device="cuda")      # downloads Cnn14_mAP=0.431.pth
clip, emb = at.inference(batch_32k)                           # clip: (B,527), emb: (B,2048)
```
**v1 approach**: (1) cheap heuristic pass with librosa features (centroid, flatness, rolloff, duration, low-band energy ratio, pyin voicing) to assign `kick/snare/hat/perc/tonal`; (2) PANNs on the drum-stem slices (pad to >= 1 s at 32 kHz) to confirm with AudioSet labels mapped to our vocabulary; (3) optional CLAP zero-shot for `--rich-tags`.

---

## 8. Near-duplicate detection

- **Chromaprint / pyacoustid**: pyacoustid MIT, Chromaprint "as a whole ... LGPL 2.1" because it embeds FFmpeg code (https://github.com/beetbox/pyacoustid, https://raw.githubusercontent.com/acoustid/chromaprint/master/LICENSE.md). Designed for full tracks (needs `fpcalc` binary + FFmpeg); poor fit for 100 ms-2 s slices. Skip.
- **v1**: embedding cosine similarity. Cheap path: per-slice feature vector = mean+std of 20 MFCCs + centroid/flatness/rolloff + log-duration (librosa), L2-normalize, cosine > 0.98 and duration ratio within 5% => duplicate group (keep loudest/earliest). Rich path: PANNs 2048-d or CLAP 512-d embeddings (sec. 7), same thresholding; group with a union-find over a `sklearn.neighbors.NearestNeighbors` radius query. Store `dup_group` id rather than deleting.

---

## 9. Recommendations for v1

### 9.1 Stack (all permissive unless noted)
torch (cu128+ build for sm_120) · soundfile (BSD; libsndfile LGPL dyn) · soxr (LGPL dyn, via librosa) · librosa 1.x · beat_this (MIT) · torchcrepe (MIT) · basic-pitch (Apache; 3.11-only classifiers, isolate or defer) · pyloudnorm (MIT) · python-stretch (MIT, optional) · panns-inference (MIT) · laion-clap (CC0, optional). **Defer allin1** until NATTEN installs cleanly on Win11/5090 or PR #39 lands; it is the only source of section labels, so v1 ships `sections=null` and a `--allin1-json` flag that reads an existing `struct/*.json` if the user ran it in WSL.

### 9.2 Pipeline per stem (input: Demucs `htdemucs` 44.1 kHz stems; keep native rate for export, analyze at 22.05 kHz)
1. **Global**: beat_this on the full mix -> `beats[]`, `downbeats[]`; `bpm = 60/median(diff(beats))`; `beats_per_bar` = mode of beats between downbeats. Key: HPSS harmonic of (other+bass+vocals) -> `chroma_cqt(tuning=estimate_tuning)` -> Krumhansl correlation -> `key`, `mode`, `key_confidence` (margin between best two). LUFS of each stem for relative gain.
2. **drums** -> one-shots: `onset_strength(hop=256, n_fft=1024, aggregate=np.median)`; `onset_detect(backtrack=True, delta=0.15, wait=~50 ms, units="samples")`; end = next onset or decay to -40 dB below slice peak, max 2 s; snap start to zero crossing, fades 1 ms / 8 ms; tag with heuristics + PANNs; dedup by cosine; also emit **loops**: cut `[downbeat_i, downbeat_{i+1,2,4}]` (1/2/4 bars), rank by RMS stability.
3. **bass** -> loops (1/2 bars on downbeats) **and** note one-shots: `onset_detect(backtrack=True, delta=0.1)` on the bass stem; per-slice `pyin(fmin=30, fmax=400)` median f0 -> `root_midi`; drop slices with `voiced_prob < 0.5`.
4. **other** (keys/guitars/synths) -> loops 1/2/4 bars + **phrases**: Basic Pitch `note_events` -> cluster onsets with gaps > 1 beat into phrases; phrase start snapped to nearest beat; per-slice key via chroma vote; drop near-silent bars (`effects.split(top_db=45, ref=1.0)`).
5. **vocals** -> phrases: `effects.split(top_db=40)` on the vocal stem merged with beat grid (start snapped to nearest beat if within 60 ms, else free); torchcrepe median f0 + voiced ratio -> `root_midi`, `voiced_ratio`; tag `vocal_phrase`, duration in beats.
6. **Export**: `out/<song>/<stem>/<type>/<song>_<stem>_<type>_<idx:03d>_<tag>_<bpm>_<key>.wav` as PCM_24 at source rate; `index.json` (schema below). No normalization applied to files in v1 (store measurements; add `--normalize-peak -1` flag).

### 9.3 Per-sample metadata schema (`index.json`)
```json
{
  "schema_version": 1,
  "source": {"file": "song.wav", "sr": 44100, "duration_s": 213.4, "bpm": 124.0,
             "beats_per_bar": 4, "key": "F#", "mode": "minor", "key_confidence": 0.31,
             "beat_tracker": "beat_this/final0", "separator": "demucs/htdemucs",
             "sections": null},
  "samples": [{
    "id": "song_drums_oneshot_017", "path": "drums/oneshots/song_drums_oneshot_017_kick_124_F#m.wav",
    "stem": "drums", "type": "oneshot|loop|phrase",
    "start_s": 12.3451, "end_s": 12.7102, "start_sample": 544420, "length_samples": 16101,
    "start_beat": 25.0, "length_beats": 0.75, "bars": null, "bar_aligned": false,
    "bpm": 124.0, "root_midi": 36, "root_note": "C1", "root_confidence": 0.82,
    "key": null, "voiced_ratio": 0.0,
    "lufs_i": -18.2, "lufs_short": null, "peak_dbfs": -3.1, "rms_dbfs": -14.9,
    "spectral_centroid_hz": 420.0, "spectral_flatness": 0.08, "rolloff85_hz": 1900.0,
    "tags": ["kick"], "tag_source": "heuristic+panns", "tag_confidence": 0.91,
    "panns_top3": [["Bass drum", 0.81], ["Drum", 0.62], ["Drum kit", 0.44]],
    "dup_group": 3, "is_primary": true, "fade_in_ms": 1.0, "fade_out_ms": 8.0,
    "zero_snapped": true, "embedding_ref": "emb/panns.npy#17"
  }]
}
```
Field notes: `lufs_*` are `null` when pyloudnorm returns `-inf` or the slice is shorter than the block; `type=loop` fills `bars`/`bar_aligned=true` and `length_beats` is an integer; `root_*` filled from pyin/torchcrepe (bass/vocals) or Basic Pitch note histogram (other); `embedding_ref` points into a sidecar `.npy` so the JSON stays small.

### 9.4 License summary for the MIT project
OK: librosa (ISC), soundfile (BSD-3), resampy (ISC), beat_this (MIT), torchcrepe (MIT), basic-pitch (Apache-2.0), pyloudnorm (MIT), python-stretch (MIT), panns-inference (MIT), laion-clap (CC0), allin1 (MIT), demucs (MIT), NATTEN (MIT), pyacoustid (MIT). **FLAG-LGPL (dynamic only, document in NOTICE)**: libsndfile via soundfile, libsoxr via python-soxr, Chromaprint. **FLAG-NC/GPL (exclude)**: madmom models (CC BY-NC-SA; transitively needed by allin1/BeatNet), Essentia (AGPL), pedalboard / Rubber Band / pyrubberband (GPL), keyfinder-cli/libkeyfinder (GPL-3).