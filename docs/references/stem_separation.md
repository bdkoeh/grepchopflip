# Stem separation reference (automated sample chopper)

Compiled 2026-10-24 from upstream READMEs, PyPI pages and docs. Target: Windows 11, Python 3.12, RTX 5090 (Blackwell, sm_120, 32 GB VRAM), 64 GB RAM, MIT-licensed project. Every claim carries its source URL; where a source was silent or two sources disagree, that is stated.

Verify before pinning: numbers below (package versions, CUDA matrix) move monthly.

---

## 1. python-audio-separator (nomadkaraoke) -- primary engine

Repo: https://github.com/nomadkaraoke/python-audio-separator -- PyPI: https://pypi.org/project/audio-separator/ -- License: MIT (README asks that integrations credit UVR / Anjok07 for UVR-trained models) https://github.com/nomadkaraoke/python-audio-separator

- Latest: **0.47.1**, released 2026-10-19; requires `Python >=3.10, !=3.14.1` https://pypi.org/project/audio-separator/
- Hard deps (no version pins on torch/onnxruntime beyond floors): `torch>=2.3,<3` (non-macOS), `onnx-weekly`, `onnx2torch-py313>=1.6`, `librosa>=0.10`, `numpy>=2`, `samplerate==0.1.0`, `diffq-fixed>=0.2` (Windows), `rotary-embedding-torch>=0.6.1,<0.7` https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/pyproject.toml
- Extras: `cpu` = `onnxruntime>=1.17`; `gpu` = `onnxruntime-gpu>=1.17`; `dml` = `onnxruntime-directml>=1.17` + `torch_directml` https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/pyproject.toml
- FFmpeg must be installed separately on pip installs; check with `audio-separator --env_info` (should print "FFmpeg installed" and CUDAExecutionProvider available) https://github.com/nomadkaraoke/python-audio-separator
- README still says "supported CUDA versions 11.8 and 12.2" (stale; contributor lock is PyTorch 2.13 + CUDA 13.0, needing an R580+ NVIDIA driver) https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md
- The pip extra does NOT pin a CUDA-specific torch; you must install torch yourself first (see section 5), then `pip install "audio-separator[gpu]"`.

### 1.1 Install (Windows, RTX 5090)

```powershell
# 1) PyTorch first, CUDA 13.0 wheel (sm_120 needs CUDA >= 12.8; see section 5)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
# 2) audio-separator with GPU extra (pulls onnxruntime-gpu; >=1.27 defaults to CUDA 13.0)
pip install "audio-separator[gpu]"
# 3) sanity
audio-separator --env_info
python -c "import torch;print(torch.cuda.get_arch_list())"   # must include sm_120
```
Sources: PyTorch 2.14 ships CUDA 12.6/13.0/13.2 wheels, default wheel = CUDA 13.0 https://pytorch.org/blog/pytorch-2-14-release-blog/ ; onnxruntime-gpu on PyPI is built with CUDA 13.0 from 1.27 onward https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html ; `torch.cuda.get_arch_list()` check https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099

If ORT fails with "Failed to load library" / DLL error 126, the README's fix is: uninstall torch and onnxruntime, purge pip cache, `--force-reinstall` torch + onnxruntime-gpu with matching CUDA majors https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md. On Windows, ORT can borrow CUDA/cuDNN DLLs from the installed torch if majors match: `import torch` before creating a session, or call `onnxruntime.preload_dlls()` (ORT >= 1.21) https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

### 1.2 Python API

Constructor (defaults verbatim from `separator.py`) https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py :

```python
from audio_separator.separator import Separator
Separator(
    log_level=logging.INFO, log_formatter=None,
    model_file_dir="/tmp/audio-separator-models/",   # hardcoded default; env AUDIO_SEPARATOR_MODEL_DIR overrides
    output_dir=None,                 # cwd
    output_format="WAV",             # CLI default is FLAC
    output_bitrate=None,
    normalization_threshold=0.9, amplification_threshold=0.0,
    output_single_stem=None,         # e.g. "Drums", "Vocals", "Instrumental", "Bass", "Guitar", "Piano", "Other"
    invert_using_spec=False,
    sample_rate=44100,
    use_soundfile=False, use_autocast=False, use_directml=False,
    chunk_duration=None,             # seconds; README suggests 300-600 for >1h files
    mdx_params={"hop_length":1024,"segment_size":256,"overlap":0.25,"batch_size":1,"enable_denoise":False},
    vr_params={"batch_size":1,"window_size":512,"aggression":5,"enable_tta":False,"enable_post_process":False,"post_process_threshold":0.2,"high_end_process":False},
    demucs_params={"segment_size":"Default","shifts":2,"overlap":0.25,"segments_enabled":True},
    mdxc_params={"segment_size":256,"override_model_segment_size":False,"batch_size":None,"overlap":None,"pitch_shift":0},
    ensemble_algorithm=None, ensemble_weights=None, ensemble_preset=None,
    info_only=False, use_torch_compile=False, use_native_fp16=False,
)
```

- `model_file_dir` default is the literal string `/tmp/audio-separator-models/` (not `tempfile`); on Windows that resolves to `C:\tmp\audio-separator-models\` on the current drive. Always pass an explicit path (e.g. `%LOCALAPPDATA%\beat_making\models`) https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py
- `load_model(model_filename=...)` -- Python default `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt`; CLI default `model_bs_roformer_ep_317_sdr_12.9755.yaml`. Same-model reloads are reused; `force_reload=True` to reload https://github.com/nomadkaraoke/python-audio-separator
- `separate(audio_file_path, custom_output_names=None)` -- accepts a path, list of paths, or directory; returns `list[str]` of output paths; list/dir inputs raise `BatchSeparationError` on partial failure; `custom_output_names={"Vocals": "vocals_output"}` renames stems https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py
- `use_autocast` and `use_native_fp16` are mutually exclusive (ValueError). `use_native_fp16` verified only for MelBand/BS-RoFormer on CUDA/MPS. `use_torch_compile` needs torch >= 2.6; first run slower, helps warm repeated same-shape runs https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md
- Device detection: `torch.cuda.is_available()` -> CUDA; ORT uses `CUDAExecutionProvider` only if present in `ort.get_available_providers()`, else warns and runs CPU https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py

Listing models programmatically https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py :

```python
s = Separator(info_only=True)
by_arch = s.list_supported_model_files()      # {"VR":{...},"MDX":{...},"Demucs":{...},"MDXC":{...}}
                                              # each friendly-name -> {filename, scores, stems, target_stem, download_files}
flat = s.get_simplified_model_list(filter_sort_by="drums")   # keyed by filename; {"Name","Type","Stems","SDR"}; sorts by that stem's SDR
s.list_ensemble_presets()
```
CLI equivalents: `audio-separator --list_models`, `--list_filter=drums`, `--list_limit N`, `--list_format=json`, `--list_presets` https://github.com/nomadkaraoke/python-audio-separator

Two-pass example (full mix -> 4 stems, then drum stem -> drum parts; see section 2 for the drum model caveat):

```python
import logging
from audio_separator.separator import Separator

MODELS = r"C:\Users\bob\AppData\Local\beat_making\models"
sep = Separator(log_level=logging.WARNING, model_file_dir=MODELS,
                output_dir="work/stems", output_format="WAV", sample_rate=44100)

sep.load_model("htdemucs_ft.yaml")                 # vocals/drums/bass/other
stems = sep.separate("song.wav")                   # ["song_(Vocals)_htdemucs_ft.wav", ...]
drums = next(p for p in stems if "(Drums)" in p)

sep.output_dir = "work/drum_parts"
sep.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")   # kick/snare/toms/hh/ride/crash (6 stems)
parts = sep.separate(drums)
```
Output filename pattern `<input>_(<Stem>)_<model>.wav` is the library default (observed in README examples) https://github.com/nomadkaraoke/python-audio-separator

### 1.3 Model catalogue and download location

- Model metadata JSON: `audio_separator/models.json` has four sections `vr_download_list`, `mdx_download_list`, `mdx23c_download_list`, `roformer_download_list`; mdx23c/roformer entries map `ckpt -> config.yaml` https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/models.json
- Download hosts: UVR models from GitHub releases `TRvlvr/model_repo`; fallback configs from `nomadkaraoke/python-audio-separator` releases (`model-configs` tag); Demucs weights from `dl.fbaipublicfiles.com`; score metadata from `raw.githubusercontent.com/TRvlvr/application_data`. No Hugging Face hosts https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py
- Demucs support is "v4 only" per code comment; README `--list_filter=drums` shows `htdemucs_ft.yaml`, `hdemucs_mmi.yaml`, `htdemucs.yaml` (vocals/drums/bass/other) and `htdemucs_6s.yaml` (vocals/drums/bass/guitar/piano/other) https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md
- Recommended by README for vocals/instrumental: `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (vocals 12.9 / instrumental 17.0 SDR); also `melband_roformer_big_beta4.ckpt`, `UVR_MDXNET_KARA_2.onnx`, `MDX23C-8KFFT-InstVoc_HQ.ckpt` https://github.com/nomadkaraoke/python-audio-separator
- Note: BS-RoFormer/MelBand/MDX23C-InstVoc entries in audio-separator are **2-stem** (vocals/instrumental). The only multi-stem (4/6) models exposed are the Demucs v4 family, plus the 6-stem **drum** model `MDX23C-DrumSep-aufr33-jarredou.ckpt` (config `config_drumsep_mdx23c.yaml`) in `mdx23c_download_list` https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/models.json
- Checkpoint sizes (same files on HF mirror): MDX23C-DrumSep 438 MB, MDX23C-InstVoc_HQ 448 MB https://huggingface.co/Politrees/UVR_resources/tree/main/models/MDX23C ; MelBand Kim vocals ~913 MB https://github.com/openmirlab/melband-roformer-infer

### 1.4 RTX 50-series status

- Issue #219 "is audio separator compatible with the 5090?" (opened 2025-06-30): failure was torch built for cu126; fix was switching to the cu128 wheel. Maintainer agreed to document it; issue still open as of capture https://github.com/nomadkaraoke/python-audio-separator/issues/219
- No sm_120-specific code in audio-separator; the dependency is entirely on torch (>= 2.7 with CUDA >= 12.8) and onnxruntime-gpu matching that CUDA major https://pytorch.org/blog/pytorch-2-7/ https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

---

## 2. Drum sub-separation (drum stem -> kick / snare / toms / cymbals...)

### 2.1 Public models and where they live

| Model | Arch | Stems | SDR (jarredou validation set: kick/snare/toms/hh/cym) | Weights | Weights license |
|---|---|---|---|---|---|
| DrumSep (inagoy, 2022 thesis) | Hybrid Demucs | kick, snare, toms, cymbals | 10.52 / 6.05 / 4.68 / 5.03 | `model_drumsep.th` + `config_drumsep.yaml` from ZFTurbo release v1.0.5; original via gdown id `1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC` | Repo code MIT; weights license not stated |
| DrumSep MDX23C v0.1 (aufr33 + jarredou) | MDX23C TFC-TDF | kick, snare, toms, hh, ride, crash | 14.54 / 9.79 / 10.63 / 3.19 / 6.08 (ride+crash) | In audio-separator as `MDX23C-DrumSep-aufr33-jarredou.ckpt`; re-hosted by openmirlab (sha256 d2a4aa53...) | **Not formally documented** -- treat as non-commercial-safe only |
| DrumSep MDX23C 5-stem (jarredou) | MDX23C | kick, snare, toms, hh, cymbals | 16.66 / 11.53 / 12.33 / 4.04 / 6.36 | Was `drumsep_5stems_mdx23c_jarredou.ckpt` on github.com/jarredou/models -- **now 404** | Undocumented; no longer auto-downloadable |
| MVSep SCNet XL (4/5/6 stems), MelBand RoFormer DrumSep (4/6), BS-RoFormer (5), ensemble | SCNet / RoFormer | up to kick, snare, toms, hh, ride, crash | MelBand 4-stem: 18.67 / 13.55 / 13.60 / 8.76 | **MVSep service only**; no public checkpoint found | n/a |
| LarsNet (Politecnico di Milano) | 5 parallel U-Nets | kick, snare, toms, hihat, cymbals | not given | Google Drive zip, ~562 MB | **CC BY-NC 4.0** |

Sources: ZFTurbo pretrained list (SDRs, URLs) https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md ; inagoy repo + scripts https://github.com/inagoy/drumsep https://github.com/inagoy/drumsep/blob/main/drumsepInstall https://github.com/inagoy/drumsep/blob/main/drumsep ; MVSep model table https://mvsep.com/algorithms/29?lang=en ; license analysis + rehost https://github.com/openmirlab/mdxnet-infer https://pypi.org/project/mdxnet-infer/ ; jarredou/models dead https://github.com/jarredou/models/releases ; LarsNet https://github.com/polimi-ispl/larsnet ; config_drumsep stems https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_drumsep.yaml

MVSep's own SDR table (its main dataset) ranks MelBand RoFormer 4-stem highest (kick 22.22, snare 17.09, toms 15.86, cymbals 11.87) vs HDemucs DrumSep lowest (14.13 / 8.42 / 5.67 / 5.63); MVSep is the vendor, so treat as indicative https://mvsep.com/algorithms/29?lang=en

### 2.2 Install paths on Windows

**Option A (simplest, already wired): audio-separator.** `MDX23C-DrumSep-aufr33-jarredou.ckpt` is in `models.json`, downloads to `model_file_dir`, runs on CUDA via the MDXC (torch) path. No extra install https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/models.json

**Option B: mdxnet-infer (OpenMIRLab, 0.1.0, 2026-07-12, MIT code, Python >= 3.10, torch >= 2.0)** https://pypi.org/project/mdxnet-infer/
```powershell
pip install mdxnet-infer
mdxnet-infer drums.wav -o output/ --model drumsep-6stem --device cuda [--combine-cymbals] [--cache-dir DIR]
```
```python
from mdxnet_infer import separate, MDX23CInference
paths = separate("drums.wav", output_dir="out/", model_name="drumsep-6stem")
engine = MDX23CInference.from_pretrained("drumsep-6stem", device="cuda")
stems = engine.separate(audio_T, sample_rate=sr)   # kick, snare, toms, hh, ride, crash @ 44.1k stereo
```
Weights auto-download from `github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/`, sha256-verified; cache `--cache-dir` > `MDXNET_INFER_CACHE_DIR` > `~/.cache/mdxnet-infer/`. `drumsep-5stem` no longer auto-downloads (`MDX23CConfig.drumsep_5stem()` remains) https://github.com/openmirlab/mdxnet-infer

**Option C: inagoy DrumSep via Demucs.** The repo's scripts are Linux bash: `pip install demucs gdown`, `gdown 1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC` into `./model/`, then `demucs --repo "model" -o OUT -n 49469ca8 IN` (4 stems: Bombo/kick, Redoblante/snare, Platillos/cymbals, Toms). No Windows instructions; Demucs itself installs fine on Windows so the same command works from PowerShell if you place the `.th` in a folder and pass `--repo` https://github.com/inagoy/drumsep https://github.com/inagoy/drumsep/blob/main/drumsep https://github.com/inagoy/drumsep/blob/main/drumsepInstall . ZFTurbo hosts the same weights: https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.5/model_drumsep.th with https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_drumsep.yaml (hdemucs, 44.1 kHz stereo, 40 s chunks) https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_drumsep.yaml

**Option D: ZFTurbo MSST inference.py** (MIT) -- generic runner for any ckpt+config https://github.com/ZFTurbo/Music-Source-Separation-Training :
```powershell
python inference.py --model_type mdx23c --config_path CONFIG.yaml --start_check_point MODEL.ckpt --input_folder in/ --store_dir out/
```

### 2.3 License flags for an MIT project

- **Do not vendor or redistribute** the aufr33/jarredou MDX23C DrumSep weights: license undocumented; one HF mirror tags MIT (uploader-asserted), the OpenVINO conversion tags CC-BY-NC-SA-4.0; OpenMIRLab advises "non-commercial-safe only" https://pypi.org/project/mdxnet-infer/ https://github.com/openmirlab/mdxnet-infer
- LarsNet weights are CC BY-NC 4.0 (non-commercial) -- incompatible with MIT redistribution; StemGMD dataset is CC-BY 4.0 https://github.com/polimi-ispl/larsnet
- inagoy repo code is MIT but the README never states a weights license https://github.com/inagoy/drumsep
- Safe pattern: ship MIT code that **downloads weights at runtime** from their original hosts (as audio-separator / mdxnet-infer do) and document each model's license in the index.

---

## 3. Demucs (facebookresearch) and htdemucs_6s

- Repo archived 2025-01-01 (read-only); author left Meta; critical fixes only at https://github.com/adefossez/demucs ; license **MIT** https://github.com/facebookresearch/demucs
- Install `python -m pip install -U demucs` (Python >= 3.8); Python API = `demucs.separate.main(shlex.split("..."))` https://github.com/facebookresearch/demucs
- Models (`-n`): `htdemucs` (default; MusDB + 800 songs), `htdemucs_ft` (fine-tuned; ~4x slower, "somewhat better"), `htdemucs_6s` (adds guitar, piano), `hdemucs_mmi`, `mdx`, `mdx_extra`, `mdx_q`, `mdx_extra_q` https://github.com/facebookresearch/demucs
- 6s caveat (README): guitar "okay quality" but with bleed/artifacts; piano "a lot of bleeding and artifacts", "not working well" https://github.com/facebookresearch/demucs . Community ONNX port notes `other` SDR drops (~5.5 dB) because the model also predicts guitar+piano https://huggingface.co/adowu/htdemucs-6s-onnx
- Memory: >= 3 GB VRAM, default args ~7 GB; `--segment` max 7.8 s for HT models; `--shifts N` is N times slower; `--overlap` default 0.25 (0.1 for speed); CPU ~1.5x track duration https://github.com/facebookresearch/demucs
- Weights: `ROOT_URL = https://dl.fbaipublicfiles.com/demucs/`, fetched with `torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)` -> cached by torch.hub (default `%USERPROFILE%\.cache\torch\hub\checkpoints` unless `TORCH_HOME` set) https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/pretrained.py https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/repo.py
- ZFTurbo's SDR (MUSDB test): htdemucs 4-stem drums 10.88, bass 11.76, vocals 8.24 (Multisong set); 6-stem drums 10.22; htdemucs_ft drums 11.13 https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md
- **audio-separator exposes all four** (`htdemucs.yaml`, `htdemucs_ft.yaml`, `htdemucs_6s.yaml`, `hdemucs_mmi.yaml`) and `output_single_stem` accepts `Guitar`/`Piano`, so no direct `demucs` dependency is needed https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md . Demucs is not available on DirectML https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md

---

## 4. Alternatives

| Project | Status | Stems | License | Verdict |
|---|---|---|---|---|
| Spleeter (deezer) | Last PyPI 2.4.2, 2025-04-03; requires Python >=3.8,<3.12 (no 3.12); TensorFlow; dedicated GPU package dropped in 2.1.0 | 2 / 4 / 5 (piano) | Code MIT; model license unstated | Skip (Python 3.12 incompatible, outdated quality) https://pypi.org/project/spleeter/ https://github.com/deezer/spleeter |
| Open-Unmix (sigsep) | Last news 2024-04; torch 2.0 level | vocals/drums/bass/other; `umxl` drums 7.15 SDR | Code MIT; **umxl weights CC BY-NC-SA 4.0**; umxhq/umx MUSDB-trained | Skip (lower SDR than Demucs, NC weights) https://github.com/sigsep/open-unmix-pytorch |
| ZFTurbo Music-Source-Separation-Training | Active; MIT; 17 archs (mdx23c, htdemucs, bs_roformer, mel_band_roformer, scnet, ...) | Best public 4-stem: SCNet XL IHF avg 10.08 SDR (drums 11.81); BS-RoFormer 4-stem drums 11.61 | MIT code; weights per-model | Use for advanced 4-stem later (SCNet XL) https://github.com/ZFTurbo/Music-Source-Separation-Training https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md |
| openmirlab/bs-roformer-infer | Git-install only; MIT code; 34-model registry; default BS-RoFormer-SW 6-stem (vocals/drums/bass/guitar/piano/other) | 6 | Default SW weights license "unknown" -- not a grant for commercial use | Interesting 6-stem alternative to htdemucs_6s, but weights provenance unverified https://github.com/openmirlab/bs-roformer-infer |
| openmirlab/melband-roformer-infer | Git-install; 99 registry models, no drum category | 2-stem mostly | MIT code; weights mostly undeclared | Not needed https://github.com/openmirlab/melband-roformer-infer |
| MVSep | Commercial web service hosting the best DrumSep models; weights not public | -- | -- | Reference benchmark only https://mvsep.com/algorithms/29?lang=en |
| Moises | Commercial | -- | -- | Skip (per brief) |

Public 4-stem SCNet XL IHF: config https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml , ckpt https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt (license not stated on page) https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md

---

## 5. PyTorch + ONNX Runtime for RTX 5090 on Windows (as of 2026-10)

**Facts**
- Blackwell (sm_120) support arrived in PyTorch 2.7.0 with CUDA 12.8 wheels; blog command `pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128` https://pytorch.org/blog/pytorch-2-7/
- Latest torch: **2.14.1** (2026-09-30), Python 3.10-3.14 https://pypi.org/project/torch/ . 2.13 and 2.14 ship CUDA **12.6, 13.0, 13.2** wheels; **default PyPI wheel = CUDA 13.0** https://pytorch.org/blog/pytorch-2-14-release-blog/ . CUDA 12.6 wheels cannot target sm_120 (needs >= 12.8), so on 2.13+/5090 use cu130 or cu132. cu128 wheels exist up to 2.12.x https://pytorch.org/get-started/previous-versions/
- CUDA 13 runtime needs an R580+ driver (audio-separator contributor-lock note) https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md
- onnxruntime-gpu latest **1.31.0** (2026-10-09), requires Python >= 3.11 https://pypi.org/project/onnxruntime-gpu/ . From 1.27 the PyPI package is built with CUDA 13.0; separate CUDA 12.x packages exist for 1.27-1.30 (built with 12.8, need >= 12.8) https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html . Third-party changelog says 1.27 marks CUDA 12 packages deprecated https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.27.0 (unofficial mirror)
- cuDNN 8.x and 9.x ORT builds are not interchangeable; all current builds use cuDNN 9 https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- Windows: ORT needs VC++ 2019+ runtime; CUDA `bin` and cuDNN `bin` on PATH if not using torch's bundled DLLs; cuDNN 9 statically links zlib https://onnxruntime.ai/docs/install/
- `pip install onnxruntime-gpu[cuda,cudnn]` pulls the NVIDIA runtime wheels so no system CUDA Toolkit install is required https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ORT 1.31 release notes mention Windows support for Blackwell SM120 in the CUDA plugin EP (QMoE kernels) -- i.e. sm_120 is a known/tested target https://github.com/microsoft/onnxruntime/releases
- The ORT install page still says "default CUDA is 12.x since 1.19" -- stale vs the EP page; trust the EP page/table https://onnxruntime.ai/docs/install/

**Recommended pairing (both CUDA 13.0, cuDNN 9):**
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130   # or plain `pip install torch` (default = cu130)
pip install "audio-separator[gpu]"                                                # onnxruntime-gpu >= 1.27 -> CUDA 13.0
```
Fallback pairing on CUDA 12.8: `torch==2.12.1 --index-url https://download.pytorch.org/whl/cu128` https://pytorch.org/get-started/previous-versions/ + ORT CUDA 12 package (consult https://onnxruntime.ai/docs/install/ for the current CUDA-12 feed; the stable feed URL was not shown in docs fetched).

Which models actually use ORT: only the legacy `.onnx` MDX-Net and VR models. Demucs, MDX23C and RoFormer checkpoints run through torch https://huggingface.co/raydotac/AudioSeparatorONNX (community note) and `separator.py`'s MDXC/Demucs paths https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py . The v1 chain below is torch-only, so an ORT CUDA mismatch would not block it.

---

## 6. Two-pass strategy, performance, quality caveats

**Pipeline:** full mix -> `htdemucs_ft` (vocals/drums/bass/other) -> drums stem -> `MDX23C-DrumSep` (kick/snare/toms/hh/ride/crash; optionally merge ride+crash -> cymbals as mdxnet-infer's `--combine-cymbals` does) https://github.com/openmirlab/mdxnet-infer . MVSep's own "Drums" model also runs on the drum stem after vocals removal, confirming the cascade approach https://mvsep.com/algorithms/29?lang=en

**Time (no independent benchmark found; vendor numbers):** htdemucs ~20 s and htdemucs_ft ~25 s per 4-minute song on an RTX 3080 per a stem-service blog https://stemsplit.io/blog/spleeter-vs-demucs ; Demucs README: htdemucs_ft is ~4x the compute of htdemucs (bag of 4 models), CPU ~1.5x realtime https://github.com/facebookresearch/demucs . Expect well under 30 s per pass on a 5090; measure on a fixed test file.

**Memory:** Demucs default ~7 GB VRAM https://github.com/facebookresearch/demucs ; RoFormer VRAM scales with `chunk_size` (community guide: ~8 GB card handles 8-11 s chunks) https://huggingface.co/raydotac/AudioSeparatorONNX . 32 GB is ample; leave `segment_size` defaults. `--chunk_duration` only for >1 h files (hard concatenation, no crossfade) https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md

**Quality caveats for sample chopping**
- Separated stems are estimates, not the original multitracks; expect bleed. Hi-hat / cymbal stems show "seasick compression"/ducking against kick and snare on busy acoustic material; vocal spill survives in overheads; tom quality depends on how boomy the kick is (Sound on Sound review of a commercial drum separator) https://www.soundonsound.com/reviews/acon-digital-drum-production-suite
- Drum-part SDRs are low for metals: MDX23C DrumSep hh 3.19 / cymbals 6.08 vs kick 14.54 https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md -- kick and snare one-shots will be usable; hats/cymbals will smear and need gating or onset-based trimming.
- htdemucs_6s piano stem is unreliable; guitar okay with bleed https://github.com/facebookresearch/demucs
- `demucs_params["shifts"]` default 2 in audio-separator; more shifts = slower but cleaner https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md
- `normalization_threshold=0.9` applies peak normalisation to outputs; set `1.0`/disable if you want untouched levels for sample consistency https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py

---

## 7. Recommendations for v1

1. **Single dependency for separation: `audio-separator[gpu]` 0.47.x** (MIT). Covers 4-stem, 6-stem and the 6-part drum model through one `Separator` API, handles model download/caching, and runs on torch for every model in the chain https://github.com/nomadkaraoke/python-audio-separator
2. **Default chain:** pass 1 `htdemucs_ft.yaml` (best public Demucs drums SDR 11.13) https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md ; pass 2 `MDX23C-DrumSep-aufr33-jarredou.ckpt` on the Drums stem. Offer `--stems 6` flag switching pass 1 to `htdemucs_6s.yaml` with a warning that piano is low quality https://github.com/facebookresearch/demucs . Offer `--fast` using `htdemucs.yaml`.
3. **Install docs:** torch cu130 wheel first, then audio-separator; verify `sm_120` in `torch.cuda.get_arch_list()` and `audio-separator --env_info` https://pytorch.org/blog/pytorch-2-14-release-blog/ https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099 . Pin `torch>=2.7` (first sm_120 release) https://pytorch.org/blog/pytorch-2-7/ and `onnxruntime-gpu>=1.27` to keep both on CUDA 13 https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
4. **Always set `model_file_dir`** to a per-user app folder (avoid the literal `/tmp/...` default on Windows) and expose `AUDIO_SEPARATOR_MODEL_DIR` https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/separator.py
5. **Licensing:** project code MIT; do not bundle weights. Record per-model license in the exported index (`htdemucs*`: MIT Demucs release https://github.com/facebookresearch/demucs ; `MDX23C-DrumSep`: undocumented / treat non-commercial https://pypi.org/project/mdxnet-infer/ ). Print a notice when the drum model is used. Skip LarsNet (CC BY-NC) and Open-Unmix umxl (CC BY-NC-SA) https://github.com/polimi-ispl/larsnet https://github.com/sigsep/open-unmix-pytorch
6. **Output:** `output_format="WAV"`, `sample_rate=44100`, keep `output_single_stem=None` for pass 1 (need all 4), pass 2 writes all 6 drum parts then optionally sum ride+crash into `cymbals` for a 4-folder kick/snare/toms/cymbals layout https://github.com/openmirlab/mdxnet-infer
7. **Chopper heuristics:** trust kick/snare/toms one-shots; treat hh/ride/crash as "needs review" (low SDR, smearing) https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md https://www.soundonsound.com/reviews/acon-digital-drum-production-suite ; run onset detection on the separated part, not the full drum stem.
8. **v2 candidates:** SCNet XL IHF 4-stem via ZFTurbo `inference.py` (avg SDR 10.08 vs htdemucs 9.16) https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md ; `use_autocast=True` + `use_torch_compile=True` for batch jobs https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/README.md ; watch for a licensed public MelBand RoFormer DrumSep release (currently MVSep-only) https://mvsep.com/algorithms/29?lang=en
