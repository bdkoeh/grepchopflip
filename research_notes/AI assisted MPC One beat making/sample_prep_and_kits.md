# AI / automated sample prep and kit building for the Akai MPC One (state as of 2026-10-24)

Scope note: per the coordinator's mid-task update, OPEN-SOURCE / free tools get the most depth; commercial tools are kept brief. Host assumption: Windows 11 desktop + original MPC One (not One+/G2). Items that are from 2023–2024 are flagged as such. Several GitHub pages were fetched; "last commit date" is often not exposed by GitHub's rendered page to the fetch tool, so commit/star counts are given instead and dates are listed as gaps where unknown.

---

## Key Question 0 (context): Does the original MPC One run current MPC 3.x firmware, and what does Akai's own software automate?

### Takeaway
Yes. The original MPC One is on the supported-hardware list for every MPC 3.x firmware found, including MPC 3.8 (April 30, 2026). Akai's own pipeline already automates threshold/BPM/region chopping -> drum program conversion, and (paid, $9.99) 4-stem separation in standalone; the MPC 3 *desktop* software is still labeled beta. No evidence found of any Akai "AI auto-tagging" or "auto kit builder" feature.

### Cited Findings
- MPC 3.8 release notes (PDF created 2026-04-30): "This release is for MPC XL, MPC Live III, MPC Key 37, MPC One+, MPC X Special Edition, MPC Key 61, MPC One, MPC Live II, MPC X, MPC X SE, MPC Live and Force (standalone only)." — [Akai MPC 3.8 Release Notes PDF](https://cdn.inmusicbrands.com/Software/AM20P3C8FWUPGZL12/MPC%203.8%20Release%20Notes.pdf)
- Same notes: "The MPC 3 desktop software remains in beta... projects created in Standalone Mode on version 3.4 to version 3.8 are not backward-compatible with the MPC 2.15 desktop software." — [Akai MPC 3.8 Release Notes PDF](https://cdn.inmusicbrands.com/Software/AM20P3C8FWUPGZL12/MPC%203.8%20Release%20Notes.pdf)
- MPC 3.8 new features: five new effects from "MPC Sample" (Color Compressor, Trigger FX, Vintage, Vinyl Emulation, Tape Emulation; activated via Preferences > Activations); "Export as MPC Sample File" for drum tracks from desktop and standalone; MPC can now load MPC Sample projects. Bug fix listed under "Stems": converting slices of a stemmed sample to new samples no longer causes a Save failure (confirms stem-slice-to-sample workflow exists in 3.8). Known issue: "Sample Edit slices are unavailable in Trim mode until slice linking is toggled." — [Akai MPC 3.8 Release Notes PDF](https://cdn.inmusicbrands.com/Software/AM20P3C8FWUPGZL12/MPC%203.8%20Release%20Notes.pdf)
- MPC 3.5 desktop beta release notes list MPC One among supported hardware; MPC 3 installs alongside MPC 2; MPC 3 projects won't open in MPC 2; desktop activation moved from iLok to inMusic Software Center. — [MPC Desktop Beta 3.5 Release Notes](https://cdn.inmusicbrands.com/Software/ACVS-3.5.0%20Public%20Beta/MPC%20Desktop%20Beta%203.5%20Release%20Notes.pdf)
- MPC 3 desktop public beta on Windows: requires Windows 10 (21H2) through Windows 11 (25H2), 4 GB RAM (8 GB recommended), licensed via inMusic Profile + inMusic Software Center. — [Akai MPC 3 Desktop Public Beta FAQ](https://support.akaipro.com/en/support/solutions/articles/69000869156-akai-pro-mpc-3-desktop-public-beta-faq)
- Forum reports (March 2026) say MPC 3 desktop still lacks arranger mode and that controller mode is "largely unusable"; a May 2026 forum user recommends staying on 2.xx firmware for 32-bit units (One/One+) and notes they lack "pad quadrants" — these are user opinions, not Akai statements. — [MPC Forums (Mar 2026)](https://www.mpc-forums.com/viewtopic.php?p=1900325); [MPC Forums (May 2026)](https://www.mpc-forums.com/viewtopic.php?p=1903884)
- A newer model, MPC One G2, exists (ships with MPC software 3.9; 4 GB RAM, 64 GB storage, 8-core CPU, 256 voices, vs. One+ 2 GB / 16 GB / 64 voices; Akai says "4x the power of the currently available MPC One"). Stems Separation is "sold separately." — [Akai MPC One G2 FAQ](https://support.akaipro.com/en/support/solutions/articles/69000879145-mpc-one-g2-frequently-asked-questions)
- Akai's standalone chop workflow: load sample -> Sample Edit -> Chop tab -> Threshold auto-chops and assigns segments to pads -> Shift+Convert -> "New Drum Track Using Slices". — [Akai: Convert Your Chopped Samples into a Drum Program](https://support.akaipro.com/support/solutions/articles/69000859300-mpc-series-convert-your-chopped-samples-into-a-drum-program)
- MPC Stems (standalone stem separation): 4 stems (vocals, bass, drums, "music"/other), each with semitone/fine/level/pan; $9.99 covers both hardware and software; supported on MPC Live series, MPC One, MPC One+, MPC X/XSE, Key 61/37, Studio Mk2 (not Force at launch); announced July 2024 (DATED: 2024). — [Synth Anatomy, July 2024](https://synthanatomy.com/2024/07/akai-pro-brings-stems-separation-to-its-mpc-hardware-and-software.html); [MusicRadar headline](https://www.musicradar.com/news/akai-mpc-standalone-stem-separation)
- MPC 2.15.1 (2024, DATED) contained "stem separation improvements" per forum thread. — [MPC Forums](https://www.mpc-forums.com/viewtopic.php?f=48&t=218300)
- .xpn expansions must be installed in the desktop software first, then File > Export > Expansion... to the MPC's drive. — [Akai: How to Install .xpn Expansions](https://support.akaipro.com/en/support/solutions/articles/69000826395-akai-pro-mpc-how-to-install-xpn-expansions)
- Akai's program doc: drum programs are 16-pad kits stored as .xpm; keygroup programs are multisamples, also .xpm; samples are expected next to the program file. — [Akai: Understanding and Loading Programs](https://support.akaipro.com/en/support/solutions/articles/69000804211-akai-pro-mpc-series-understanding-and-loading-programs)
- Splice integration in MPC: 2.11.5 added Splice "By BPM"/"By Key" subfolders in the browser; MPC Desktop 3.6 beta fixed Splice folders on external media showing wrong sizes/dates; a Jan 2025 user said Splice integration was broken on 3.4.1. — [MPC 2.11.5 notes](https://support.akaipro.com/en/support/solutions/articles/69000824943-mpc-v2-11-5-release-notes); [MPC Desktop 3.6 Beta Improvements](https://cdn.inmusicbrands.com/Software/36/MPC%20Desktop%203.6%20Beta%20Improvements.pdf); [MPC Forums](https://www.mpc-forums.com/viewtopic.php?p=1889169)
- A May 2026 KVR thread says Akai updated the MPC Program File Format and that a JSON variant of XPM now exists (not independently verified). — [KVR thread](https://www.kvraudio.com/forum/viewtopic.php?p=9252901)

### Inferences
- For an original MPC One, staying on current MPC 3.8 firmware is officially supported; the practical risk is RAM (2 GB) and the desktop beta's instability, not feature exclusion.
- The cheapest "AI" step is Akai's own $9.99 MPC Stems; everything else (auto-tagging, kit building) must be done on the PC and transferred via SD/USB as WAV + .xpm.
- If Akai has introduced a JSON XPM variant (KVR report), any third-party generator should target the classic XML .xpm, which 3.x still loads.

### Gaps
- Could not fetch the akaipro.com release-notes index page (content not returned) or the MPC 3.9 notes; the newest confirmed-by-document firmware is 3.8 (April 2026), with 3.9 only mentioned as shipping on the One G2.
- No official statement found of any MPC 3.x feature excluded on the original MPC One; no official "AI" feature (beyond stems) found in any release note.
- Did not confirm whether MPC Stems processing time or quality differs on MPC One vs. newer units.

---

## Key Question 1: Which tools auto-detect transients / chop loops into slices and export to MPC-friendly formats?

### Takeaway
No tool was found that both auto-chops and writes a drum .xpm directly except Akai's own MPC software and small community apps (Kit Creator, Kit Maker, MPC Sample Kit Creator). The robust open-source path is: librosa (onset detection, well maintained, Windows OK) -> write WAV slices -> generate .xpm XML with a small script (community scripts exist but are thin).

### Cited Findings — open source / free
- librosa 0.11.0 released March 2025; metadata lists Python 3.12 support; pure-Python/NumPy so it installs on Windows via pip. — [librosa on PyPI mirror](https://simple-repository.app.cern.ch/project/librosa)
- madmom: code BSD; "model and data files" are CC BY-NC-SA 4.0 (commercial use requires contacting Gerhard Widmer); README still says Python 2.7 / 3.5+; RNN onset detection and DBN beat tracking included; no Windows-specific instructions; last PyPI release 0.16.1 dates to Nov 2018 per piwheels. — [madmom GitHub](https://github.com/CPJKU/madmom); [piwheels madmom](https://www.piwheels.org/project/madmom)
- Essentia on PyPI: latest is pre-release 2.1b6.dev1438 (May 19, 2026); wheels only for CPython 3.14 on Linux x86-64 and macOS 15+ (x86-64/ARM64); no Windows wheel and no sdist listed; license AGPL-3.0. — [Essentia on PyPI](https://pypi.org/project/essentia/)
- Audacity-family note: the SMPL2WAV tool (Aug 2026, Sonicstate) does silence-based auto-slicing/trimming of long recordings into one-shots on PC/Mac — not confirmed open source, pricing not found. — [Sonicstate, Aug 2026](https://sonicstate.com/news/2026/08/10/-hardware-sample-cd-extractor)
- Akai MPC (hardware + desktop): Threshold chop auto-assigns slices to pads; Shift+Convert -> "New Drum Track Using Slices". — [Akai support](https://support.akaipro.com/support/solutions/articles/69000859300-mpc-series-convert-your-chopped-samples-into-a-drum-program)
- A forum user notes exporting with the "pad parameters" option sets slice regions as pad-layer start/end so later region edits don't affect existing programs. — [MPC Forums](https://www.mpc-forums.com/viewtopic.php?p=1533216)

### Cited Findings — commercial (brief)
- Serato Sample 2.x: Serato staff say slices cannot be exported directly; must bounce via DAW (thread undated, pre-2.0); 2.0 added stems (Sept 2023, DATED); 2.2.0 (Dec 2025) added per-stem levels/automation and SLAB hardware control — no export feature mentioned. — [Serato forum](https://serato.com/forum/discussion/1627776); [MusicRadar on Sample 2.0](https://www.musicradar.com/news/serato-sample-2-stem-separation); [Mixdown on 2.2.0](https://mixdownmag.com.au/news/serato-sample-2-2-0-brings-stem-automation-and-native-slab-controller-integration)
- Algonaut Atlas 2: $99 (upgrade from v1 $19), 14-day trial, Win/Mac/Linux VST3/AU; ML "sonic map" of one-shots, designed for one-shots not loops; can export sequenced loop as audio. — [Gearnews](https://www.gearnews.com/algonaut-atlas-2-ai-driven-sample-manager-and-beat-creator-gets-bigger-and-better); [MusicTech review](https://musictech.net/reviews/software-instruments/algonaut-atlas-2-review/); [Sound on Sound](https://www.soundonsound.com/news/algonaut-update-atlas-ai-powered-drum-sample-manager)
- XLN Audio XO: ~$89–149 depending on retailer (price tracking from April 2026); stores samples in .xpak but "exporting samples from a kit is straightforward" and can export individual sample stems. — [Equipboard XO](https://equipboard.com/items/xln-audio-xo)
- Koala Sampler: mobile app ($4.99 base on iOS; up to ~$10 + free tier), auto-chop up to 32 slices — not a Windows desktop tool. — [App Store](https://apps.apple.com/app/id1021510437); [AlternativeTo](https://alternativeto.net/software/koala-sampler/about/)
- Splice Create (2023, DATED): assembles "Stacks" of up to eight library loops; download originals or timestretched stems (subscribers only). — [MusicRadar](https://musicradar.com/news/splice-create-stacks)

### Inferences
- On Windows 11 with Python 3.12, librosa is the only one of librosa/madmom/aubio/Essentia confirmed to install cleanly from wheels; Essentia has no Windows wheel at all and madmom's models are non-commercial.
- A Claude-Code-driven script can realistically do: `librosa.onset.onset_detect` -> slice with soundfile -> normalize/trim -> write 16 WAVs -> emit .xpm XML -> copy to SD card. The XPM writer is the only piece without a maintained library.

### Gaps
- aubio PyPI page failed to load; could not verify Windows wheels for Python 3.12.
- Sononym, ADSR Sample Manager, Loopcloud, Samplab, Chop Suey, Phase Plant: searches returned nothing useful (de-prioritized per coordinator's open-source focus); current pricing/features unverified.
- Serato Sample export behavior post-2.0 not confirmed against Serato's own release notes.

---

## Key Question 2: Which stem separation tools work well, and does MPC 3 have built-in stem separation?

### Takeaway
Yes — MPC standalone/desktop has had 4-stem "MPC Stems" as a $9.99 add-on since mid-2024 and 3.8 still references stems; for drum-kit prep the stronger open-source path is python-audio-separator (MIT, pip, CUDA/CPU/DirectML) feeding a drum-only stem into DrumSep (MIT, kick/snare/toms/cymbals). Demucs itself is archived but still installable; Spleeter is older (TensorFlow) and effectively superseded.

### Cited Findings — open source
- Demucs (facebookresearch): MIT; repository archived Jan 1, 2025 ("not maintained anymore"; bug fixes only at adefossez/demucs fork); models htdemucs (default), htdemucs_ft (~4x slower, slightly better), htdemucs_6s (adds guitar/piano, piano "not working great"); `pip install -U demucs`; Windows: use `python.exe` from Anaconda console, torchaudio limited on Windows so ffmpeg needed; GPU needs >=3 GB (default ~7 GB), `--segment` reduces memory, `-d cpu` ~1.5x track length; output 44.1 kHz stereo WAV stems drums/bass/other/vocals; no kick/snare split. — [Demucs GitHub](https://github.com/facebookresearch/demucs)
- python-audio-separator (nomadkaraoke): MIT; wraps UVR-trained models — MDX-Net (.onnx), VR (.pth), Demucs (.yaml, 4/6 stems), MDXC/RoFormer (.ckpt; default `model_bs_roformer_ep_317_sdr_12.9755.ckpt`); `pip install "audio-separator[gpu]"` (CUDA 11.8/12.2), `[cpu]`, or experimental `[dml]` for Windows AMD/Intel (Demucs unsupported under DirectML); Python 3.10+ (not 3.14.1); needs ffmpeg; CLI `audio-separator file.wav --model_filename ...`; 388 commits, 1.4k stars; no drumsep model bundled. — [python-audio-separator GitHub](https://github.com/nomadkaraoke/python-audio-separator)
- UVR5 GUI (Anjok07/ultimatevocalremovergui): MIT; latest release v5.6 with Windows installer `UVR_v5.6.0_setup.exe` (Win10+, must install to C:\), separate DirectML build for AMD/Intel Arc; Nvidia GTX 1060 6 GB minimum, 8 GB+ recommended; models Demucs v3/v4, MDX-Net, VR, MDX23C; dev update dated Oct 6, 2026 (Issue 2399); 26.8k stars; no documented CLI. — [UVR GitHub](https://github.com/Anjok07/ultimatevocalremovergui)
- DrumSep (inagoy/drumsep): MIT; Hybrid Demucs model splitting a drum recording into Kick, Snare, Cymbals, Toms; install via `drumsepInstall` script then `drumsep "<in>" "<out>"` (file or directory); Linux + Colab documented only, no Windows notes; 150 stars, 63 commits; mirrored on Hugging Face. — [DrumSep GitHub](https://github.com/inagoy/drumsep); [HF mirror](https://huggingface.co/vincewin/drumsep)
- Newer drum-part models (MDX23C 6-stem by aufr33/jarredou, SCNet XL 4/5/6-stem, Mel-Band RoFormer) are hosted on MVSep and "work only on a drum-only track"; the `mdxnet-infer` PyPI package offers 6-stem (kick, snare, toms, hi-hat, ride, crash) inference. — [MVSep DrumSep page](https://mvsep.com/algorithms/29); [mdxnet-infer on PyPI](https://pypi.org/project/mdxnet-infer/)
- Spleeter (deezer): MIT; not archived; references 2.1.0 release; 2/4/5-stem models; `pip install spleeter` plus ffmpeg/libsndfile; Windows fallback `python -m spleeter separate`; 28.5k stars, 246 open issues. — [Spleeter GitHub](https://github.com/deezer/spleeter)
- Third-party 2026 roundups rate UVR5 ensemble mode as matching or beating paid services; LALAL.AI requires paid plan to download; Moises free tier 5 tracks/month; RipX DAW ~$179–198 one-time (secondary sources, vendor-affiliated). — [Rysup Audio 2026](https://rysupaudio.com/blogs/news/best-free-stem-separators-2026); [MixingGPT 2026](https://mixinggpt.com/blog/best-ai-stem-separation-tools-2026); [StemSplit](https://stemsplit.io/blog/best-vocal-remover-tools)

### Cited Findings — Akai built-in
- MPC Stems: 4 stems, $9.99 one-time for hardware+software, MPC One supported (July 2024, DATED). — [Synth Anatomy](https://synthanatomy.com/2024/07/akai-pro-brings-stems-separation-to-its-mpc-hardware-and-software.html)
- MPC 3.8 bug fix under "Stems" confirms stems workflow persists in 3.x. — [MPC 3.8 notes](https://cdn.inmusicbrands.com/Software/AM20P3C8FWUPGZL12/MPC%203.8%20Release%20Notes.pdf)

### Inferences
- Best Windows 11 open-source chain: `audio-separator` (BS-RoFormer or htdemucs) -> `drums.wav` -> DrumSep/mdxnet-infer -> per-piece stems -> onset slicing -> one-shots. DrumSep's install script is Linux-oriented, so on Windows expect to install demucs manually and load its checkpoint, or use WSL2.
- Akai's on-device stems are fine for loops but give only a single "drums" stem; kick/snare isolation still needs the PC.

### Gaps
- No head-to-head benchmark found comparing htdemucs vs. BS-RoFormer vs. MDX23C on drum material; MVSep publishes SDR tables but these were not fetched.
- DrumSep Windows installation not documented anywhere found.
- Could not fetch DJ Mag's standalone-stems article (403) to confirm processing-time figures.

---

## Key Question 3: Tools or scripts that generate .xpm drum/keygroup programs automatically from a folder of WAVs

### Takeaway
Open-source coverage is thin: BlackCursive/mpckitcreator (Python, builds an MPC program from WAVs in a directory, 9 stars, license not shown) and the Rust/WASM "mpc_valet" keygroup tool are the only direct generators found; ConvertWithMoss (LGPL-3, Java) is the most active open-source multisample converter (458 stars, 825 commits) but its MPC output support was not confirmed by the fetch. Commercial Kit Creator ($59.99 list) and Kit Maker ($35) do WAV-folder -> .xpm.

### Cited Findings — open source
- BlackCursive/mpckitcreator: "Tool to create an Akai MPC program with wave files - MPC Live 1/2, MPC X or MPC one"; Python 3.6+, deps SoundFile + BeautifulSoup (i.e., builds XML via bs4); `python3 mpckitcreator.py` builds a program from WAVs in the current directory; also `read_wav_files.py` and `read_smpl_mpckit.py`; 27 commits, 9 stars, 3 forks; no license file visible, no releases, pad-mapping rules undocumented. — [mpckitcreator GitHub](https://github.com/BlackCursive/mpckitcreator)
- mpc_valet (Rust crate / browser tool): builds keygroup .xpm programs, auto-detects root notes from sample names, assigns note ranges; author notes "created programs will always need fine tuning" and it is "quick&dirty." — [mpc_valet docs.rs](https://docs.rs/mpc_valet); [Rust Digger](https://rust-digger.code-maven.com/crates/mpc_valet)
- ConvertWithMoss (git-moss): LGPL-3.0; Java (Maven, jpackage scripts incl. `jpackage-win.cmd`); sources listed include WAV, SFZ, SF2, NKI, KMP, Wavestate; has `ConvertWithMossCLI.properties` and `run.cmd` (CLI implied); 458 stars, 825 commits. A May 2026 KVR thread names ConvertWithMoss v18.0 as handling MPC/Force conversion and says the author will support Akai's updated program format "in a later update." — [ConvertWithMoss GitHub](https://github.com/git-moss/ConvertWithMoss); [KVR thread](https://www.kvraudio.com/forum/viewtopic.php?p=9252901)
- Michaelvdlnz/mpc-preview-builder: MIT, Python 3.8+, Windows-only (uses winsound); scans a folder of .xpm drum programs and creates `[Previews]/ProgramName.xpm.wav` by matching WAV filenames (exact/substring/fuzzy); 9 commits, 0 stars. Useful last step for browsable kits. — [mpc-preview-builder GitHub](https://github.com/Michaelvdlnz/mpc-preview-builder)
- tarikcampos/MPC-Sample-Toolkit (MPCTK): MIT, Python 3.11+, CLI `mpctk generate` + PySide6 GUI; targets the Akai *MPC Sample* device's gzip .xpj projects (not .xpm); one WAV -> chromatic/scale pad layout; multi-bank and batch tooling "planned"; 41 commits, 3 stars. Relevant because MPC 3.8 can now load MPC Sample projects. — [MPC-Sample-Toolkit GitHub](https://github.com/tarikcampos/MPC-Sample-Toolkit); [MPC 3.8 notes](https://cdn.inmusicbrands.com/Software/AM20P3C8FWUPGZL12/MPC%203.8%20Release%20Notes.pdf)
- XPM structure as seen by community code: XML with `<Program>` -> `<Instruments>` -> `<Instrument>` per keygroup, parsed with `xml.etree.ElementTree`; a forum Python merge script works on MPC One 2.10 / Force 3.1.3, with no error checking. — [MPC Forums script](https://www.mpc-forums.com/viewtopic.php?p=1850940)
- liotier/AkaiMPC (Unlicense, vanilla JS, 458 commits): chord-progression tools plus ready-made MIDI .xpm programs for Roland gear — NOT a WAV-to-kit generator. — [AkaiMPC GitHub](https://github.com/liotier/AkaiMPC)
- GitHub topic pages exist for `akai-mpc`, `akaipro`, `mpc-x` for further mining. — [akai-mpc topic](https://github.com/topics/akai-mpc)

### Cited Findings — commercial (brief)
- Kit Creator (Homegrown Sounds): Win/Mac; auto-generates 128-sample kits; exports include MPC .xpm; priced $59.99 (intro $39.99 in Aug 2023, DATED). — [KVR](https://kvraudio.com/news/homegrown-sounds-releases-kit-creator-v1-0-58507); [KVR forum](https://www.kvraudio.com/forum/viewtopic.php?p=8611722)
- Kit Maker: $35 on Gumroad, Mac+PC, 2-computer license, makes presets for MPC Live/X/One and MPC 2 desktop. — [Kit Maker Gumroad](https://kitmaker.gumroad.com/l/kit-maker)
- MPC Sample Kit Creator (Ko-fi): one-click conversion of Maschine/Ableton Drum Rack libraries to 16-pad MPC kits, auto-categorizes kick/snare/hat, optional key search, custom XPM template support. — [Ko-fi listing](https://ko-fi.com/s/409e4308f4); [MPC Forums](https://www.mpc-forums.com/viewtopic.php?p=1869370)
- MPC Keygroup Creator (paid, Mac/Win): bulk keygroups from one sample, pitch detection from filenames. — [Gumroad](https://drewh.gumroad.com/l/mpc-keygroup)
- "MPC Pad Perfect": no product by this name found.

### Inferences
- A Claude-Code-driven pipeline is feasible: use mpckitcreator's `templates.py` as the XML skeleton reference (or dump a known-good .xpm saved from MPC 3.8 and treat it as a template), fill 16 `<Instrument>`/pad entries with sample names, write WAVs alongside, run mpc-preview-builder, copy to SD. Keep the classic XML .xpm until the reported JSON variant is understood.
- For keygroups (melodic one-shots stretched across pads), mpc_valet or ConvertWithMoss (if MPC output confirmed) is more reliable than hand-rolled XML.

### Gaps
- Could not confirm mpckitcreator's license, last commit date, or exact pad-assignment logic; its output format (.xpm) is implied but not stated on the page.
- ConvertWithMoss destination-format list (and whether "Akai MPC Keygroup" is included) was not shown in the fetched page; release version/date not shown.
- No official Akai XPM schema found; the May 2026 JSON-XPM report is unverified.
- mpc_valet GitHub repository URL not found via search (only docs.rs/crates listing).

---

## Key Question 4: Does the original MPC One support current MPC 3 firmware and features? Any limits?

### Takeaway
Officially supported through MPC 3.8 (April 2026) with no feature exclusions stated; practical limits are the One's 2 GB RAM / 32-bit-era hardware (forum anecdotes) and the MPC 3 desktop beta being incomplete. MPC Stems works on MPC One.

### Cited Findings
- MPC One listed in MPC 3.8 compatibility. — [MPC 3.8 notes](https://cdn.inmusicbrands.com/Software/AM20P3C8FWUPGZL12/MPC%203.8%20Release%20Notes.pdf)
- MPC One listed in MPC 3.4.1 and MPC 3.5 compatibility. — [MPC 3.4.1 notes](https://cdn.inmusicbrands.com/akai/MPC3-NI/MPC%203.4.1%20Firmware%20Update%20Release%20Notes.pdf); [MPC 3.5 notes](https://cdn.inmusicbrands.com/Software/ACVS-3.5.0%20Public%20Beta/MPC%20Desktop%20Beta%203.5%20Release%20Notes.pdf)
- MPC One supported by MPC Stems standalone. — [Synth Anatomy](https://synthanatomy.com/2024/07/akai-pro-brings-stems-separation-to-its-mpc-hardware-and-software.html)
- Forum user (May 2026): recommends 2.xx firmware for One/One+ and notes no pad quadrants on 32-bit units (opinion). — [MPC Forums](https://www.mpc-forums.com/viewtopic.php?p=1903884)
- One+ spec for reference: 2 GB RAM, 16 GB storage (7.8 GB user-available); One G2 doubles RAM, quadruples voices. — [Akai One G2 FAQ](https://support.akaipro.com/en/support/solutions/articles/69000879145-mpc-one-g2-frequently-asked-questions)

### Inferences
- Kits built on PC should keep sample counts/sizes modest (16-bit 44.1 kHz WAV, trimmed one-shots) because of the One's 2 GB RAM.

### Gaps
- No Akai document found listing MPC 3 features unavailable on original MPC One; the "32-bit" claim is forum-only.

---

## Key Question 5: AI sample generators producing usable drum one-shots/loops with clear licensing

### Takeaway
Open-weights options with commercial-friendly terms: Stable Audio Open 1.0 (47 s, 44.1 kHz stereo) and Stable Audio Open Small (11 s, ~0.5B params, CPU/Arm-friendly), both under the Stability AI Community License (free commercial use reported below US$1M revenue; confirm at stability.ai/license). MusicGen/AudioCraft weights are CC-BY-NC (non-commercial). AudioLDM2 license not displayed. No turnkey open "drum one-shot generator" checkpoint was found; research models (DrumGAN, CRASH, DOSE) lack public drum-trained weights. Commercial: ElevenLabs SFX royalty-free on paid plans; Splice's 2026 Variations/Craft are "licensed for commercial use."

### Cited Findings — open weights
- Stable Audio Open 1.0: Stability AI Community License ("For commercial use, please refer to https://stability.ai/license"); up to 47 s; 44.1 kHz stereo; trained on 486,492 CC0/CC BY/CC Sampling+ recordings (Freesound + FMA); runs via `stable-audio-tools` (`get_pretrained_model`, `generate_diffusion_cond`, steps=100, cfg_scale=7) or `diffusers` `StableAudioPipeline`; model card says it is "better at generating sound effects and field recordings than music"; example prompt "128 BPM tech house drum loop." — [HF stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- Stable Audio Open Small: same license; up to 11 s; 44.1 kHz stereo; ~0.5B params (F32/F16); Arm CPU optimization guide linked; paper "Fast Text-to-Audio Generation with Adversarial Post-Training" (May 13, 2025). — [HF stable-audio-open-small](https://huggingface.co/stabilityai/stable-audio-open-small)
- Community license terms reported as free commercial use for individuals/orgs under US$1,000,000 annual revenue; above that requires enterprise license; outputs may not train competing models (unofficial mirror + Stability explainer — vendor/secondary). — [HF unofficial mirror](https://huggingface.co/joeriben/stable-audio-open-small); [Stability explainer](https://stability.ai/explainers/stable-audio-vs-competitors-licensing-export-rights-and-self-hosting-compared)
- Older 2024 coverage says the original release was non-commercial research license — superseded (DATED 2024). — [VentureBeat 2024](https://venturebeat.com/ai/stability-ai-debuts-new-stable-audio-open-for-sound-design)
- Stable Audio 3.0 announced with open weights and up-to-six-minute tracks (The Decoder; details not fetched). — [The Decoder](https://the-decoder.com/stability-ai-launches-stable-audio-3-0-with-up-to-six-minute-tracks-and-open-weights/)
- AudioCraft/MusicGen: code MIT, weights CC-BY-NC 4.0 (non-commercial); trained on 20K hours licensed music incl. ShutterStock/Pond5. — [MusicGen README mirror](https://huggingface.co/spaces/civichacker/MusicGen/blob/main/README.md); [HF output-license discussion](https://huggingface.co/spaces/facebook/MusicGen/discussions/8)
- AudioLDM2: text-to-audio/SFX/music, TTS, 48 kHz checkpoint; install `pip3 install git+https://github.com/haoheliu/AudioLDM2.git` in Python 3.8 env; cpu/cuda/mps; ~2.6k stars, 86 commits; LICENSE contents not shown on page. — [AudioLDM2 GitHub](https://github.com/haoheliu/AudioLDM2)
- Drum-specific research: DrumGAN and CRASH trained on a non-public ~300k one-shot dataset; DOSE (2025) extracts kick/snare/hat one-shots from mixtures, no code release noted; a Charles University thesis trained a VAE on ~16k free samples in 9 categories. — [DrumGAN arXiv](https://arxiv.org/pdf/2008.12073); [CRASH arXiv](https://arxiv.org/pdf/2106.07431); [DOSE arXiv](https://arxiv.org/pdf/2504.18157); [CUNI thesis](https://dspace.cuni.cz/handle/20.500.11956/192871)

### Cited Findings — commercial (brief)
- ElevenLabs Sound Effects: "royalty free and can be used in commercial projects"; free plan requires attribution to elevenlabs.io, paid plans do not; SFX terms updated 12 Feb 2026; third-party guide cites paid plans from ~$6/month. — [ElevenLabs commercial SFX page](https://elevenlabs.io/sound-effects/commercial); [ElevenLabs SFX terms (PL)](https://elevenlabs.io/pl/sound-effects-terms); [Uravation guide, Aug 2026](https://uravation.com/media/elevenlabs-complete-guide-2026/)
- Splice (April 15, 2026): Variations, Craft, Magic Fit (Magic Fit "summer 2026") built on 3M+ licensed human-made samples; "All output from the tools is licensed for commercial use"; creators paid per use; Splice acquired Kits AI (Jan 2026) and partnered with ElevenLabs. — [Music Business Worldwide](https://www.musicbusinessworldwide.com/splice-launches-ai-tools-that-compensate-sample-creators/); [Billboard](https://www.billboard.com/pro/splice-partners-with-elevenlabs-create-new-ai-products/)
- Suno/Udio: secondary 2026 sources conflict; Suno commercial rights tied to paid tiers with download caps reported from Sept 3, 2026; Udio downloads reported suspended / personal-use-only by one source and commercial by another; neither confirmed to offer stems. Primary terms not fetched. — [Dubspot 2026](https://blog.dubspot.com/ai-music-licensing-explained-2026); [Dupple](https://dupple.com/learn/suno-vs-udio); [SimilarLabs](https://similarlabs.com/blog/best-ai-music-generators-2026)

### Inferences
- For an open pipeline, Stable Audio Open Small is the practical generator on a Windows 11 PC without a large GPU (0.5B params, 11 s is ample for one-shots); batch-prompt per drum type, then auto-trim via librosa onset/silence detection.
- MusicGen is unusable for commercial kits due to CC-BY-NC weights.

### Gaps
- Official Stability AI license page (stability.ai/license) not fetched; revenue threshold is from secondary sources.
- AudioLDM2 license (weights) not verified.
- Google MusicFX/Lyria, Output Co-Producer/Arcade, Soundraw, Soundful, Beatoven: no useful results found (de-prioritized per open-source focus).
- No ready-to-run open-source drum one-shot generator (kick/snare-specific) with public weights was found.

---

## Key Question 6 (coordinator addendum): Can an LLM-driven script chain these into "folder of WAVs -> ready MPC drum kit" on Windows 11?

### Takeaway
Yes, with one hand-written component (the .xpm writer). Every other stage has a pip-installable, MIT/BSD-licensed, Windows-compatible piece: audio-separator (stems) -> DrumSep/mdxnet-infer (drum parts) -> librosa + soundfile (onset slicing, trimming, normalization) -> XML .xpm generation (mpckitcreator as reference / template-from-real-xpm) -> mpc-preview-builder (.xpm.wav previews) -> copy to SD/USB -> load on MPC One.

### Cited Findings
- audio-separator: pip, MIT, Python 3.10+, CUDA/CPU/DirectML, CLI. — [GitHub](https://github.com/nomadkaraoke/python-audio-separator)
- DrumSep: MIT, directory input supported (`drumsep "<in>" "<out>"`). — [GitHub](https://github.com/inagoy/drumsep)
- librosa 0.11.0 (Mar 2025), Python 3.12 listed. — [PyPI mirror](https://simple-repository.app.cern.ch/project/librosa)
- mpckitcreator: Python + SoundFile + BeautifulSoup builds program from WAVs in cwd. — [GitHub](https://github.com/BlackCursive/mpckitcreator)
- mpc-preview-builder: MIT, Windows 10/11, no deps, generates `[Previews]` files. — [GitHub](https://github.com/Michaelvdlnz/mpc-preview-builder)
- Akai: programs and samples must be in the same folder when moved to hardware; .xpn requires desktop install + export. — [Akai programs doc](https://support.akaipro.com/en/support/solutions/articles/69000804211-akai-pro-mpc-series-understanding-and-loading-programs); [Akai .xpn doc](https://support.akaipro.com/en/support/solutions/articles/69000826395-akai-pro-mpc-how-to-install-xpn-expansions)
- Demucs on Windows needs ffmpeg; CPU fallback ~1.5x realtime. — [Demucs GitHub](https://github.com/facebookresearch/demucs)

### Inferences
- Skip .xpn packaging: generating plain folders of `Kit.xpm` + WAVs + `[Previews]` on the SD card gives the same "browse and play" result without the desktop software round-trip.
- Essentia (no Windows wheel, AGPL) and madmom (non-commercial models, stale) should be avoided in a Windows pipeline; librosa covers onset detection adequately.
- Optional auto-tagging (kick/snare/hat classification) has no verified open-source drop-in; a simple spectral-centroid/decay heuristic in librosa, or an LLM-supervised rule set, would be the pragmatic fallback.

### Gaps
- No existing repository was found that already implements the full chain end-to-end.
- Exact XPM pad-element schema (e.g., `ProgramPads`, layer start/end fields) not documented in any fetched source; must be reverse-engineered from a saved MPC 3.8 program.
- Not tested whether a hand-generated classic-XML .xpm loads on MPC 3.8 firmware (community merge script was last confirmed on MPC One 2.10 / Force 3.1.3).
