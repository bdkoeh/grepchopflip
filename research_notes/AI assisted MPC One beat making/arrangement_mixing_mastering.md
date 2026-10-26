# AI-assisted arrangement, performance capture, mixing and mastering for an MPC One + Windows 11 PC workflow (as of October 2026)

Scope note: per the coordinator's mid-task update, this file gives the most depth to open-source DAWs, LLM/MCP controllers for DAWs, and open-source mixing/mastering tools that an AI assistant (e.g. Claude Code on the PC) could drive while the MPC One is the instrument. Commercial AI tools are covered briefly for comparison. Research budget was limited to roughly 45 search/fetch calls; GitHub pages were fetched for the most relevant repos, but GitHub's HTML does not expose last-commit dates to the fetcher, so "last activity" is noted as a gap wherever it could not be confirmed.

## Key Question 1: Which open-source DAWs on Windows 11 could an AI assistant drive, and how do they compare with Reaper?

### Takeaway
Ardour 9 (GPL, Feb 2026) is the only mature FOSS DAW with a built-in, official LLM control surface ("MCPHttp", a Model Context Protocol server on port 4820) plus OSC and Lua; its Windows binaries are pay-what-you-want (US$1 minimum) or a free demo that goes silent periodically. Zrythm 2.0 and LMMS 1.3 are both still alpha. Reaper is not open source (proprietary, $60 discounted license) but has by far the richest scripting (ReaScript Lua/Python/EEL) and the most MCP servers.

### Cited Findings
**Ardour**
- Ardour 9.0 was released 5 February 2026 with downloads for Linux/Windows/macOS, multi-touch GUI on Windows and Linux, plug-in enhancements and new Lua scripting capabilities — [Phoronix](https://phoronix.com/news/ardour-9.0-daw)
- Ardour ships an experimental "MCP HTTP Surface": "MCP" here means Model Context Protocol, not Mackie Control. Default endpoint `http://127.0.0.1:4820/mcp`, HTTP transport (not SSE). Enable via Preferences > Control Surfaces > MCPHttp. The manual gives client setup for Codex CLI, Gemini CLI, Claude Desktop (via `mcp-remote`) and Copilot CLI — [Ardour manual: Using the MCP HTTP Surface](https://manual.ardour.org/using-control-surfaces/mcp-http/)
- Exposed tool areas: session (save, undo/redo, snapshots, mixer scenes); transport (locate, play, stop, loop, punch, record-enable, markers); tracks/buses (list, create, rename, fader, pan, mute, solo, rec-enable); sends; plugins (discovery, insert, remove, reorder, bypass, parameter control); markers/ranges incl. arrangement markers; regions (move, copy, resize, split, gain, normalize); MIDI (region creation, note insert/edit, bulk JSON import/export) — [Ardour manual](https://manual.ardour.org/using-control-surfaces/mcp-http/)
- Limitations stated in the manual: experimental and may change; most edits identify tracks/plugins/regions by explicit IDs or names which may not be unique; selected-region targeting not implemented; "listen on all interfaces" can expose control to the network, so snapshot/save before destructive edits — [Ardour manual](https://manual.ardour.org/using-control-surfaces/mcp-http/)
- Ardour's OSC surface: "Ardour is probably one of the most OSC-controllable audio applications around"; listens on UDP port 3819 by default once enabled — [Ardour manual: Controlling Ardour with OSC](https://git.ardour.org/ardour/manual/raw/commit/05ca69f6720ed6353d3f25511212e0fe2cf8fb6a/_manual/22_using-control-surfaces/01_controlling-ardour-with-osc.html)
- Community alternative: `ardour-mcp` by @pyroqbit, written in Rust, translates MCP requests into Ardour OSC commands; listed as supporting Ardour 6.x–9.x; covers transport, track mute/solo/rec/gain/trim and menu actions — [mcp.so listing](https://mcp.so/servers/ardour-mcp) (GitHub URL not surfaced by the listing; see Gaps)
- Windows pricing model (third-party summary, confirm at ardour.org): free demo build "periodically goes silent after ~10 minutes"; pay-what-you-want one-time payment with US$1 minimum; paying at least US$45 includes the next major version and nightly builds; subscriptions US$1/4/10/50 per month via PayPal; binaries for Windows/macOS/Linux with no seat limits; Ardour keeps working regardless of subscription status — [FitGap](https://us.fitgap.com/products/ardour); an Ardour developer forum reply says "the Windows version is complete, maintained and as robust as the other versions" and that the demo is free with a pick-your-own-price binary — [Ardour Discourse](https://discourse.ardour.org/t/is-there-ardour-support-in-2018-for-windows/100069)
- Building from GPL source on Windows is described by a forum user as hard, requiring ~832 cross-compiled packages — [Ardour Discourse](https://discourse.ardour.org/t/this-software-is-free/109152)
- OSC was listed as a key new feature back in Ardour 5.0 — [ardour.org 5.0 news](https://ardour.org/news/5.0.html)

**Zrythm**
- Zrythm 2.0 first alpha reported 31 May 2026; UI moved from GTK to Qt/QML; adds ASIO support on Windows — [Phoronix](https://www.phoronix.com/news/Zrythm-2.0-Alpha)
- Official site lists v2.0.0-alpha.3, 64-bit Windows installer plus experimental ARM64EC build; requires Windows 10 (1809+) or 11 and a CPU with AVX; warns "not suitable for production use" and that v2 projects are not compatible with v1 — [zrythm.org v2 alpha download](https://www.zrythm.org/en/download-v2-alpha.html)
- License: older coverage describes Zrythm as open source under AGPL with source on GitHub — [GameFromScratch](https://gamefromscratch.com/zrythm-hands-on-with-the-1-0-alpha-release-of-this-new-daw/) (no 2.0-specific license statement found)

**LMMS**
- Latest build found: LMMS 1.3.0-alpha.2, dated 6 September 2026 per Wikipedia; project says it is "considerably more polished than the typical nightly build" — [Wikipedia](https://en.wikipedia.org/wiki/LMMS); [AlternativeTo news](https://alternativeto.net/news/2026/9/lmms-releases-1-3-0-alpha-2-bringing-the-long-awaited-daw-s-update-closer-to-stable/)
- Release notes: Windows 10/11 x86_64 (x86-64-v2 CPU or newer) plus a Windows arm64 build requiring Windows 11; 32-bit dropped; projects saved in this alpha cannot be reopened in previous versions; adds detachable windows, microtonal tunings, native LinuxVST support, Tap Tempo — [newreleases.io mirror of GitHub v1.3.0-alpha.2](https://newreleases.io/project/github/LMMS/lmms/release/v1.3.0-alpha.2)

**Reaper (NOT open source)**
- Cockos official pricing: $60 discounted license (personal use, or revenue under USD $20,000/yr, or edu/non-profit); $225 commercial; free upgrades through v8.99; 60-day fully functional evaluation; purchases final — [reaper.fm purchase page](https://www.reaper.fm/purchase.php)
- Unlicensed use after 60 days only produces a short startup message; no functional restriction — [ProRec commentary](https://ProRec.com/?p=6209)
- Python ReaScript: `python-reapy` installs via pip; on Windows use `py -m pip install python-reapy`; manual route is REAPER Preferences > "Enable Python for use with ReaScript" and "Custom path to Python dll directory", then enable reapy's "dist API" via a ReaScript action; Python 2 unsupported — [python-reapy install guide](https://python-reapy.readthedocs.io/en/stable/install_guide.html); [PyPI python-reapy](https://pypi.org/project/python-reapy)
- Caveat: a separate project warns that Microsoft Store Python may not be callable from scripts launched inside REAPER; the python.org installer with "Add Python to PATH" avoids this — [ICST Ambisonics setup](https://ambisonics.ch/icst-ambisonics-plugins/16_motion_map_setup/)
- Retrospective MIDI recording: Reaper users describe that Reaper "will buffer all you play - and you can fetch it all back" on a track with a loaded instrument (Reaper 6.71-era report) — [Yamaha Musicians forum](https://yamahamusicians.com/forum/viewtopic.php?p=108256)

### Inferences
- For a Claude-Code-driven pipeline on Windows 11 with no budget, Ardour 9 is the only FOSS DAW with a first-party, documented MCP server; the US$1 binary or the demo (with silence interruptions) are the realistic install paths since a Windows source build is impractical.
- Zrythm 2 and LMMS 1.3 are not production-ready in October 2026 and have no known MCP/LLM bridges; they are not recommended as the recording host for this workflow.
- Reaper at $60 is the pragmatic "cheap but not FOSS" choice: three independent MCP servers (179, 60 and transport-level tool sets) plus Lua/Python ReaScript, and native retrospective MIDI record, make it the most automatable DAW on Windows.

### Gaps
- Could not confirm the GitHub URL, license or activity of the community `ardour-mcp` (Rust) project; only the mcp.so listing surfaced.
- Which Ardour version introduced MCPHttp is not stated in the manual page; Phoronix's 9.0 coverage did not mention it. It may be a 9.x point release or nightly feature — verify in Ardour 9.x release notes.
- Ardour's current pricing page (community.ardour.org/download) redirected and was not fetched; figures above are from a third-party summary.
- No Zrythm 2.0 license statement found; AGPL is from 1.0-era coverage.
- Whether Ardour 9's Windows build supports ASIO and how it handles the MPC One's USB audio was not researched.

## Key Question 2: Which LLM/MCP controllers exist for DAWs and audio environments, and what can they do from a script?

### Takeaway
As of October 2026 there are working, MIT-licensed MCP servers for Ableton Live (3.2k-star ahujasid/ableton-mcp and 280-star uisato/ableton-mcp-extended), REAPER (TwelveTake 179 tools incl. loudness/true-peak measurement; bonfire-systems/itsuzef 60 tools incl. mastering/analysis), Bitwig (WigAI in-process extension, Bitwig Control Deck, jxstanford WIP), Ardour (built-in MCPHttp), SuperCollider (several, OSC-based) and Pure Data (two). All run locally and most document Windows paths; none are affiliated with the DAW vendors.

### Cited Findings
**Ableton Live**
- `ahujasid/ableton-mcp`: MIT; 3.2k stars, 446 forks, 73 commits; Remote Script runs inside Live as a MIDI Remote Script and opens a JSON-over-TCP socket server; MCP server connects to it; version handshake `get_remote_script_info`. Capabilities: session/track info, create/modify MIDI and audio tracks, "build full song arrangements in Arrangement View", create/edit/trigger clips, add MIDI notes, playback/tempo, load instruments/effects from browser. Requires Live 10+ (installer needs 10.1.13+), Python 3.8+, `uv`. Windows path `C:\Users\[User]\Documents\Ableton\User Library\Remote Scripts\AbletonMCP\`. Limitations: complex arrangements need to be split into steps; designed for default devices; one MCP server instance at a time; timeouts; anonymous telemetry on by default — [GitHub ahujasid/ableton-mcp](https://github.com/ahujasid/ableton-mcp)
- `uisato/ableton-mcp-extended`: MIT; 280 stars, 63 forks, 77 commits; adds device parameter listing/setting (normalized 0–1), clip automation points ("not working perfectly yet"), batch MIDI edits/transpose/quantize, loop and follow-action settings, browser loading by URI/path, audio import, ElevenLabs TTS/SFX (separate server), hybrid TCP/UDP server for low-latency control, custom controller framework. Requires Live 11+, Python 3.10+, Claude Desktop/Cursor/Gemini CLI. Windows: Remote Scripts folder path given; Control Surface set to "AbletonMCP" with Input/Output None; UDP install command is Unix-style and must be adapted — [GitHub uisato/ableton-mcp-extended](https://github.com/uisato/ableton-mcp-extended)
- `adindotdev/ableton-mcp`: Node-based, pairs with the free AbletonOSC Remote Script; covers transport, tracks, clips, mixer, device parameters; registers with Claude Code via one command — [Glama listing](https://glama.ai/mcp/servers/@adindotdev/ableton-mcp)
- `itsuzef/ableton-mcp`: fork of the original focused on easier install incl. Smithery for Claude Desktop — [Glama README](https://glama.ai/mcp/servers/@itsuzef/ableton-mcp/blob/7a0216f58180507167b9c1a4e97ee2aec407b39b/README.md)
- `LivePilot`: connects any MCP client (Claude, VS Code Copilot) to Ableton with "full control over track creation, MIDI programming, instrument loading, parameter tweaking, song arrangement, and mixing" — [MCP Market](https://mcpmarket.com/server/livepilot)
- Ableton Live 12 pricing (EUR, April 2026 sale): Suite €444 (normally €599), Standard €209 (normally €279), Intro €59 (normally €79); Suite rent-to-own 24 x €24.96 — [Bonedo](https://www.bonedo.de/?p=1377828). No official USD list found.

**REAPER**
- `TwelveTake-Studios/reaper-mcp` v1.9.3: MIT; 94 stars, 18 forks, 56 commits; file-based transport — MCP server and a Lua bridge script inside REAPER exchange JSON request/response files in a mailbox directory, no ports (HTTP transport removed in v1.7.2). ~179 tools: Track ops 23, FX 16, ReaEQ 5, Routing 9, Mixing helpers 6, Take FX 11, Take mgmt/comping 7, Audio items 17, MIDI 8+14, Transport 10, Project 15, Markers/regions 8, Selection/editing 11, Automation 8, FX param automation 5, Measurement (loudness, true peak, spectrum, stereo) 2, Advanced (run actions, FX chunks) 4. Requires REAPER (tested through 7.79), Python 3.10+; install via `uvx`/`pipx`. Windows bridge path `%APPDATA%\REAPER\Scripts\` with `--install-bridge` one-command installer and optional autostart. Limits: ~50 ms per call, 5 s default timeout; render/measurement near-realtime with 600 s timeout; `save_fx_preset` always errors; `render_region` not implemented; true-peak/spectrum need bundled JSFX analyzer; only x86_64-linux Nix shell tested — [GitHub TwelveTake-Studios/reaper-mcp](https://github.com/TwelveTake-Studios/reaper-mcp)
- `itsuzef/reaper-mcp` (redirects to bonfire-systems/reaper-mcp): MIT; 163 stars, 34 forks, 44 commits; uses python-reapy distant API (`scripts/enable_reapy.py`); 60 tools: project 9, tracks 10, templates 2, MIDI 4, FX/instruments 10, audio 5, mixing 8, rendering 3, mastering 3 (`apply_mastering_chain`, `apply_limiter`, `normalize_project`), analysis 6 (`analyze_loudness`, `analyze_dynamics`, `analyze_frequency_spectrum`, `analyze_stereo_field`, `analyze_transients`, `detect_clipping`); requires REAPER 7+, Python 3.10+ — [GitHub itsuzef/reaper-mcp](https://github.com/itsuzef/reaper-mcp)
- `T-Rzeznik/reaper-mcp`: transport, track volume/mute/arm, FX management, automation modes, tempo — [Glama](https://glama.ai/mcp/servers/T-Rzeznik/reaper-mcp)
- `dschuler36` Reaper MCP: parses .RPP project files and answers questions (read-only, not live control) — [LangDB listing](https://langdb.ai/app/mcp-servers/reaper-mcp-server-e2e73cd7-4409-450f-807a-9444e52e589e)

**Bitwig**
- `WigAI` (github.com/fabb/WigAI): MCP server implemented as a Bitwig controller extension (`WigAI.bwextension`), runs at `http://localhost:61169/mcp`; requires Bitwig 5.2.7+, Java 21 LTS; capabilities listed: transport start/stop, device parameter control for selected device, clip and scene launching; license not stated on listing — [mcp.so WigAI](https://mcp.so/servers/wigai)
- `Bitwig Control Deck`: controller extension listens on UDP 50701 with an authenticated local protocol; MCP server gives "read-back-verified control of tracks, devices, transport, native devices, and third-party plug-in parameters" plus status dashboard; binds loopback only; independent community tool — [Glama](https://glama.ai/mcp/servers/dsol2ohgsm) (GitHub URL not surfaced)
- `jxstanford/bitwig-mcp-server`: Python, for Claude Desktop; author states "work in progress... not all functionality may work as expected"; requires Bitwig 5.2+ and DrivenByMoss 5.2+ — [README](https://cdn.jsdelivr.net/gh/jxstanford/bitwig-mcp-server@main/README.md)
- `Beat Twin`: Node.js, controller script speaks JSON-RPC over local TCP; write tools hidden/blocked by default — [Glama](https://glama.ai/mcp/servers/cmuq33p0yg)
- Bitwig Studio 6 itself still lacks retrospective MIDI capture per a review ("retrospective MIDI capture and audio recording is still unavailable") — [Rekkerd review of Bitwig 6](https://rekkerd.org/review-bitwig-studio-6/)

**SuperCollider / Pure Data**
- `Ziforge/supercollider-mcp`: pip/uv install, wired into Claude Code; architecture "Claude ↔ MCP server ↔ OSC :57120 → sclang → OSC :57121 → MCP → Claude"; load a setup.scd once per session — [Glama](https://glama.ai/mcp/servers/jb5239u407)
- `BradA1878/mcp-wave`: plain-English to synthesized sound; 10 built-in synth types, pattern sequencing, recording; Node 18+ — [Glama README](https://glama.ai/mcp/servers/@BradA1878/mcp-wave/blob/517c5633b06d974a3e9720dfa9814a0a19c17230/README.md)
- `Synohara/supercollider-mcp` (npm @makotyo/mcp-supercollider): uses supercolliderjs; author says tested only on macOS Apple Silicon — [mcpservers.org](https://mcpservers.org/servers/Synohara/supercollider-mcp)
- `damagethundercat/PureData-MCP`: starts Pd, opens GUI patches, generates .pd files, edits patches while running; `npx -y @damagethundercat/puredata-mcp`; Node 20+ — [Glama](https://glama.ai/mcp/servers/damagethundercat/PureData-MCP)
- `nikmaniatis/Pd-MCP-Server`: Python with OSC daemon talking to a Pd bridge patch; Python 3.7+, Pd 0.51+ — [Glama](https://glama.ai/mcp/servers/lgw0pq4p4a)
- Generic bridge `yyf/mcp2osc` lets an LLM generate/debug arbitrary OSC messages (would need custom mapping to a DAW's address space) — [GitHub](https://github.com/yyf/mcp2osc)

**Carla (plugin host)**
- Carla's README describes "full OSC control" via a separate Carla-Control front end; controllable items include Dry/Wet, Volume, Balance and all plugin parameters; JACK default but supports native drivers incl. DirectSound — [Carla README snapshot](https://git.kx.studio/falkTX/Carla/src/commit/1a6ff078ba99fcb351a718e8f9ae7c2d49704f5b)
- Distro packaging lists Carla 2.5.9 as GPL-2.0-or-later with project at github.com/falkTX/Carla — [Alpine package](https://pkgs.alpinelinux.org/package/v3.22/community/loongarch64/carla-dev); a KXStudio site commit says pre-compiled binaries exist for Linux, macOS and Windows via GitHub releases — [KXStudio website commit](https://git.kx.studio/KXStudio/Website/commit/c4a7878713d01793e3ee1db9843e2e9001dd3b40)

### Inferences
- For the MPC-One-as-instrument goal, the DAW's job is: arm audio tracks, record, cut regions at section boundaries, insert/adjust plugin chains, measure loudness, render. Both REAPER MCP servers and Ardour's MCPHttp cover every step of that list; the Ableton servers cover it only if Live is purchased.
- TwelveTake's file-mailbox transport avoids port/firewall issues on Windows and ships a measurement toolset (LUFS/true-peak/spectrum) that lets an LLM close the loop on mixing decisions, which is unusual among these servers.
- WigAI (in-process Java extension) is architecturally the most robust Bitwig option, but its listed toolset is small (transport, selected-device params, clip launch) compared with the REAPER/Ardour servers.
- The SuperCollider/Pd servers are relevant mainly as scriptable effect/synth engines or for generative accompaniment, not as recording hosts.

### Gaps
- GitHub does not expose last-commit dates to the fetcher; recency of every repo above is unverified. Star counts are as of fetch (Oct 2026).
- No GitHub URL found for Bitwig Control Deck or the Rust ardour-mcp.
- No MCP server found for Zrythm or LMMS.
- Carla's current Windows release status (2.5.x/2.6) and whether it can run headless from a Windows command line were not confirmed; results were 2013-era README snapshots.

## Key Question 3: Which open-source tools can do (scripted) mixing and mastering on Windows 11, and how would a script drive them?

### Takeaway
Matchering (GPL-3, ~2.7k stars) is the only mature open-source "automatic mastering" engine; it matches RMS, frequency response, peak and stereo width to a reference track and is a pip-installable Python library. Spotify's Pedalboard (GPL-3) hosts VST3 plugins from Python on Windows, so Airwindows (MIT), LSP (LGPL-3), Dragonfly, Zam and x42 plugins can be chained in scripts. No open-source library found does per-stem auto-EQ/balance; `auto-mix-prep` does gain staging only, and `allin1` does AI section detection.

### Cited Findings
**Mastering**
- Matchering 2.0: GPL-3.0; ~2.7k stars, 297 forks, 408 commits; takes a target and a reference and matches RMS level, frequency response, peak amplitude and stereo width; includes its own brickwall limiter "Hyrax"; `pip install -U matchering`, call `mg.process(target, reference, results=[mg.pcm16(...), mg.pcm24(...)])`; CLI via matchering-cli / kubinka0505/matchering-cli; Docker guides for Windows; also available as a ComfyUI node and inside UVR5 desktop app; needs Python 3.8+, 4 GB RAM; SoundFile/libsndfile auto-installs on Windows; FFmpeg optional for MP3; result quality depends on "carefully selecting a proper song as reference" — [GitHub sergree/matchering](https://github.com/sergree/matchering)

**Scriptable plugin hosting**
- Pedalboard (Spotify): GPL-3.0; supports VST3 plugins on macOS, Windows and Linux; Windows tested in CI with VSTs; bundled Steinberg VST3 SDK is GPLv3; PitchShift uses Rubber Band (GPLv2/commercial); MP3Compressor uses LAME; Windows wheels amd64 only — [PyPI pedalboard](https://pypi.org/project/pedalboard/0.9.2); [Spotify engineering blog](https://engineering.atspotify.com/introducing-pedalboard-spotifys-audio-effects-library-for-python/)
- VCMix / OpenDAW (PyPI `vcmix`): "AI-native open-source DAW — YAML-driven, cross-platform, AI Agent friendly" headless mixing host with auto gain staging and an `automix` command that "analyzes, suggests, and applies changes"; built around its own VC plugin chain — [PyPI vcmix](https://pypi.org/project/vcmix/)
- `auto-mix-prep`: gain staging (peak to target, default -12 dB) and stereo-to-mono splitting using pydub; no EQ — [PyPI auto-mix-prep](https://pypi.org/project/auto-mix-prep)
- `phantom` (fadelabs): MCP-based WAV stem analysis giving gain staging and EQ advice (assistant, not automatic mixer) — [README](https://cdn.jsdelivr.net/gh/fadelabs/phantom@main/README.md)
- `remucs`: Demucs wrapper for stem separation/remix with gain and L/R balance per stem (drums/bass/vocals/other) — [PyPI remucs](https://pypi.org/project/remucs)

**Open-source plugin collections (Windows availability)**
- Airwindows Consolidated: "a CLAP/AU/VST3/LV2/Standalone which uses JUCE to expose all the Airwindows as a single plugin"; Windows Setup.exe or .zip; ~400 effects — [airwin2rack docs](https://git.tdem.in/baconpaul/airwin2rack/blob/main/doc/manualdaw.md); [Rekkerd](https://rekkerd.org/airwindows-consolidated-400-effects-in-one-multi-fx-plugin/); Chris Johnson describes downloads as free and open source under MIT — [Tapatalk post](https://www.tapatalk.com/groups/prorecordingworkshop/airwindows-totape9-free-mac-windows-linux-pi-clap--t19417539.html); winget id `Airwindows.AirwindowsConsolidated.DAWPlugin`, listed version 2025-08-03 — [wingetgui](https://wingetgui.com/apps/Airwindows-AirwindowsConsolidated-DAWPlugin); caveat: combining with JUCE/VST3 SDK brings GPL3 dependencies into redistributed builds — [airwin2rack docs](https://git.tdem.in/baconpaul/airwin2rack/blob/main/doc/manualdaw.md)
- LSP Plugins: official site lists VST3 (full support) and CLAP (full support); download notes include a standard Windows CLAP install path — [lsp-plug.in](https://lsp-plug.in/); license LGPL-3.0-or-later — [SUSE package](https://packagehub.suse.com/packages/lsp-plugins/); source github.com/lsp-plugins/lsp-plugins — [GitHub](https://github.com/lsp-plugins/lsp-plugins)
- ZamAudio zam-plugins: GPL-2.0-or-later; LADSPA/LV2/VST/JACK (CLAP package exists on Alpine) — [Alpine zam-plugins-clap](https://pkgs.alpinelinux.org/package/v3.19/community/x86_64/zam-plugins-clap)
- x42-plugins: source github.com/x42/x42-plugins, "cross platform LV2 audio/midi plugins"; meters.lv2 (metering) and fil4.lv2 (4-band parametric EQ with shelves and HP/LP) — [LAU announce](https://lists.linuxaudio.org/archives/linux-audio-user/2015-July/101897.html); distro license GPL-2/GPL-3 — [Debian](https://packages.debian.org/unstable/x42-plugins); official FAQ says Windows installer exists but prebuilt Windows binaries are sold as a "partner product from the Mixbus Store" requiring a license file — [rg42.org FAQ](https://rg42.org/x42/faq)
- Vital: source "licensed under the GPLv3" per the Vita Python bindings project (Windows supported) — [PyPI vita](https://pypi.org/project/vita/); forum thread says Vital source "isn't up publicly on Matt's GitHub" while an older HN comment cites github.com/mtytel/vital — [Vital forum](https://forum.vital.audio/t/is-vital-actually-open-source/1396); [SaaSHub](https://www.saashub.com/compare-vital-vs-unleash); nixpkgs lists GPL-3.0-or-later but flags "unfree" — [MyNixOS](https://mynixos.com/nixpkgs/package/vital)

**AI section detection (for chopping a jam into song parts)**
- `allin1` (All-In-One Music Structure Analyzer): predicts tempo, beats, downbeats and functional segment boundaries/labels (intro, verse, chorus, bridge, outro); CLI `allin1 file.wav`; Python `allin1.analyze(...)` returns segments with start/end/label; requires PyTorch, NATTEN (required on Windows), madmom from GitHub — [PyPI allin1](https://pypi.org/project/allin1)
- MSAF: classic structural segmentation toolkit, `msaf.process(audio_file)` returns boundaries and labels; Nieto & Bello ISMIR 2016 — [msaf.readthedocs.io](https://msaf.readthedocs.io/); [GitHub urinieto/msaf](https://www.github.com/urinieto/msaf)
- No consumer tool found that splits a jam into sections; SplitJam / Jam Recorder apps are 6-stem separators, not section detectors — [SplitJam App Store](https://apps.apple.com/app/id6765638522); [Jam Recorder](https://appgoblin.info/apps/6751782457)

### Inferences
- A fully open-source "record jam -> chop -> mix -> master" pipeline on Windows is feasible with: REAPER/Ardour MCP for capture and region cuts; `allin1` to propose section boundaries; Pedalboard or the DAW's FX chain to apply Airwindows/LSP processing; the DAW MCP's loudness tools (TwelveTake) or `pyloudnorm`-style checks for targets; Matchering with a reference hip-hop master as the final mastering stage.
- The missing piece is per-stem automatic balance/EQ: nothing open-source found; an LLM reading spectrum/LUFS measurements and setting ReaEQ/LSP parameters via MCP is the practical substitute.
- x42 is effectively "free on Linux, paid binaries on Windows" unless self-compiled; LSP and Airwindows are the cleanest free Windows installs.

### Gaps
- Dragonfly Reverb license and Windows VST3/CLAP builds not confirmed (only "open source and free" from a forum). Repo is believed to be github.com/michaelwillis/dragonfly-reverb but this was not verified.
- No last-commit dates for matchering, pedalboard, LSP, Airwindows.
- Whether Pedalboard can host CLAP plugins (vs VST3 only) was not researched.
- `allin1` Windows install difficulty (NATTEN compile) not verified.

## Key Question 4: How can a live MPC One performance be captured automatically, and what does MPC 3 itself offer for song structure?

### Takeaway
MPC 3 (free for MPC One) replaced the old sequence-based Song mode with a DAW-style linear "Arrange Mode" that records, edits and arranges MIDI and audio on a timeline with automation across tracks, buses and master; it is in beta on desktop and v3.9 is the latest release notes found (PDF created June 2026). On the PC side, Ableton Capture, Studio One Retrospective Record (MIDI + up to 60 s audio pre-record) and REAPER's retroactive MIDI buffer exist; Bitwig 6 lacks it. No official confirmation found that a standalone MPC One streams multitrack audio to a PC over USB.

### Cited Findings
- MPC 3 public beta is a free update for MPC One and other standalone models; "colour-coded DAW-style Arrange Mode that collects all the tracks in a project together in one window, allowing you to record, edit and arrange both MIDI and audio on a linear timeline" — [MusicRadar](https://musicradar.com/news/akai-pro-mpc-3-software); compatible models listed by Akai: Live, Live II, One, One+, X, X SE, Key 37, Key 61 — [akaipro.com/mpc3](https://akaipro.com/mpc3.html)
- MPC 3.5 beta notes describe an "all-new unified track architecture" — [MPC Desktop Beta 3.5 Release Notes PDF](https://cdn.inmusicbrands.com/Software/ACVS-3.5.0%20Public%20Beta/MPC%20Desktop%20Beta%203.5%20Release%20Notes.pdf)
- MPC 3.9 release notes: desktop software remains in beta; adds AU and VST3 plug-in versions; Touch UI arrange tools operate on the Loop Brace with Cut/Copy/Paste/Duplicate keys; projects created in standalone 3.4–3.9 are not backward compatible with MPC 2.15 desktop; PDF metadata creation date 2026-06-15 — [MPC 3.9 Release Notes PDF](https://cdn.inmusicbrands.com/Software/15JM26PSBC/MPC%203.9%20Release%20Notes.pdf) (binary PDF; details from search snippets and metadata)
- Arranger Mode offers automation across tracks, buses and master — [KVR news](https://www.kvraudio.com/news-print.php?id=61332)
- MPC Standalone OS v3.4 user guide (reference for Arrange Mode operation) — [MPC Standalone OS User Guide v3.4 PDF](https://cdn.inmusicbrands.com/akai/MPC3-NI/MPC%20Standalone%20OS%20-%20User%20Guide%20-%20v3.4.pdf)
- Third-party course lists lessons "Recording in Arrangement Mode" and "Bounce a MIDI Track to an Audio Track in Arrangement Mode" — [noquantize course](https://noquantize.teachable.com/courses/2739474/lectures/60357030)
- User workaround for live arrangement: mapping an external controller to pad mutes to "record pad mutes live into the arranger" (MPC X user) — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1885818); a forum commenter sees unified sequence tracks as "the price to pay for having Disk Streaming and Arranger mode" — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1885954)
- MPC One has a built-in audio interface that appears as an audio device in MPC Software over USB (Preferences > Audio) — [Audeobox guide](https://www.audeobox.com/learn/mpc-software/mpc-controller-setup/); a 2022 forum reply said standalone mode could not send audio to a DAW over USB and the workaround was controller/plugin mode — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1839638); a June 2026 Windows user reported MPC One only connecting in controller mode when MPC 3 was already open — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1904893)
- Ableton Live: retailer listing describes recording an arrangement directly from Session View and improvising a finished composition (Capture workflow) — [PMT Live 12 Standard](https://pmtonline.co.uk/ableton-live-12-standard-digital-download); Live 12.4 released 5 May 2026 adds Link Audio (stream audio between devices over LAN), stem-separation on time selections, merge stems to one track, Learn View — [Ableton blog](https://www.ableton.com/en/blog/live-12-4-is-out-now/); [MusicRadar](https://www.musicradar.com/music-tech/ableton-live-12-4-is-out-now-with-link-audio-and-updated-erosion-delay-and-chorus-ensemble-devices); Live 12.3 added stem separation (Suite), Splice in browser, group bouncing — [Ableton help: What's new](https://help.ableton.com/hc/en-us/articles/12746369006108-What-s-new-in-Ableton-Live-12)
- Studio One Retrospective Record (since 5.1): remembers all notes, parameter adjustments, controller movements on an enabled Instrument track; predefined audio pre-record buffer up to 60 seconds — [Tape Op review](https://tapeop.com/reviews/gear/142/studio-one-professional-v51-update); Studio One Pro 7 listing: "capture every note with Retrospective Record/Pre-Record Audio Input, even without hitting Record" — [Adorama](https://www.adorama.com/prs1093169.html)
- Bitwig Studio 6: "retrospective MIDI capture and audio recording is still unavailable" — [Rekkerd](https://rekkerd.org/review-bitwig-studio-6/)
- REAPER: buffers everything played on an instrument track for later retrieval — [Yamaha Musicians forum](https://yamahamusicians.com/forum/viewtopic.php?p=108256)

### Inferences
- For "always-on" capture of an MPC One jam, the dependable approach is a DAW audio track in continuous record (or Studio One's 60 s audio pre-record) fed from the MPC One's analog outputs via an audio interface, plus MIDI capture over USB; a script can then cut regions. MPC 3's own Arrange Mode can do the same on-device but is not externally scriptable.
- The MPC One's USB audio appears usable only in controller/plugin mode (MPC 3 desktop), so an open-source DAW capture path most likely needs an audio interface for the MPC's analog outs. This should be verified against current MPC 3 docs.

### Gaps
- The MPC 3.9 release notes PDF could not be text-extracted; exact wording on Song mode removal, audio track count on MPC One, and release date remain unconfirmed.
- No official statement found on whether MPC 3 standalone exposes multitrack USB audio to a computer.
- Ableton Capture official documentation not fetched (retailer wording only).
- No tool found that automatically segments a long jam into sections other than the research libraries in Key Question 3.

## Key Question 5: Which commercial/AI arrangement and song-structure tools exist (brief)?

### Takeaway
Commercial "AI arrangement" is mostly chord/MIDI-idea generation (Scaler 3 $99, Captain Plugins Epic ~$99, Orb Producer Suite 3 ~€99, Playbeat 3 ~$79) plus generative services (BandLab SongStarter free, Suno/Udio extend, Google Lyria 2 in the restricted Music AI Sandbox). Only FL Studio 2026's agentic Gopher and the MCP controllers above arrange a user's own material from natural language; Logic Pro 11 is Mac/iPad only.

### Cited Findings
- Scaler 3: $99 list (introductory $79 in March 2025) — [Dubspot](https://blog.dubspot.com/plugins/scaler-3); [Sound on Sound](https://www.soundonsound.com/news/scaler-music-introduce-scaler-3)
- Captain Plugins Epic: $99 USD Windows/Mac VST/AU, $29 upgrade (2022 announcement) — [Rekkerd](https://rekkerd.org/mixed-in-key-releases-captain-plugins-epic-plugin-suite)
- Orb Producer Suite 3: 99 EUR VST3/AU (2021) — [Rekkerd](https://rekkerd.org/?p=202360)
- Audiomodern Playbeat 3: £79 list; Riffer/Chordjam $49 each — [Music Matter](https://musicmatter.co.uk/audiomodern-playbeat-3); [Rekkerd](https://rekkerd.org/?p=212498)
- Ableton Live 12 generative MIDI tools are "not strictly AI" and work like Scaler/Instachord; Sound Similarity Search analyses samples — [Midnight Rebels](https://midnightrebels.com/tldr-ableton-live-12-whats-new/); [DJ Mag](https://DJMAG.COM/tech/ableton-live-12-available-now)
- BandLab SongStarter: AI idea generator; genre or lyrics input; "roll the dice" gives 3 variations; patterns are free to keep — [BandLab help](https://help.bandlab.com/hc/en-us/articles/24226593087001-What-is-SongStarter); [RouteNote](https://create.routenote.com/blog/what-is-bandlab-songstarter/)
- Google Lyria 2 / Music AI Sandbox: 48 kHz stereo; Create/Extend/Edit tools; ~30 s clips; US-only waitlist as of April 2025; Lyria RealTime in Gemini API; SynthID watermark — [AlternativeTo](https://alternativeto.net/news/2025/4/google-expands-music-ai-sandbox-toolset-with-lyria-2-model-for-professional-music-creation/); [TechRadar](https://www.techradar.com/computing/artificial-intelligence/google-has-tuned-up-its-ai-music-sandbox-for-musicians-and-producers); [DeepMind Lyria](https://deepmind.google/lyria)
- Udio audio upload extends an uploaded clip forward/backward by 32 s using up to 2 min of context — [Analytics India Mag](https://analyticsindiamag.com/ai-news-updates/udio-launches-audio-uploads-feature-to-extend-and-remix-user-clips); as of 2026 Udio's help center says audio/video/stem downloads are disabled (post-UMG deal) while Suno still offers downloads, stems, section replacement and extensions; Covers for Pro/Premier — [Sonilo comparison](https://sonilo.com/blog/comparisons/suno-vs-udio-video-creators-2026)
- Suno Studio 2.0 (13 Aug 2026): browser DAW with MIDI import/record/edit, wavetable synth, chat bar (beta) that generates instruments/vocals and "creates new and totally unique plugins and synth presets", sidechain compression, convolution reverb, 32-bit/48 kHz stem export; Premier-only ($24/mo annual, $30 monthly); no third-party VSTs; Studio 1.x retired early Sept 2026 — [Suno release notes](https://suno.com/release-notes/studio-2); [Dubspot](https://blog.dubspot.com/suno-studio-2-0); [RouteNote](https://routenote.com/radar/?p=3200)
- FL Studio 2026: Gopher "transformed into an agentic assistant that can access and make changes to your project" — adjust track levels, plugin parameters, mixer channels; can create a four-on-the-floor kick pattern, backbeat snares, gated reverb; "not taking musical responsibility yet"; cannot create automation or write melodies/chords; Loop Starter and Stem Separation (Producer Edition) — [MusicRadar](https://www.musicradar.com/music-tech/you-can-now-control-fl-studio-with-an-ai-chatbot-from-inside-the-daw); [RouteNote](https://routenote.com/blog/fl-studio-2026-gopher-ai-production-assistant/); [NewsBytes](https://www.newsbytesapp.com/news/science/fl-studio-2026-update-lets-gopher-handle-drums-and-effects/tldr)
- Logic Pro 11 (Mac; features also on Logic Pro for iPad 2 — NOT Windows): Session Players (Bass Player with 8 AI players, Keyboard Player), ChromaGlow, Stem Splitter; £199.99 new, free update — [BusinessWire](https://www.businesswire.com/news/home/20240507524533/en/Logic-Pro-takes-music-making-to-the-next-level-with-new-AI-features); [MusicTech](https://musictech.com/news/industry/apple-logic-pro-for-mac-11-logic-pro-for-ipad-2/)

### Inferences
- None of the commercial arrangement plugins operate on recorded MPC audio; they generate MIDI. For an MPC-as-sound-source workflow they are peripheral.
- Suno Studio 2.0 is the only commercial product combining a DAW timeline with an LLM chat bar, but its lack of VST support and cloud-only model make it unsuitable as the capture host.

### Gaps
- Current (2026) prices for Captain Epic, Orb 3, Playbeat 3 not found; figures are 2021–2022.
- No results for Soundtrap, Hookpad, Ripchord, WavTool, Staccato, Amper/Shutterstock, Splice Create details, Native Instruments AI assistant, Mureka or AIVA.
- Logic Pro Mastering Assistant specifics not found in results (feature exists since Logic 10.8; Mac only — not sourced here).

## Key Question 6: Which commercial AI mixing tools exist (brief)?

### Takeaway
iZotope Neutron 5 ($299; Elements $55) Mix Assistant builds a per-track chain from a short analysis; sonible smart:EQ 4 ($129) and smart:limit ($129) are the other mainstream choices; all are Windows VST3. RoEx Automix is the main cloud stem auto-mixer (~$9.99–$24.99/mo depending on source).

### Cited Findings
- Neutron 5 $299, Neutron 5 Elements $55 (reseller sale $224.25 / $25) — [iZotope dynamics collection](https://www.izotope.com/collections/dynamics); Mix Assistant: tell it the source type, analyze audio, it "builds a signal chain, sets some initial parameters, and gives you simple macro controls"; reviewers call it a starting point — [Making a Scene review](https://www.makingascene.org/?p=129917); new modules Clipper, Density, Phase, Mid/Side and Transient/Sustain modes, Delta buttons — [NI press](https://www.native-instruments.com/en/press-area/music-production/neutron-5/); Sound on Sound review — [SOS](https://soundonsound.com/reviews/izotope-neutron-5)
- sonible: smart:EQ 4 $129 list ($79 on sale Aug–Sep 2026); smart:limit $129 list ($59 on sale); pure:bundle $159 ($99 sale) — [Plugin Boutique sonible](https://www.pluginboutique.com/manufacturers/190-Sonible); [Plugin Boutique deal](https://www.pluginboutique.com/deals/23906)
- RoEx Automix: own page lists Pro at £19.99/$24.99 per month (one version) and $9.99/month (another version); MixingGPT: 1 free download, $7.99 per download, Pro $24.99/mo or $149.99/yr; up to 32 tracks; free previews — [RoEx blog](https://www.roexaudio.com/blog/best-ai-mixing-and-mastering-services-compared/); [MixingGPT](https://mixinggpt.com/blog/best-ai-mixing-services-2026); RoEx Studio Pro tier announced — [Headliner](https://headlinerhub.com/roex-announces-premium-studio-pro-subscription-tier.html)
- Waves Clarity Vx: $199 list, $49.99 sale — [Sweetwater](https://www.sweetwater.com/insync/?p=217746)
- Focusrite FAST bundle: usual $359.85, discounted to $39.99 in a 2023 sale — [MusicRadar](https://www.musicradar.com/news/focusrite-fast-plugins-summer-sale)
- Baby Audio Smooth Operator $69 regular; Smooth Operator Pro €129 (€78 Thomann deal Jan 2026) — [Rekkerd](https://rekkerd.org/?p=196316); [Bonedo](https://www.bonedo.de/?p=2705823)

### Inferences
- For an LLM-driven setup, these plugins are useful as single-knob insert effects the agent can load via REAPER/Ardour MCP; their own AI runs on an analyze-button, which MCP servers generally cannot press.

### Gaps
- No pricing found for Gullfoss, Mastering The Mix, Techivation, Sonarworks, Tonn, Nectar 4 standalone.
- Neutron 5 "Unmask" mechanics not described in sources.

## Key Question 7: Which commercial AI mastering tools exist (brief)?

### Takeaway
Ozone 12 (Elements $55 / Standard $219 / Advanced $499) adds a customisable Master Assistant with LUFS targets, Stem EQ, Bass Control and Unlimiter; cloud options range from free (BandLab) to ~$4–6/track (LANDR, Masterchannel, Waves) to $15–30/mo subscriptions.

### Cited Findings
- Ozone 12 Advanced $499 (20 tools), Standard $219 (14 tools), Elements $55 (AI Assistant) — [iZotope filter collection](https://www.izotope.com/collections/filter); Mix & Master Bundle Advanced $599 — [iZotope shop](https://www.izotope.com/en/shop/mix-and-master-bundle-advanced-ozone-12)
- Master Assistant auditions a section and suggests a chain; v12 lets you choose modules, set LUFS targets, more genres; new Stem EQ, Bass Control (ML), Unlimiter; IRC 5 transparent limiter; "won't rescue a poor mix" — [Sound on Sound review](https://www.soundonsound.com/reviews/izotope-ozone-12-advanced?amp); MusicTech 8/10: Stem EQ "the real star"; Standard lacks individual module plugins — [MusicTech](https://musictech.com/reviews/plug-ins/izotope-ozone-12-review/)
- LANDR: ~$4 per track; Studio tiers quoted at $23.99/$49.99/$99.99 annual (conflicting $12.99/mo elsewhere) — [fast.io](https://fast.io/resources/best-ai-mastering-tools-2026/); [Dynamoi](https://dynamoi.com/ko/learn/ai-music-distribution/ai-music-mastering-tools)
- eMastered: ~$20/mo or $15/mo annual vs $49.99 monthly (sources conflict) — [MixingGPT eMastered review](https://mixinggpt.com/blog/emastered-review-2026)
- BandLab Mastering: free, unlimited, no watermark — [fast.io](https://fast.io/resources/best-ai-mastering-tools-2026/)
- Masterchannel: $5/track via partners; Artist tier $15–25/mo, Professional $19–29/mo (includes Wez Clarke "digital clone"); reinforcement-learning based — [Masterchannel blog](https://blog.masterchannel.ai/what-does-it-cost-to-master-a-song/); [RecordJet](https://blog.recordjet.com/en/?p=283390)
- Waves Online Mastering: earlier pricing $6 single credit to $180/60 credits (~$3/track); Aug 2025 update "reduced pricing" and added Electronic style — [MusicRadar](https://musicradar.com/news/waves-online-mastering); [Synthtopia](https://www.synthtopia.com/content/2025/08/01/waves-audio-upgrades-online-mastering-service-adds-electronic-music-mastering-style/)

### Inferences
- Matchering (free, scriptable, reference-based) covers the same use case as the cloud masterers for a producer who has a reference track; the cloud services add genre models and loudness presets but cannot be driven locally from a script without an API.

### Gaps
- No CloudBounce, Aria Mastering or Logic Mastering Assistant data found.
- Vendor pricing pages for LANDR/eMastered not fetched; secondary sources conflict.
