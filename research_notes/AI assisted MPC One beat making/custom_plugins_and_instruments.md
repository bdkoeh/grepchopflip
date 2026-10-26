# AI-Assisted Custom Plugins & Instruments (VST3/AU/CLAP) and How They Relate to the Akai MPC One

State as of October 2026. Host assumed: Windows 11 PC. Hardware: Akai MPC One (Gen 1, standalone, no third-party plugin hosting). Emphasis per coordinator update: open-source tools, the practical Claude Code -> compile -> VST3 path on Windows 11, open-source instruments as MPC sound sources (via MPC desktop controller mode), and open-source tooling to render multisamples / keygroup programs automatically. Commercial tools kept brief.

Note on GitHub metadata: WebFetch of GitHub repo pages returned star/commit counts but the rendered pages did not expose last-commit dates or release dates; where "last commit" is requested below, it is listed as a gap rather than guessed.

---

## KQ1. Practical, working paths in 2026 for an LLM (Claude Code / Cursor / Copilot) to build and compile a VST3 on Windows 11 — and which frameworks are most LLM-friendly

### Takeaway
The proven 2026 route is: Claude Code (or Cursor) + JUCE (free Starter tier / AGPLv3) + CMake + Visual Studio 2022 C++ workload, optionally scaffolded by the MIT-licensed "Audio Plugin Coder" (APC) framework which packages the whole agent workflow for JUCE 9 and emits VST3/Standalone (CLAP beta) on Windows. Documented experiments (The Audio Programmer, Feb 2026; Sound On Sound, Jan 2026) confirm simple plugins (gain, sequencer) work with zero hand-written code, while complex custom UIs and subtle DSP still break down; JSFX (Reaper) is the lowest-friction "no compiler" format, and Faust / Cmajor / Plugdata / RNBO offer higher-level DSL paths with their own export toolchains.

### Cited Findings

**Documented LLM "vibe-coding" plugin experiments (2025–2026)**
- The Audio Programmer (Joshua Hodge) published "Building an Audio Plugin with Claude Code: A Vibe Coding Experiment" on 26 Feb 2026: a gain plugin built with "zero hand-written code" using Claude Code (CLI, Claude Max plan), JUCE fetched via CMake, tested in Ableton Live; the first build/test succeeded; dB scaling and a rotary slider were added, but the custom UI work eventually "broke down"; installation issues consumed noticeable setup time; author's framing: the role shifts "from developer to curator." — [The Audio Programmer](https://www.theaudioprogrammer.com/content/building-an-audio-plugin-with-claude-code-a-vibe-coding-experiment)
- Sound On Sound, "Vibe Coding: How Anyone Can Create Plug-ins" (Benjamin McCulloch, Jan 2026): used Claude (judged best vs ChatGPT and Gemini in the author's trials) to generate a 303-style 1–32-step sequencer as a Reaper JSFX plug-in because JSFX is "the most straightforward format and Reaper has a built-in compiler"; per-step velocity never worked despite repeated attempts; the article says VST/AU/AAX are "more complex because they need an extra development tool alongside Claude and a DAW"; it names JUCE (weeks–months to learn; only route to AAX) and Cabbage/Csound (hours–days; exports VST/AU) as next steps; practical advice: start with the simplest plugin, paste exact error messages, fix function before appearance, keep rollback versions. — [Sound On Sound](https://www.soundonsound.com/techniques/vibe-coding-how-anyone-can-create-plug-ins)
- A February 2026 YouTube video ("Vibe Coding a VST Plug In with Claude Code") tests whether Claude Code can produce a working audio plugin. — [YouTube](https://www.youtube.com/watch?v=6PHZc5oqeJc)
- A 2026 producer-oriented guide states Claude Code or Cursor are "the standards in 2026 for deeper projects on your own machine." — [Future Proof Music School](https://futureproofmusicschool.com/blog/vibe-coding-for-music-producers)

**Audio Plugin Coder (APC) — the main open-source "agent + JUCE" template (Windows-capable)**
- APC (Noizefield) is an open-source, MIT-licensed framework for building VST3/AU plugins with an LLM coding agent; it guides the agent through ideation, architecture, UI, DSP, testing, and packaging in phases (setup, dream, plan, design, impl, ship); built on JUCE 9 (submodule); supports Codex, Claude Code, Kilo, Cursor, OpenCode, Antigravity/Gemini (and Devin CLI per v1.6.0 notes). GitHub metrics at fetch: 343 stars, 58 forks, 124 commits; latest release listed v1.8.0 (added CLAP beta). — [GitHub: Noizefield/audio-plugin-coder](https://github.com/Noizefield/audio-plugin-coder)
- APC Windows requirements: an LLM agent, Git, Node.js 18+, Python 3.8+, Visual Studio 2022 with C++, CMake 3.22+, WebView2 Runtime; JUCE 9 and pluginval as submodules. Format matrix: VST3 and Standalone on Windows/macOS/Linux; AU macOS-only; LV2 Linux-only; CLAP opt-in beta on all three. Start: `npx github:Noizefield/audio-plugin-coder` (or clone with submodules), then `/apc-setup` and `/apc-dream <PluginName>` in the agent. Plugins inherit JUCE's dual license (AGPLv3 free, or commercial for closed-source). Known limits: Visage pure-C++ UI experimental; CLAP "not yet battle-tested"; clap-validator state-reproducibility failures for JUCE/APVTS plugins; author notes significant API costs. — [GitHub: Noizefield/audio-plugin-coder](https://github.com/Noizefield/audio-plugin-coder)
- Noizefield's announcement (published 5 Feb 2026, updated 31 Aug 2026) describes APC as agent-agnostic, requiring no C++ background, with a five-phase Dream/Plan/Design/Implement/Ship workflow, generating installers for Windows, macOS, Linux; users can pause/resume or switch AI models between phases. — [Noizefield](https://www.noizefield.com/news/create-vst-plugins-with-ai-apc)
- KVR lists APC as "an AI-first, open-source framework enabling musicians and developers to create VST3/AU/CLAP audio plugins using natural language," built on JUCE 8 (older listing), described by its developer as "in its early stages." — [KVR Audio](https://www.kvraudio.com/product/audio-plugin-coder-by-noizefield-instruments)

**Other JUCE templates with agent guidance**
- Pamplejuce (sudara/pamplejuce) is a C++23 JUCE 8.x template using CMake, CPM dependency management, integrated Catch2 testing/benchmarking; its README includes a note for AI coding agents pointing to CLAUDE.md / AGENTS.md files. — [Pamplejuce docs mirror](https://docsearch.algolia.com/mcp/docs/repo/sudara/pamplejuce); [GitHub fork mirror](https://github.com/nofishonfriday/pamplejuce)
- JUCE CMake Plug-in Template (JUCE Marketplace): MIT-licensed drop-in replacement for Projucer for JUCE 7/8, with GitHub Actions building and validating on macOS and Windows. — [JUCE Marketplace](https://marketplace.juce.com/products/juce_cmake_plug-in_template)
- fshstk/juce-project-template: barebones ISC-licensed CMake template that downloads dependencies at configure time, with a Catch2 unit-test template. — [GitHub](https://github.com/fshstk/juce-project-template)
- A community MCP server lets an LLM control JUCE synthesizer parameters via natural-language tool calls translated to OSC (parameter control, not code generation). — [Glama MCP listing](https://glama.ai/mcp/servers/integrations/juce)

**JUCE licensing (relevant because every JUCE-built plugin inherits it)**
- JUCE tiers (current page refers to "JUCE 9 licence"): Starter — free, revenue/funding up to $20,000, perpetual, all features; Indie — $40/user/month (or $800 perpetual), up to $300,000; Pro — $175/user/month (or $3,500 perpetual), unlimited; Educational — free. JUCE is dual-licensed under the JUCE licence and AGPLv3; AGPLv3 may be used "if you are not conveying closed-source software containing JUCE outside your organisation." — [JUCE](https://juce.com/get-juce/)
- JUCE 8 forum post: Starter tier revenue/funding limit up to $20,000, Indie $300,000, Pro no limit; the splash-screen requirement was removed in JUCE 8. — [JUCE Forum](https://forum.juce.com/t/amendments-to-the-juce-end-user-licence-agreement-for-juce-8/61265)
- Projucer limitations on Windows: the Code::Blocks exporter cannot build VST3; forum advice is to use CMake or install Visual Studio Community and use that exporter; "The Projucer doesn't export anything Visual Studio Code can use... switch over to using cmake." — [JUCE Forum (VS Code)](https://forum.juce.com/t/compiling-juce-ides-for-visual-studio-code/60220); [JUCE Forum (VST3 Code::Blocks)](https://forum.juce.com/t/vst3-plugin-with-code-blocks-windows/27034)

**iPlug2 (permissive alternative to JUCE)**
- iPlug2 is "licensed with a liberal zlib-like license, which means that it is free to use in closed source projects"; targets CLAP, VST2, VST3, AUv2, AUv3, AAX (Native), and Web Audio Module (WAM); minimum platforms Windows 8, macOS 10.11, iOS 14. GitHub license metadata shows "other." — [GitHub: iPlug2/iPlug2](https://de.github.com/iPlug2/iPlug2)
- A third-party build guide lists CMake 3.14+, Ninja or Visual Studio generators on Windows, per-format targets (e.g. `-vst3`, `-clap`), and says the CLAP and VST3 SDKs must be downloaded first (not official docs). — [Smithery skill page](https://smithery.ai/skills/iPlug2/build-cmake)
- A forum report says building CLAP and VST3 "mostly works the same way after adding the respective SDKs just like you would on win or mac," with Linux the weak spot. — [iPlug2 Discourse](https://iplug2.discourse.group/t/linux-support-for-my-deimos-synth/1035)

**Cmajor (DSL + JIT + exporter; GPLv3/commercial)**
- Cmajor is "available under a dual GPLv3/Commercial licensing scheme"; "If you fully comply with the terms of the GPLv3 (or later) then you can use Cmajor without needing to contact us"; commercial Indie license (revenue/funding under $300k, up to 5 seats) listed around GBP 2000; Pro per-seat on request; educational licences for institutions only. — [cmajor.dev Licence](https://cmajor.dev/docs/Licence)
- Windows install: download binaries (cmaj.exe CLI, redistributable libraries, and the Cmajor VST/AU plugin) from GitHub releases; `cmaj help`; `cmaj play <patch>.cmajorpatch` opens a window with controls and hot-reloads on file change; `cmaj create --name="Hello" <Folder>` scaffolds a patch. VS Code extension ("Cmajor") provides Run patch, Create patch, "Export as JUCE plugin" (= `cmaj generate --target=juce`, yields a JUCE C++ project for native VST/AU/AAX), "Export as CLAP plugin" (= `cmaj generate --target=clap`), and WebAssembly/HTML export. Patches can also be loaded by dragging a .cmajorpatch onto the Cmajor VST/AU plugin GUI in a DAW; the docs warn VST/AU limit dynamic parameter/channel changes. — [cmajor.dev Getting Started](https://cmajor.dev/docs/GettingStarted)
- GitHub: cmajor-lang/cmajor, 756 stars, 55 forks, 1,079 commits; Windows build requires Visual Studio 2019 and only Release/RelWithDebInfo builds are supported; plugin build not supported on Windows/arm64. — [GitHub: cmajor-lang/cmajor](https://github.com/cmajor-lang/cmajor)
- Cmajor was created by Julian Storer, the original author of JUCE. A JUCE forum poster criticises Cmajor as "biased towards the time domain" (FFT-based analysis clutters the syntax) but praises its sinc oversampling (opinion, not benchmark). — [Headchannel podcast](https://headchannel.co.uk/blog/writing-the-future-of-digital-music-creation-with-cmajor-software-co-founder-julian-storer-19/); [JUCE Forum](https://forum.juce.com/t/whats-your-opinion-on-cmajor/54152)

**Faust (functional DSP language; GRAME)**
- GRAME recommends the Faust Web IDE because "most of the features of Faust are accessible through the Faust Web IDE... which doesn't require any installation"; the online IDE "provides all the compilation services, including binaries generation for all the supported platforms and architectures"; a Windows local installer is available (`Faust-XXX-win64.exe`) on GitHub releases. — [Faust Downloads](https://faust.grame.fr/downloads/); [Faust GitHub mirror description](https://github.com/hiwatts/faust)
- Faust plugin exporters documented in the manual: `faust2juce` (JUCE plugin project unless `-standalone`), `faust2faustvst` (VST2, needs VST SDK; `-nvoices N` for polyphony), `faust2lv2`, `faust2clap` (CLAP; `--clap-sdk DIR`), `faust2ladspa`, `faust2dssi`, `faust2dplug`, `faust2vcvrack`, `faust2unity` (explicit Windows `-w64` option), `faust2max6`. No `faust2vst3` or `faust2dpf` listed. — [Faust Manual: Tools](https://faustdoc.grame.fr/manual/tools/)
- DawDreamer (below) embeds Faust and can transpile Faust to C++, Rust, Wasm, JAX/Flax — another route to run LLM-written Faust DSP from Python. — [GitHub: DBraun/DawDreamer](https://github.com/DBraun/DawDreamer)

**Plugdata / Pure Data (visual; GPLv3) and Camomile**
- plugdata ("Pure Data as a plugin, with a new GUI"): GPL-3.0 (JUCE AGPL-3.0 for the framework); 2.4k stars, 10,809 commits on develop; ships as VST3, LV2, standalone, with CLAP builds (CMake 3.21+); Windows installer available. — [GitHub: plugdata-team/plugdata](https://github.com/plugdata-team/plugdata)
- plugdata's "compiled mode" uses Wasted Audio's Heavy (hvcc) compiler; hvcc "can only compile plugins using a portion of the objects included with plugdata" and incompatible objects are reported in the Console; export is via the DISTRHO Plugin Framework (DPF) to LV2/VST2/VST3/CLAP/JACK (WIP wiki). A 2023-era post said the toolchain download is ~1.5 GB and compiled plugins "do not include any graphical interface (yet)." A separate `plugdata-plugin-export` GitHub-Actions project lists VST3/AU/CLAP/LV2/Standalone but says "don't use this in production yet!" — [plugdata wiki (WIP)](https://github.com/thouldcroft/plugdata_wiki_wip_tdh/blob/update-UI-overview/Compiling-And-Compilation-Mode.md); [MOD forum](https://forum.mod.audio/t/plugdata-new-pd-editor-that-can-create-lv2-plugins/8985); [GitHub: plugdata-plugin-export](https://github.com/plugdata-team/plugdata-plugin-export)
- Camomile (Pierre Guillot) appears only in plugdata's credits; no current export role found. — [GitHub: plugdata-team/plugdata](https://github.com/plugdata-team/plugdata)

**DPF / DISTRHO (permissive C++ framework)**
- DPF "can build for LADSPA, DSSI, LV2, VST2, VST3 and CLAP formats" from one codebase, plus JACK/Standalone for testing. — [GitHub: DISTRHO/DPF](https://github.com/DISTRHO/DPF)

**HISE (sampler-instrument builder; GPLv3)**
- HISE (christophhart/HISE) is an open-source framework for sample-based virtual instruments on JUCE; exports VSTi, AU, standalone; Windows build requires adding the ASIO SDK and VST3 SDK into `tools/SDK`, opening the bundled Projucer, and compiling in Visual Studio 2022; "Every compiled plugin or VSTi built with HISE has to be published and open-sourced under the GPLv3 License" unless a commercial HISE license (plus JUCE license) is bought. — [HISE docs](https://docs.hise.audio/introduction/index.html); [HISE GitHub (mirror)](https://github.com/nicroto/hise)

**RNBO (commercial, Max 9 add-on) — brief**
- RNBO's Audio Plugin export target builds VST3 or AU for macOS, Windows, or Linux regardless of host OS, via a cloud compiler (internet required); Windows VST3 location `C:\Program Files\Common Files\VST3`; docs state audio plugin export "is designed for your own personal use." — [RNBO docs](https://rnbo.cycling74.com/learn/audio-plugin-target-export-overview); [RNBO quickstart](https://rnbo.cycling74.com/learn/quickstart)

### Inferences
- For a non-expert on Windows 11 wanting a real VST3 (not JSFX), the lowest-risk path is APC + Claude Code: it pre-solves the CMake/VS2022/JUCE boilerplate that The Audio Programmer reported as the biggest time sink, and its pluginval submodule gives the agent an automated validation loop.
- LLM-friendliness ranking (inferred from documentation volume and toolchain simplicity): JUCE (massive training corpus, most templates, but C++/CMake/VS2022 required) > Faust (tiny single-file DSL, web IDE compiles without local toolchain, but polyphonic MIDI instruments require `-nvoices` and UI is auto-generated) ≈ Cmajor (hot-reload `cmaj play`, exports JUCE/CLAP, but far smaller corpus and GPLv3) > iPlug2 (permissive license but smaller community) > Plugdata (visual patching is awkward for text-generating LLMs; compiled export still has no GUI and limited object support).
- JSFX in Reaper is the fastest "proof of DSP idea" loop (no compiler), after which the same DSP can be ported to JUCE by the agent.
- Licensing: a hobbyist distributing nothing can use JUCE Starter (free, <$20k) or AGPLv3; Cmajor and HISE force GPLv3 unless paid; iPlug2 and DPF are permissive.

### Gaps
- No controlled benchmark comparing LLM success rates across JUCE vs Cmajor vs Faust vs iPlug2 was found; "LLM-friendly" is inferred, not measured.
- Last-commit dates for APC, Cmajor, iPlug2, DPF, HISE, plugdata were not exposed in fetched GitHub pages.
- The youngju.dev "Audio Plugin Development 2026" deep-dive (dated 2026-05-16) returned empty content when fetched; its claims could not be used.
- Whether the Faust Web IDE can export a Windows VST3 (vs VST2 via faust2faustvst / CLAP via faust2clap) was not confirmed; the manual lists no faust2vst3.
- No GitHub template specifically branded for Cursor/Copilot + JUCE was found beyond APC and Pamplejuce's CLAUDE.md.

---

## KQ2. Can the MPC One load custom instruments at all? What runs standalone vs only in controller mode?

### Takeaway
Akai confirms the MPC One cannot load any third-party plugin in standalone mode — only AIR/Akai/Native Instruments/Spitfire titles activated under Menu > Preferences > Activations. Custom sound content reaches the standalone MPC One only as samples, Drum programs, or Keygroup programs (or, from MPC 3.9, built-in MPC Oscillators on Keygroup/Drum tracks). Third-party VST3/AU plugins run only when the MPC One is in Controller Mode driving the MPC 3 Desktop software on a PC.

### Cited Findings
- Akai support: it is "not possible to load/install additional 3rd party plugins into the MPC in Standalone mode"; standalone includes only the AIR, Akai Pro, and Native Instruments plugins under Menu > Preferences > Activations. In Controller Mode (USB-B, enable from the Menu microchip icon) MPC 3 Desktop "supports VST3 plugins"; scan under Preferences > Plugins ("Scan New"/"Rescan All"); Windows folders listed are `C:\Program Files\VstPlugins` and `C:\Program Files\Steinberg\vstplugins`; actions in the desktop software are mirrored on the MPC screen; to return to standalone, bounce plugin tracks to audio. The article does not explicitly state VST2 support. — [Akai Support: How To Use 3rd Party VSTs With Your MPC](https://support.akaipro.com/en/support/solutions/articles/69000876025-akai-mpc-series-how-to-use-3rd-party-vst-s-with-your-mpc)
- Direct USB-MIDI (standalone MPC acting as a USB MIDI controller for DAW plugins without controller mode) exists only on MPC Key 37 (Gen 1) and the Gen 2 models MPC XL, Live III, Key 37 G2, One G2 — i.e. not on the original MPC One; all MPCs can instead use 5-pin MIDI Out to a USB/MIDI interface. — [Akai Support](https://support.akaipro.com/en/support/solutions/articles/69000876025-akai-mpc-series-how-to-use-3rd-party-vst-s-with-your-mpc)
- Akai: standalone hardware has "a predetermined and fixed amount of RAM," which limits the plugin selection. Free standalone instruments: Bassline, Electric, Drumsynth Multi, Hype, Mellotron, Odyssey, Solina, TubeSynth. Paid standalone (10-day trial): Fabric Collection, Fabric Vintage series, Jura, IONA, MiniD, OPx-4, Organ, Ring The Alarm, Stage EP, Stage Piano, Studio Strings, Sub Factory; NI Play Series MPC Edition (standalone & desktop 3.5+); Spitfire Cinematic Pads / Intimate Strings (3.7.1+). Desktop-only: AIR Essential Keyboards, Boom, DB-33, Loom II, Hybrid 3, Mini Grand, Strike, theRiser, TimewARP 2600, Vacuum, Velvet, BFD Player. MPC One standalone activation requires internet (Ethernet to router). — [Akai Support: Why are some plugins not available on my MPC](https://support.akaipro.com/en/support/solutions/articles/69000872472-akai-pro-mpc-series-why-are-some-plugins-not-available-on-my-mpc-)
- Akai support on missing plugin instruments: third-party VST plugins "are not compatible with the MPC operating system," so in Standalone a plugin track will have MIDI but no sound; the suggested fix is to convert the VST sound into a Keygroup program using Auto Sampler. — [Akai Support](https://support.akaipro.com/support/solutions/articles/69000855034-akai-pro-mpc-series-plugin-instruments-missing-in-project-)
- MPC 3.6 firmware (Oct 2025) lists the MPC One among supported hardware (with Live III, Key 37, One+, X SE, Key 61, Live II, X, Live, Force); in controller mode MPC 3.4 remained compatible with the MPC 2.15 desktop software. — [Akai MPC 3.6 release notes PDF](https://cdn.inmusicbrands.com/Software/36/MPC%203.6%20Firmware%20Update.pdf); [MPC 3.4.1 release notes PDF](https://cdn.inmusicbrands.com/akai/MPC3-NI/MPC%203.4.1%20Firmware%20Update%20Release%20Notes.pdf); [Synth Anatomy](https://synthanatomy.com/2025/10/akai-mpc-force-update.html)
- MPC 3.9 (19 June 2026) added built-in "MPC Oscillators" (wavetable, FM, ring mod, single-cycle, virtual analog) so Keygroup and Drum tracks can use oscillators instead of samples, and lets MPC Software run as an AU/VST3 plugin inside a DAW. MPC 3.10 (dated 23 Oct, year inferred 2026) adds MPK Mini Plus II/IV integration, enhanced Arranger Mode, 8x8 Notes/Clip Launch for MPCe pads (Live III/XL), moves MPC 3 desktop out of beta, and is free. Neither update adds third-party plugin hosting to standalone. — [Gearnews](https://www.gearnews.com/akai-mpc-3-synth/)
- Forum opinion (not Akai): the standalone platform "runs on Linux and barely anyone is making plugins for that." — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1853484)

### Inferences
- "Custom instrument" on a standalone MPC One therefore means one of: (a) a Keygroup program built from multisamples; (b) a Drum program of one-shots; (c) from 3.9, an MPC Oscillator-based Keygroup patch programmed on-device. Any AI-built VST3 synth can only be heard on the MPC One in Controller Mode or after being multisampled.
- Because the original MPC One lacks Direct USB-MIDI, using it as a plain MIDI controller for a PC-hosted plugin while in standalone requires 5-pin MIDI out + an interface; otherwise Controller Mode is the path.

### Gaps
- Akai's article names only VST3 (and AU via the Components path) for MPC 3 Desktop; whether MPC 3 Desktop still scans VST2 was not confirmed from Akai docs.
- Whether an MPC One (Gen 1) is still supported by MPC 3.9/3.10 firmware was not confirmed from the 3.9/3.10 release notes themselves (3.6 confirms it).

---

## KQ3. Realistic route to get a custom AI-built synth sound onto the MPC One (Auto Sampler, keygroups, open-source renderers)

### Takeaway
The official route is: host the plugin in MPC Desktop (Controller Mode), run Tools > Auto Sampler (note range/stride, velocity layers, note+tail length, loop, base name), then File > Save Current Programs to the MPC's drive — producing a .xpm Keygroup program plus WAVs that load standalone. Open-source alternatives automate the rendering step on Windows from Python: Spotify pedalboard (GPLv3, loads VST3 instruments and renders MIDI) or DawDreamer (GPLv3, VST + Faust + MIDI rendering), then assemble the .xpm with a tool like MPC Valet (Rust) or a script; keygroups hold max 4 sample layers each, 128 keygroups per program.

### Cited Findings

**Official Auto Sampler workflow (MPC 2 article; MPC 3 Desktop retains Auto Sampler)**
- Steps: open the plugin program (plug icon under Track) → Program: select the virtual instrument and preset → Tools > Auto Sampler → Sample Source auto-filled, set "Record from" to Resample L+R → Note Range: minimum/maximum octave and note stride → Velocity: which velocities to sample, optionally layered → Sampling: note length, tail length, Base name (becomes the keygroup program name) → Looping: enable if desired → On completion: auto-load as current program → "Do it." Then select the keygroup program, File > "Save Current Programs...", choose the external drive used with the standalone MPC. — [Akai Support: Create Keygroup programs using Auto Sampler](https://support.akaipro.com/en/support/solutions/articles/69000815776-akai-pro-mpc-2-software-how-to-create-keygroup-programs-using-auto-sampler)
- Community workaround when a plugin cannot be hosted in MPC software: host the VST in a DAW, route MIDI from the MPC64 plugin track to the VST track, route audio back, put the MPC in Controller Mode and run Auto Sampler as usual (written for MPC Live/MPC 2.x). — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1861977)
- Keygroup limits: a keygroup holds only four sample layers (switchable by velocity, round-robin, or random); stacking two or three keygroups on the same notes yields 8 or 12 velocity layers, counting against the 128-keygroup total; velocity switching is overridden once round-robin/random is selected. — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1876042)

**Open-source offline renderers usable on Windows 11 (to replace or script the sampling step)**
- pedalboard (spotify/pedalboard): GPL-3.0 (core audio code from JUCE 6), 6.4k stars; "Supports VST3 instrument and effect plugins on macOS, Windows, and Linux" (AU macOS only); `pedalboard.load_plugin` loads an instrument and renders MIDI by passing note messages (created with `mido`) with a duration and sample rate; Python 3.10–3.15; Windows wheels amd64 only, x86_64 requires AVX. — [GitHub: spotify/pedalboard](https://github.com/spotify/pedalboard)
- pedalboard caveats: issue #339 reports VST parameter changes reset to defaults when generating audio from MIDI; a third-party ComfyUI node built on it notes notes sustain for the full clip with release only in the final ~50 ms (secondary source). — [pedalboard issues](https://fr.github.com/spotify/pedalboard/issues); [ComfyUI VST Adapter](https://comfy.icu/node/ComfyUIVSTSynthAdapter)
- DawDreamer (DBraun/DawDreamer): GPL-3.0 ("to make it easier to comply with all of the dependent projects," including Steinberg VST2/3); 1.3k stars, 255 commits; "a Digital Audio Workstation for Python" with VST instruments/effects, parameter automation, Faust (effects and polyphonic instruments, Box/Signal APIs, transpile to JAX/C++/Rust/Wasm), JUCE processors; `make_plugin_processor` loads a VSTi, add MIDI notes, render, retrieve audio; VST state loading/saving supported; 64-bit Python 3.11–3.14 on macOS/Windows/Linux. — [GitHub: DBraun/DawDreamer](https://github.com/DBraun/DawDreamer); [PyPI](https://pypi.org/project/dawdreamer/0.6.9/)
- Surge XT exposes its internals as the `surgepy` Python module (`-DSURGE_BUILD_PYTHON_BINDINGS=ON`, examples in surge-python) — a direct headless route to render Surge patches without a plugin host. — [GitHub: surge-synthesizer/surge](https://github.com/surge-synthesizer/surge)
- Carla (falkTX): GPLv2+, hosts LADSPA/DSSI/LV2/VST2/VST3/AU, remote-controllable via OSC, Windows builds list VST3 and ship Carla as LV2/VST2 plugin; no documented command-line offline render found. — [GitHub: falkTX/Carla](https://github.com/falkTX/Carla); [Carla Windows README](https://git.kx.studio/falkTX/Carla/src/branch/main/data/windows)

**Open-source / community keygroup (.xpm) builders**
- MPC Valet: browser-based keygroup builder written in Rust, "a helper to create instruments that can run on MPC X, Live, One"; drop samples in and it generates the XPM with ranges set, though created programs "will always need fine tuning on MPCs"; published as the `mpc_valet` crate (v0.3.5 seen). — [MPC-Forums thread](https://www.mpc-forums.com/viewtopic.php?p=1846239); [crates.io mirror listing](https://crates-io.ganymede-static.progval.net/crates/mpc_valet/0.3.5)
- A Python script posted on MPC-Forums merges keygroup programs (takes Program/Instruments/Instrument structures from program 2 into program 1, saves a new .xpm; no error checking; tested on Force 3.1.3 and MPC One 2.10) — demonstrates the .xpm is plain XML that scripts can write. — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1850940)
- Instrumap is another community keygroup-creation tool mentioned on the forums (pitch detection on roadmap). — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1840050)
- The MPC Bible 3 (Revision 2, MPC 3.5 compatibility) is a paid community manual covering MPC 3 keygroup workflows. — [MPC-Forums](https://www.mpc-forums.com/viewtopic.php?p=1897464)

**Open-source sample-player formats as intermediate**
- sfizz (sfztools): BSD-2-Clause SFZ library with sfizz-ui providing AU, LV2, Pure Data and VST3 plugins; Reaper crashes reported in issues; no CLAP found. DSK SFZ Player is a separate free/open-source VST3/standalone SFZ player. — [GitHub: sfztools/sfizz-ui](https://github.com/sfztools/sfizz-ui); [Rekkerd](https://rekkerd.org/free-dsk-sfz-player-vst3-and-standalone/)

### Inferences
- A fully scriptable pipeline on Windows 11 that an LLM can write end-to-end: (1) LLM builds synth as VST3 (APC/JUCE) or as a Faust/Surge patch; (2) Python + pedalboard or DawDreamer iterates over notes (e.g. every 3rd semitone) × velocities, renders N-second WAVs with tails; (3) a script writes the .xpm (MPC Valet's generated files or the forum XML script serve as format references); (4) copy to the MPC One's SD/USB and load as a Keygroup program. The GUI Auto Sampler remains the zero-code fallback and is the only Akai-supported path.
- Four-layer-per-keygroup and 128-keygroup limits mean velocity sampling should be capped at 4 layers per key range unless keygroups are stacked.
- Since MPC 3.9 added MPC Oscillators to Keygroup tracks, a simple AI-designed subtractive/FM/wavetable patch could alternatively be re-created natively on the MPC One without sampling at all — but this is manual parameter transcription, not a file import.

### Gaps
- No public, maintained open-source tool was found that goes end-to-end from "VST3 plugin" to ".xpm Keygroup program" automatically; MPC Valet covers sample-to-xpm, pedalboard/DawDreamer cover plugin-to-WAV, and the gap must be bridged by a script.
- MPC Valet's GitHub repository URL and license were not retrieved (crates.io fetch returned no content).
- The official MPC 3 manual's Auto Sampler section was not fetched; steps above are from the MPC 2 support article and may differ cosmetically in MPC 3.
- Whether MPC Desktop's Auto Sampler can sample a plugin hosted in another DAW via loopback on Windows (vs the Mac/Ableton forum method) was not verified.
- No reliable source on the .xpm schema (field documentation) was found beyond community scripts.

---

## KQ4. Which AI tools generate synth presets or multisamples (open-source first; commercial brief)

### Takeaway
On the open-source side, AI "sound design" today is mostly neural capture of effects/amps (Neural Amp Modeler, GuitarML Proteus/SmartGuitarAmp, all free with Colab/Python trainers) and timbre transfer / neural synthesis (Magenta DDSP-VST, RAVE, Neutone), plus LLMs writing Faust/Cmajor/Surge patches directly. The notable commercial preset-from-audio tool is Sonic Charge Synplant 2 Genopatch (EUR 149); no official AI preset generator from Xfer (Serum 2), Arturia, or Native Instruments was found.

### Cited Findings

**Open-source neural amp/effect modelers (Windows VST3)**
- NeuralAmpModelerPlugin (sdatkinson): MIT license, built on iPlug2 (refactored from the original iPlug2-based NAM plugin), VST3 + AU + standalone, Windows 10 64-bit+ and macOS 10.15+, Linux via separate LV2 plugin; 3.0k stars, 303 commits, 95 open issues. — [GitHub: sdatkinson/NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin); [NAM trainer docs](https://neural-amp-modeler.readthedocs.io/)
- NAM community ports: kovala/nam-plug (iPlug2 VST3/AU "cleaned up" version); Mike Oliphant's neural-amp-modeler-lv2 for Linux (license listed as MIT by Fedora packaging but GPL-3.0 by a Guix patch — conflicting); NeuralBlender (GPLv3 app/LV2 on RTNeural + NAM). — [GitHub: kovala/nam-plug](https://github.com/kovala/nam-plug); [Guix issue](https://issues.guix.gnu.org/73091); [SourceForge NeuralBlender](https://sourceforge.net/projects/neuralblender/)
- A TONE3000 comparison (a competing platform, so treat with caution) says NAM "often has the edge in realism and touch sensitivity" over IK Multimedia TONEX; a Stork.ai listing claims "Neural Amp Modeler Architecture 2 (A2)" released early June 2026 and NAM integration in hardware from Valeton, Darkglass, NUX, Dimehead, Chaos Audio, Blackstar (aggregator; unverified against a primary source). — [TONE3000](https://www.tone3000.com/blog/nam-vs-tonex-guide-for-guitarists); [Stork.ai](https://www.stork.ai/tool/neural-amp-modeler)
- GuitarML Proteus: GPL-3.0, "Guitar amp and pedal capture plugin using neural networks" (LSTM); Windows/Mac/Linux installers; training via Automated-GuitarAmpModelling or the `ProteusCapture.ipynb` Google Colab with the Proteus Capture Utility — snapshot models ~10 min, knob models ~30 min (five WAVs); hidden size must be 40; 204 stars, 50 commits. — [GitHub: GuitarML/Proteus](https://github.com/GuitarML/Proteus)
- GuitarML SmartGuitarAmp: Apache-2.0, JUCE, WaveNet-based; training via PedalNetRT (PyTorch on WAV pairs); custom model loading dropped in v1.3 (use SmartGuitarPedal). NeuralPi is GuitarML's Raspberry Pi 4 DIY pedal with a cross-compiled VST3. — [GitHub: GuitarML/SmartGuitarAmp](https://github.com/GuitarML/SmartGuitarAmp); [AudioTechnology on NeuralPi](https://www.audiotechnology.com/free-stuff/guitarml-neuralpi)

**Open-source / free neural synthesis & timbre transfer**
- Magenta DDSP-VST: free, open-source AU/VST3 plugin for macOS and Windows; "anyone can train a model for free with only a few minutes of audio training data" via the web trainer; KVR lists v1.1.0. — [Magenta DDSP-VST](https://magenta.tensorflow.org/ddsp-vst); [KVR](https://www.kvraudio.com/product/ddsp-vst-by-magenta)
- Magenta Studio: MIDI tools for Ableton Live (Continue, Groove, Generate, Drumify, Interpolate) plus standalone; the official page found documents only v1.0 and calls itself documentation for an older version. — [Magenta Studio](https://magenta.tensorflow.org/studio/ableton-live/)
- Neutone: offers free AI audio VSTs; Neutone Morpho full version listed at $99 (Feb 2024 review); Cocoon model marketplace announced June 2024. — [AudioCipher](https://audiocipher.com/post/neutone)

**Open-source synths usable as sound sources (MPC Desktop controller mode, or headless render)**
- Surge XT: GPL-3.0; VST3, AU, standalone (LV2 optional, VST2 personal-use only); Windows/macOS/Linux; CMake + Visual Studio 2017/2019+ with "Desktop development with C++"; `surgepy` Python bindings; 4.1k stars, 5,673 commits. — [GitHub: surge-synthesizer/surge](https://github.com/surge-synthesizer/surge)
- Vital: GPL-3.0 source for the Vital wavetable synth; closed-source use requires a license from licensing@vital.audio; repo "updated on a delay after binary releases," author "will not take any pull requests" and does not prioritize compile issues; 17 commits, ~2.2k stars. — [GitHub: mtytel/vital](https://github.com/mtytel/vital)
- Dexed: free, GPLv3 DX7 emulation with VST/AU/LV2/standalone builds and Windows support (aggregator source). Odin 2 is open source for Windows/Mac/Linux (aggregator source). — [Dubspot comparison](https://blog.dubspot.com/plugins/compare/dexed-vs-surge-xt); [Dubspot Surge XT](https://blog.dubspot.com/plugins/surge-xt)
- Airwindows: MIT-licensed, "all the Airwindows audio open source code," 1.3k stars, 531 commits, repo is read-only (one PR ever) — effects, not instruments; formats not stated on the repo page. — [GitHub: airwindows/airwindows](https://github.com/airwindows/airwindows)

**Commercial AI preset / sample tools (brief)**
- Sonic Charge Synplant 2 "Genopatch" builds synth patches from an audio sample by searching for settings that match the source (synthesis, not sampling); select up to two seconds; mono analysis; refine in the DNA editor; EUR 149, EUR 50 upgrade; MusicRadar: "a genuinely unique, effective use of AI." — [MusicTech review](https://musictech.com/reviews/plug-ins/sonic-charges-synplant-2-review/); [MusicRadar review](https://musicradar.com/reviews/sonic-charge-synplant-2-review)
- Samplab 2: AI sample editor that alters notes in audio (Melodyne-like for loops); free basic + $9.99/month premium (2023). No multisample feature found. — [Sonicstate](https://sonicstate.com/news/2023/01/13/ai-powered-sample-editor/)
- Sononym: ML sample browser with similarity search, auto-tagging, duplicate detection, Windows/Linux/Mac, EUR 89 / $99. — [Sound On Sound](https://www.soundonsound.com/news/sononym-aim-make-sample-substitution-easy)
- Serum 2: wavetable/subtractive/multisample/granular/spectral synth, $249, 626 presets; only unverified third-party articles describe LLM text-to-.fxp preset generators; no official Xfer/Arturia/NI AI preset generator found. — [Xfer manual](https://xferrecords.com/web-manual/serum-2/welcome); [Dubspot](https://blog.dubspot.com/plugins/compare/pigments-vs-serum-2)
- IK Multimedia TONEX is the commercial counterpart to NAM, integrated with AmpliTube 5. — [TONE3000](https://www.tone3000.com/blog/nam-vs-tonex-guide-for-guitarists)

### Inferences
- For an MPC One owner, the most practical "AI-made instrument" today is not a preset generator but: Surge XT / Vital / Dexed presets (or LLM-authored Surge/Faust patches) multisampled into keygroups; and NAM/Proteus captures of real pedals/amps used as desktop-side effects (in controller mode) or printed to audio.
- DDSP-VST's "train on minutes of audio" is the closest open-source analogue to Genopatch but yields a timbre-transfer model, not a portable preset.

### Gaps
- No open-source "audio-to-synth-preset" (Genopatch-like) tool was found.
- The claimed NAM "Architecture 2" (June 2026) release could not be verified against the NAM GitHub/readthedocs.
- RAVE (IRCAM) status, license, and Windows plugin availability were not found in results.
- Last-commit dates for NAM, Proteus, Surge XT, Vital, Airwindows were not exposed in fetched pages.

---

## KQ5. Notable 2025–2026 projects where people built plugins with Claude/GPT and documented it

### Takeaway
The best-documented public cases are Joshua Hodge's The Audio Programmer Claude Code experiment (Feb 2026, JUCE gain plugin, video + article), Benjamin McCulloch's Sound On Sound JSFX sequencer tutorial (Jan 2026, Claude), and Noizefield's open-source Audio Plugin Coder project (Feb–Aug 2026, JUCE 9, 343 stars) which operationalises the workflow for Claude Code/Cursor/Codex; a Feb 2026 YouTube "Vibe Coding a VST Plug In with Claude Code" video also exists.

### Cited Findings
- The Audio Programmer, 26 Feb 2026: Claude Code + JUCE/CMake gain plugin with zero hand-written code; UI customisation eventually broke; companion ~29-minute video; follow-up series promised. — [The Audio Programmer](https://www.theaudioprogrammer.com/content/building-an-audio-plugin-with-claude-code-a-vibe-coding-experiment)
- Sound On Sound, Jan 2026: Claude-generated Reaper JSFX 303-style sequencer with 1–32 steps, grid UI, polyrhythm subdivision; per-step velocity never worked. — [Sound On Sound](https://www.soundonsound.com/techniques/vibe-coding-how-anyone-can-create-plug-ins)
- Noizefield Audio Plugin Coder: published 5 Feb 2026, updated 31 Aug 2026; MIT; JUCE 9; releases up to v1.8.0 (CLAP beta), v1.7.0 (subagent orchestration), v1.6.0 (UI layout contract, Devin CLI). — [Noizefield](https://www.noizefield.com/news/create-vst-plugins-with-ai-apc); [GitHub](https://github.com/Noizefield/audio-plugin-coder)
- YouTube, Feb 2026: "Vibe Coding a VST Plug In with Claude Code." — [YouTube](https://www.youtube.com/watch?v=6PHZc5oqeJc)
- Pamplejuce added CLAUDE.md/AGENTS.md agent guidance to its JUCE 8 template. — [Pamplejuce docs](https://docsearch.algolia.com/mcp/docs/repo/sudara/pamplejuce)

### Inferences
- Public write-ups skew toward simple effects (gain) and MIDI tools (sequencers); no documented case of an LLM producing a polished polyphonic synth instrument VST3 with custom GUI end-to-end was found, suggesting instrument-building remains the harder, less-proven case.

### Gaps
- No Hacker News thread, Bedroom Producers Blog article, or GitHub repo documenting a complete LLM-built synth instrument (as opposed to effect/sequencer) was found in this pass.
- No documented project combining LLM-built plugins with MPC keygroup export was found.
