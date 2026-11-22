# VoxForge

Post-recording vocal pitch correction with neural resynthesis, built as an
**Audio Unit** (Logic Pro), **VST3** (other hosts) and a Standalone app.

| Feature | How it works |
|---|---|
| **Vibrato preservation** | Each note's pitch is split into a slow centre line and a residual (vibrato, scoops, micro-pitch). Only the centre moves; the residual is kept, scaled by the *Vibrato* knob (0 % flattens, 200 % exaggerates). |
| **Context-aware note correction** | Targets are chosen by a Viterbi search over the whole take. It weighs distance to the sung pitch, key membership (accurately sung accidentals are kept), and how well each candidate keeps the melodic interval the singer actually sang. So a phrase sung uniformly flat keeps its shape instead of splitting between scale degrees. Short ornaments and transitions are corrected less. |
| **Reference-free** | There's no guide track or MIDI. The key is detected from the take (Krumhansl–Kessler, after removing the singer's global tuning offset), or set manually. |
| **Neural resynthesis** | NSF-HiFiGAN regenerates the voice from (log-mel, corrected f0). The mel front-end matches the model's training recipe to about 1e-4 (verified by `tools/check_features.py`). |
| **Formant preservation** | The mel spectrogram (the vocal-tract envelope) is left untouched while f0 changes, so formants stay put even under large shifts. *Formant* shifts them on purpose by warping the mel axis. |
| **Hybrid mode** | Only frames whose pitch actually changed go through the vocoder. Consonants, breaths and in-tune notes stay as recorded, with 25 ms crossfades. |
| **Live latency modes** | *Live* uses streaming YIN, a causal corrector and TD-PSOLA. It needs no model and has about 42 ms of latency (set by *Lowest Note*). *Live HQ* streams NSF-HiFiGAN in small chunks on a worker thread with about 90–110 ms of fixed latency, reported to the host. |

## Workflow (Studio mode)

1. Insert VoxForge on the vocal track and click **Capture**.
2. Play the vocal section through once. VoxForge records it against the timeline.
3. Stop the transport and click **Capture** again to disarm. The take is analysed
   (pitch, notes, key) and rendered in the background.
4. Play back. The corrected vocal is played sample-aligned with zero latency.
   Every knob change re-renders, and the pitch display shows the sung curve,
   the corrected curve and the chosen note targets.

Takes are saved with the project (a WAV under the user app-data folder,
referenced from the plugin state).

## Building

Requirements: CMake ≥ 3.22, a C++17 compiler, and network access to fetch JUCE 8.0.4 (or pass `-DJUCE_DIR=/path/to/JUCE`).

```bash
# macOS, with the neural vocoder (download onnxruntime-osx-universal2-<ver>.tgz from the ONNX Runtime releases)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVOXFORGE_WITH_ONNX=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build
ctest --test-dir build --output-on-failure
cp -R build/plugin/VoxForge_artefacts/Release/AU/VoxForge.component ~/Library/Audio/Plug-Ins/Components/
```

Without `-DVOXFORGE_WITH_ONNX=ON` everything still builds and runs on the DSP
(TD-PSOLA) path. `.github/workflows/build.yml` builds the AU/VST3 on macOS,
runs `auval`, and uploads the bundles.

`VOXFORGE_WITH_COREML=ON` enables ONNX Runtime's CoreML provider (Neural Engine/GPU) when your ORT build includes it.

## The vocoder model

The plugin loads any NSF-HiFiGAN ONNX file with this contract:
`mel [1,T,128]` (natural-log mel, 44.1 kHz, hop 512, 40–16000 Hz) plus `f0 [1,T]`
(Hz, 0 = unvoiced) → `waveform [1,T·512]`. Layout `[1,128,T]` is also detected.

* The OpenVPI `nsf_hifigan_44.1k_hop512_128bin` release (also shipped as an ONNX
  file in OpenUtau/DiffSinger vocoder packages) uses exactly this contract.
  **Its weights are licensed CC BY-NC-SA 4.0 (non-commercial).** That is fine for
  personal use, but a commercial release needs a vocoder you train yourself or
  license. The architecture and this code don't have that restriction.
* `tools/export_nsf_hifigan_onnx.py` exports any OpenVPI-format checkpoint
  (or random weights with `--random`, for plumbing tests) to that contract.
* Load it with **Load model…**. The model path is saved in the session.

## Command-line tool

`voxforge_cli` runs the same Studio pipeline on WAV files:

```bash
build/tests/voxforge_cli take.wav out.wav --model nsf_hifigan.onnx --mode hybrid --key auto --retune 40
build/tests/voxforge_cli take.wav out.wav --dump /tmp/take && python tools/check_features.py /tmp/take --onnx nsf_hifigan.onnx
```

## Layout

```
core/      dependency-free C++17 DSP/ML engine (pitch, notes, key, corrector, mel, vocoders, live engines)
plugin/    JUCE AU/VST3/Standalone wrapper, Studio capture session, editor, headless integration test
tests/     core unit tests + CLI
tools/     PyTorch NSF-HiFiGAN (checkpoint-compatible), ONNX exporter, feature-parity checker
docs/      ARCHITECTURE.md (design, latency budget, roadmap)
```

## Host notes

* **Logic Pro** loads Audio Units only (not VST3); the AU is built on macOS.
* **Splice Create** is Splice's new DAW (beta, macOS). I could not confirm whether
  it hosts third-party AU/VST3 plugins; if it does, the same bundles apply.
