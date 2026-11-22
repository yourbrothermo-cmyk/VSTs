# Architecture

## Signal flow (Studio)

```
capture (host rate) ─► resample 44.1k ─► YIN pitch (hop 512) ─► note segmentation ─► key detection
                                     └─► log-mel (NSF-HiFiGAN recipe)
                                                     │
CorrectionParams ─► Viterbi target choice ─► centre/residual split ─► corrected f0
                                                     │
                       mel (+ optional formant warp) + corrected f0 ─► NSF-HiFiGAN (chunked, crossfaded)
                                                     │                    └ or TD-PSOLA fallback
                                 Hybrid blend with original ─► resample to host rate ─► timeline playback
```

Analysis runs once per take. Correction takes milliseconds. Rendering is the only
expensive step, and it runs on a worker thread, is cancellable and is debounced.

## Key algorithms

* **Pitch**: YIN with an FFT difference function, parabolic interpolation, then
  octave-error repair, hole filling, blip removal and a 3-tap median. Frame *t* is
  centred at `t·512 + 256`, the same grid as the mel frames.
* **Notes**: a running median over about 150 ms removes vibrato but keeps note
  edges sharp. A sustained departure of more than 0.7 semitones starts a note.
  Glide fragments shorter than 60 ms are merged into a neighbour.
* **Targets**: per note, the candidates are round(pitch) ± 2. The unary cost is
  4·dev² plus an out-of-key penalty (×0.1 for long notes sung within the
  accidental tolerance). The pairwise cost is 2·context·(sung interval −
  candidate interval)² plus a small tritone/major-7th penalty. Pairwise links are
  cut across rests longer than 600 ms.
* **Curve**: shift = amount·strength·(target − anchor), where the anchor mixes the
  per-frame centre line with the note median (*Drift*). Unvoiced gaps hold the
  nearest shift, so onsets are corrected immediately. A zero-phase one-pole
  (*Retune*) smooths only note-to-note glides.

## Live latency budget

* **Live (DSP)**: about 3.25 × the period of the lowest note, plus 80 samples.
  That is about 42 ms at 80 Hz and about 34 ms at 100 Hz.
* **Live HQ (neural)**: chunk K=2 frames, lookahead R=2, mel right-window 1280
  samples, one chunk of compute headroom, resamplers and one host block. That
  comes to about 90–110 ms. If inference can't keep up, the engine emits
  silence, counts underruns and keeps the stream aligned.

## Thread safety

The audio thread never allocates or locks. Studio renders are handed over with
a pointer-publish/acknowledge scheme (`StudioSession::publishRender`). Live HQ
uses SPSC FIFOs and a `try_lock` for parameter updates. Mode and latency changes
happen on the message thread inside `suspendProcessing`.

## Verified here (Linux CI-equivalent)

* 15 core unit tests covering FFT, pitch, mel, formant warp, both resamplers,
  segmentation, key detection (including off-tuned singers), context targets,
  ornaments, vibrato preservation, PSOLA, the offline engine and both live
  engines (the neural one with a stub vocoder).
* Headless plugin test: capture → render → playback through the real
  processor with a fake playhead, state save/restore, and Live mode.
* Mel parity against a PyTorch reference of the training front-end: max error
  1.2e-4. An exported NSF-HiFiGAN ONNX runs on C++ features with correct output length.

## Roadmap

1. **ARA2** (Logic supports it): direct access to the region audio, with no capture pass.
2. A per-note editor (drag targets, per-note amount and vibrato), stored in the session.
3. Incremental re-render of only the changed regions.
4. Train or license a commercial-use vocoder. Optionally fine-tune it on
   pitch-shifted data so it stays robust beyond ±1 octave.
5. A neural pitch tracker (e.g. RMVPE/CREPE-style) as an option for noisy or breathy takes.
