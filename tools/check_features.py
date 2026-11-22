#!/usr/bin/env python3
"""Verify the plugin's C++ front-end against the NSF-HiFiGAN training recipe,
and optionally run an ONNX vocoder on the C++ features.

    voxforge_cli take.wav out.wav --dump /tmp/take
    python tools/check_features.py /tmp/take [--onnx nsf_hifigan.onnx --out vocoded.wav]
"""
import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsf_hifigan import load_config, reference_mel  # noqa: E402




def load_float_wav(path):
    data = open(path, "rb").read()
    i = data.index(b"data")
    n = int.from_bytes(data[i + 4:i + 8], "little")
    return np.frombuffer(data[i + 8:i + 8 + n], dtype="<f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--onnx")
    ap.add_argument("--out")
    a = ap.parse_args()
    h = load_config()

    audio = load_float_wav(a.prefix + ".44k.wav")
    mel_cpp = np.fromfile(a.prefix + ".mel.f32", dtype="<f4").reshape(-1, h.num_mels)
    mel_ref = reference_mel(audio, h)
    assert mel_cpp.shape == mel_ref.shape, (mel_cpp.shape, mel_ref.shape)
    err = np.abs(mel_cpp - mel_ref)
    loud = mel_ref > np.log(1e-3)  # ignore the clamp floor, where float noise dominates
    print(f"mel frames {mel_cpp.shape[0]}: max |diff| {err.max():.2e} (all), {err[loud].max():.2e} (above -60 dB)")
    ok = err[loud].max() < 1e-2

    if a.onnx:
        import onnxruntime as ort
        f0 = np.fromfile(a.prefix + ".f0c.f32", dtype="<f4")
        sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
        names = {i.name: i.shape for i in sess.get_inputs()}
        print("onnx inputs:", names, "outputs:", [o.name for o in sess.get_outputs()])
        wav = sess.run(None, {"mel": mel_cpp[None], "f0": f0[None]})[0].reshape(-1)
        print(f"vocoder output {wav.shape[0]} samples (expected {mel_cpp.shape[0] * h.hop_size})")
        ok = ok and wav.shape[0] == mel_cpp.shape[0] * h.hop_size and np.isfinite(wav).all()
        if a.out:
            with wave.open(a.out, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(h.sampling_rate)
                w.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
