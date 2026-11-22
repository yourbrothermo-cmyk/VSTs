#!/usr/bin/env python3
"""Export an NSF-HiFiGAN checkpoint to the ONNX contract the plugin expects:

    inputs : mel [1, T, 128] (float32, natural-log mel), f0 [1, T] (Hz, 0 = unvoiced)
    output : waveform [1, T * 512] at 44.1 kHz

If you already have the OpenVPI/OpenUtau `nsf_hifigan.onnx`, you do not need
this script: that file uses the same contract and the plugin loads it as-is.

    python tools/export_nsf_hifigan_onnx.py --ckpt model.ckpt --config config.json --out nsf_hifigan.onnx
    python tools/export_nsf_hifigan_onnx.py --random --out test_vocoder.onnx   # untrained, for plumbing tests
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsf_hifigan import Generator, load_config  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, mel, f0):
        return self.gen(mel.transpose(1, 2), f0).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", action="store_true", help="export untrained weights (testing only)")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    h = load_config(a.config)
    gen = Generator(h)
    if not a.random:
        if not a.ckpt:
            ap.error("--ckpt is required unless --random")
        state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        state = state.get("generator", state.get("state_dict", state))
        state = {k.removeprefix("model.").removeprefix("generator."): v for k, v in state.items()}
        missing, unexpected = gen.load_state_dict(state, strict=False)
        if missing:
            sys.exit(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
        if unexpected:
            print(f"warning: ignoring {len(unexpected)} unexpected tensors, e.g. {unexpected[:3]}")
    gen.eval()
    gen.remove_weight_norm()

    T = 64
    mel = torch.randn(1, T, h.num_mels)
    f0 = torch.full((1, T), 220.0)
    torch.onnx.export(Wrapper(gen), (mel, f0), a.out, input_names=["mel", "f0"], output_names=["waveform"],
                      dynamic_axes={"mel": {1: "n_frames"}, "f0": {1: "n_frames"}, "waveform": {1: "n_samples"}},
                      opset_version=a.opset, dynamo=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
