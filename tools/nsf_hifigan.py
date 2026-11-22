"""NSF-HiFiGAN generator, parameter-compatible with OpenVPI checkpoints
(DiffSinger / openvpi/vocoders, e.g. nsf_hifigan_44.1k_hop512_128bin_2024.02),
plus the reference log-mel front-end those models were trained with.

Only used offline: to export ONNX for the plugin and to verify that the
plugin's C++ mel matches the training front-end bit-for-bit (within float error).
"""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1

DEFAULT_CONFIG = {
    "resblock": "1",
    "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "num_mels": 128,
    "sampling_rate": 44100,
    "hop_size": 512,
    "n_fft": 2048,
    "win_size": 2048,
    "fmin": 40,
    "fmax": 16000,
}


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def get_padding(k, d=1):
    return (k * d - d) // 2


class ResBlock1(nn.Module):
    def __init__(self, ch, k=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(ch, ch, k, 1, dilation=d, padding=get_padding(k, d))) for d in dilation])
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(ch, ch, k, 1, dilation=1, padding=get_padding(k, 1))) for _ in dilation])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in list(self.convs1) + list(self.convs2):
            remove_weight_norm(c)


class ResBlock2(nn.Module):
    def __init__(self, ch, k=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([weight_norm(Conv1d(ch, ch, k, 1, dilation=d, padding=get_padding(k, d))) for d in dilation])

    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x

    def remove_weight_norm(self):
        for c in self.convs:
            remove_weight_norm(c)


class SineGen(nn.Module):
    """Harmonic sine excitation from f0 (the 'NSF' in NSF-HiFiGAN). Unvoiced (f0 = 0) -> noise."""

    def __init__(self, sr, harmonic_num=8, sine_amp=0.1, noise_std=0.003, voiced_threshold=0.0):
        super().__init__()
        self.sr, self.dim = sr, harmonic_num + 1
        self.sine_amp, self.noise_std, self.voiced_threshold = sine_amp, noise_std, voiced_threshold

    def forward(self, f0, upp):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)  # [B, T, 1]
            rad = f0 / self.sr * torch.arange(1, upp + 1, device=f0.device, dtype=f0.dtype)  # per-sample phase step, frame-wise
            rad2 = torch.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
            rad_acc = rad2.cumsum(dim=1).fmod(1.0)
            rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
            rad = rad.reshape(f0.shape[0], -1, 1)
            rad = rad * torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype).reshape(1, 1, -1)
            sines = torch.sin(2 * math.pi * rad) * self.sine_amp
            uv = (f0 > self.voiced_threshold).to(f0.dtype)
            uv = uv.repeat_interleave(upp, dim=1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            return sines * uv + noise_amp * torch.randn_like(sines)


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num=8):
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0, upp):
        return self.l_tanh(self.l_linear(self.l_sin_gen(f0, upp)))


class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.m_source = SourceModuleHnNSF(h.sampling_rate, harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i), c_cur, k, u, padding=(k - u) // 2)))
            if i + 1 < len(h.upsample_rates):
                s = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=s * 2, stride=s, padding=s // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x, f0):
        """x: mel [B, num_mels, T]; f0: [B, T] Hz (0 = unvoiced) -> [B, 1, T * upp]"""
        har = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            x = x + self.noise_convs[i](har)
            xs = None
            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    return AttrDict(cfg)


def slaney_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """Same as librosa.filters.mel(htk=False, norm='slaney')."""
    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        f_sp, min_log_hz = 200.0 / 3, 1000.0
        logstep = np.log(6.4) / 27.0
        return np.where(f >= min_log_hz, min_log_hz / f_sp + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f_sp, min_log_hz = 200.0 / 3, 1000.0
        min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    fftfreqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (hz[2:n_mels + 2] - hz[:n_mels]))[:, None]
    return w.astype(np.float32)


def reference_mel(y, h):
    """OpenVPI training front-end. y: float32 [N] at h.sampling_rate -> [T, num_mels] natural-log mel."""
    y = torch.as_tensor(y, dtype=torch.float32)[None, None]
    basis = torch.from_numpy(slaney_mel_filterbank(h.sampling_rate, h.n_fft, h.num_mels, h.fmin, h.fmax))
    pad_l, pad_r = (h.n_fft - h.hop_size) // 2, (h.n_fft - h.hop_size + 1) // 2
    y = F.pad(y, (pad_l, pad_r), mode="reflect")[:, 0]
    spec = torch.stft(y, h.n_fft, hop_length=h.hop_size, win_length=h.win_size,
                      window=torch.hann_window(h.win_size), center=False, return_complex=True)
    spec = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
    mel = torch.log(torch.clamp(basis @ spec, min=1e-5))
    return mel[0].T.numpy()
