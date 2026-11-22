#pragma once
// Minimal WAV reader/writer for the CLI (PCM 16/24/32 and float32; mixes to mono).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct WavData { double sampleRate = 44100; std::vector<float> mono; };

inline bool readWav(const std::string& path, WavData& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> d;
    uint8_t buf[65536];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof buf, f)) > 0) d.insert(d.end(), buf, buf + r);
    std::fclose(f);
    auto u16 = [&](size_t o) { return uint16_t(d[o] | d[o + 1] << 8); };
    auto u32 = [&](size_t o) { return uint32_t(d[o] | d[o + 1] << 8 | d[o + 2] << 16 | uint32_t(d[o + 3]) << 24); };
    if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4)) { err = "not a WAV file"; return false; }
    int fmt = 0, ch = 0, bits = 0;
    for (size_t o = 12; o + 8 <= d.size();) {
        const uint32_t sz = u32(o + 4);
        if (!std::memcmp(d.data() + o, "fmt ", 4)) {
            fmt = u16(o + 8); ch = u16(o + 10); out.sampleRate = u32(o + 12); bits = u16(o + 22);
            if (fmt == 0xFFFE) fmt = u16(o + 32);
        } else if (!std::memcmp(d.data() + o, "data", 4)) {
            const size_t bps = size_t(bits / 8), frames = std::min<size_t>(sz, d.size() - o - 8) / (bps * size_t(ch));
            out.mono.assign(frames, 0.0f);
            for (size_t i = 0; i < frames; ++i) {
                float acc = 0;
                for (int c = 0; c < ch; ++c) {
                    const uint8_t* p = d.data() + o + 8 + (i * size_t(ch) + size_t(c)) * bps;
                    float v = 0;
                    if (fmt == 3 && bits == 32) std::memcpy(&v, p, 4);
                    else if (bits == 16) v = int16_t(p[0] | p[1] << 8) / 32768.0f;
                    else if (bits == 24) v = float(int32_t(uint32_t(p[0]) << 8 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 24) >> 8) / 8388608.0f;
                    else if (bits == 32) v = float(int32_t(u32(size_t(p - d.data())))) / 2147483648.0f;
                    acc += v;
                }
                out.mono[i] = acc / float(ch);
            }
            return true;
        }
        o += 8 + sz + (sz & 1);
    }
    err = "no data chunk";
    return false;
}

inline bool writeWav(const std::string& path, const std::vector<float>& x, double sr) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
    const uint32_t dataBytes = uint32_t(x.size() * 4);
    std::fwrite("RIFF", 1, 4, f); w32(36 + dataBytes); std::fwrite("WAVEfmt ", 1, 8, f);
    w32(16); w16(3); w16(1); w32(uint32_t(sr)); w32(uint32_t(sr) * 4); w16(4); w16(32);
    std::fwrite("data", 1, 4, f); w32(dataBytes);
    std::fwrite(x.data(), 4, x.size(), f);
    std::fclose(f);
    return true;
}
