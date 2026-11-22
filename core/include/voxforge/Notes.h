#pragma once
#include "voxforge/Types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace voxforge {

struct Note {
    int start = 0;          // first frame (inclusive)
    int end = 0;            // last frame (exclusive)
    float pitch = 0.0f;     // sung centre, fractional MIDI (median of the note)
    float target = 0.0f;    // chosen target, integer MIDI (set by the corrector)
    float strength = 1.0f;  // 0..1 how strongly this note is corrected
    int length() const { return end - start; }
};

struct SegmenterConfig {
    double frameRate = kFrameRate;
    float smoothMs = 150.0f;        // averages out vibrato (>= ~1 cycle at 6-7 Hz)
    float splitSemitones = 0.7f;    // sustained departure that starts a new note
    float splitHoldMs = 35.0f;      // how long the departure must last
    float glideMergeMs = 60.0f;     // transition fragments up to this long are merged
};

// Split a pitch contour into notes. Vibrato and slow drift stay inside a note;
// only sustained moves of more than `splitSemitones` start a new one.
std::vector<Note> segmentNotes(const std::vector<float>& f0Hz, const SegmenterConfig& cfg = {});

enum class ScaleType : int {
    Chromatic = 0, Major, NaturalMinor, HarmonicMinor, MajorPentatonic,
    MinorPentatonic, Blues, Dorian, Mixolydian, Count
};

struct Key {
    int root = 0;                       // 0 = C
    ScaleType scale = ScaleType::Major;
    float confidence = 0.0f;            // correlation of the best profile, -1..1
};

uint16_t scaleMask(int root, ScaleType scale);   // bit i set = pitch class i allowed
inline bool inScale(uint16_t mask, int midi) { return (mask >> (((midi % 12) + 12) % 12)) & 1; }
std::string keyName(const Key& key);
const char* scaleName(ScaleType s);

// Krumhansl-Kessler key finding over a duration-weighted, soft pitch-class
// histogram of the sung notes. Reference-free: no MIDI or guide track needed.
Key detectKey(const std::vector<Note>& notes);

} // namespace voxforge
