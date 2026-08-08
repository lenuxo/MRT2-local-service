#include "core/wav.h"

#include <cassert>
#include <cstring>

int main() {
  mrt_local::GenerateResult audio{48000, 2, {0.0f, 0.5f, -0.5f, 1.0f}};
  const auto wav = mrt_local::encodeWav(audio);
  assert(wav.size() == 44 + audio.audio.size() * sizeof(float));
  assert(std::memcmp(wav.data(), "RIFF", 4) == 0);
  assert(std::memcmp(wav.data() + 8, "WAVE", 4) == 0);
  assert(std::memcmp(wav.data() + 36, "data", 4) == 0);
  return 0;
}
