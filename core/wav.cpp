#include "core/wav.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace mrt_local {
namespace {
template <typename T>
void append(std::vector<std::uint8_t>& out, T value) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}
void appendTag(std::vector<std::uint8_t>& out, const char* tag) {
  out.insert(out.end(), tag, tag + 4);
}
}  // namespace

std::vector<std::uint8_t> encodeWav(const GenerateResult& audio) {
  if (audio.sample_rate <= 0 || audio.channels <= 0 ||
      audio.audio.size() % static_cast<std::size_t>(audio.channels) != 0) {
    throw std::invalid_argument("invalid audio format");
  }
  const std::uint64_t bytes64 = audio.audio.size() * sizeof(float);
  if (bytes64 > std::numeric_limits<std::uint32_t>::max() - 36ULL) {
    throw std::length_error("audio is too large for RIFF/WAV");
  }
  const auto data_size = static_cast<std::uint32_t>(bytes64);
  const auto channels = static_cast<std::uint16_t>(audio.channels);
  constexpr std::uint16_t bits = 32;
  const auto block_align = static_cast<std::uint16_t>(channels * sizeof(float));
  const auto byte_rate = static_cast<std::uint32_t>(audio.sample_rate) * block_align;

  std::vector<std::uint8_t> out;
  out.reserve(44 + data_size);
  appendTag(out, "RIFF"); append(out, std::uint32_t{36 + data_size});
  appendTag(out, "WAVE"); appendTag(out, "fmt "); append(out, std::uint32_t{16});
  append(out, std::uint16_t{3});  // IEEE float PCM
  append(out, channels); append(out, static_cast<std::uint32_t>(audio.sample_rate));
  append(out, byte_rate); append(out, block_align); append(out, bits);
  appendTag(out, "data"); append(out, data_size);
  const auto* samples = reinterpret_cast<const std::uint8_t*>(audio.audio.data());
  out.insert(out.end(), samples, samples + data_size);
  return out;
}

void writeWav(const std::filesystem::path& path, const GenerateResult& audio) {
  const auto bytes = encodeWav(audio);
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("Cannot open WAV output: " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("Failed to write WAV output: " + path.string());
}

}  // namespace mrt_local
