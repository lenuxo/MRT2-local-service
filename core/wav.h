#pragma once

#include "core/engine.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace mrt_local {

std::vector<std::uint8_t> encodeWav(const GenerateResult& audio);
void writeWav(const std::filesystem::path& path, const GenerateResult& audio);

}  // namespace mrt_local
