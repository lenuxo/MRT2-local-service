#include "core/engine.h"

#include <magentart/mlx_engine.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace mrt_local {

struct MrtEngine::Impl {
  magentart::core::MLXEngine engine;
};

MrtEngine::MrtEngine(EngineConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>()) {
  if (config_.model_name != "mrt2_small" && config_.model_name != "mrt2_base") {
    throw std::invalid_argument("model must be mrt2_small or mrt2_base");
  }
  if (config_.sample_rate != 48000) {
    throw std::invalid_argument("MRT2 only supports 48000 Hz output");
  }
}

MrtEngine::~MrtEngine() = default;

void MrtEngine::load() {
  std::lock_guard lock(mutex_);
  if (impl_->engine.is_loaded()) return;
  if (!std::filesystem::exists(config_.model_path)) {
    throw std::runtime_error("MRT2 model not found: " + config_.model_path);
  }
  if (!std::filesystem::is_directory(config_.resources_path)) {
    throw std::runtime_error("MRT2 resources not found: " + config_.resources_path);
  }
  if (!impl_->engine.init_assets(config_.resources_path.c_str(), "musiccoca")) {
    throw std::runtime_error("Failed to initialize MusicCoCa assets from: " +
                             config_.resources_path);
  }
  if (!impl_->engine.load_model(config_.model_path.c_str())) {
    throw std::runtime_error("Failed to load MRT2 model: " + config_.model_path);
  }
}

GenerateResult MrtEngine::generate(const std::string& prompt,
                                   float duration_seconds) {
  if (prompt.empty()) throw std::invalid_argument("prompt must not be empty");
  if (!std::isfinite(duration_seconds) || duration_seconds <= 0.0f) {
    throw std::invalid_argument("duration must be a positive finite number");
  }
  if (duration_seconds > 300.0f) {
    throw std::invalid_argument("duration must not exceed 300 seconds");
  }

  std::lock_guard lock(mutex_);
  if (!impl_->engine.is_loaded()) throw std::logic_error("MRT2 model is not loaded");

  impl_->engine.reset_state();
  impl_->engine.set_text_prompt(prompt);
  constexpr auto kPromptTimeout = std::chrono::seconds(60);
  const auto deadline = std::chrono::steady_clock::now() + kPromptTimeout;
  while (impl_->engine.get_text_encoder_status() == 1 ||
         impl_->engine.get_quantizer_status() == 1) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("Timed out while encoding text prompt");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (impl_->engine.get_text_encoder_status() == 3 ||
      impl_->engine.get_quantizer_status() == 3) {
    throw std::runtime_error("Failed to encode text prompt");
  }

  const auto sample_frames = static_cast<std::size_t>(
      std::llround(static_cast<double>(duration_seconds) * config_.sample_rate));
  const auto model_frames = (sample_frames + magentart::core::kFrameSamples - 1) /
                            magentart::core::kFrameSamples;
  GenerateResult result{config_.sample_rate,
                        static_cast<int>(magentart::core::kNumChannels), {}};
  result.audio.reserve(model_frames * magentart::core::kFrameSamples * 2);
  std::vector<float> left(magentart::core::kFrameSamples);
  std::vector<float> right(magentart::core::kFrameSamples);
  for (std::size_t frame = 0; frame < model_frames; ++frame) {
    if (!impl_->engine.generate_frame(left.data(), right.data())) {
      throw std::runtime_error("MRT2 generation failed at frame " +
                               std::to_string(frame));
    }
    for (std::size_t i = 0; i < magentart::core::kFrameSamples; ++i) {
      result.audio.push_back(left[i]);
      result.audio.push_back(right[i]);
    }
  }
  result.audio.resize(sample_frames * 2);
  return result;
}

bool MrtEngine::isLoaded() const {
  std::lock_guard lock(mutex_);
  return impl_->engine.is_loaded();
}

}  // namespace mrt_local
