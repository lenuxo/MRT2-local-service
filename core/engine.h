#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mrt_local {

struct EngineConfig {
  std::string model_name = "mrt2_small";
  std::string model_path;
  std::string resources_path;
  int sample_rate = 48000;
};

struct GenerateResult {
  int sample_rate = 48000;
  int channels = 2;
  std::vector<float> audio;
};

class MrtEngine {
 public:
  explicit MrtEngine(EngineConfig config);
  ~MrtEngine();
  MrtEngine(const MrtEngine&) = delete;
  MrtEngine& operator=(const MrtEngine&) = delete;

  void load();
  GenerateResult generate(const std::string& prompt, float duration_seconds);
  bool isLoaded() const;
  const EngineConfig& config() const { return config_; }

 private:
  struct Impl;
  EngineConfig config_;
  std::unique_ptr<Impl> impl_;
  mutable std::mutex mutex_;
};

}  // namespace mrt_local
