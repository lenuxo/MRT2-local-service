#include "core/engine.h"
#include "core/wav.h"
#include "server/server.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
struct Options {
  std::string command;
  std::string prompt;
  std::string model = "mrt2_small";
  float duration = 10.0f;
  std::filesystem::path output = "./output.wav";
  std::string model_path;
  std::string resources_path;
  std::string host = "127.0.0.1";
  int port = 8765;
};

std::string configuredPath(const std::string& cli, const char* env,
                           const std::string& fallback) {
  if (!cli.empty()) return cli;
  if (const char* value = std::getenv(env); value && *value) return value;
  return fallback;
}

void usage(std::ostream& out) {
  out << "MRT Local " << MRT_LOCAL_VERSION << "\n\n"
      << "Usage:\n"
      << "  mrt generate --prompt TEXT [--duration SECONDS] [--output FILE]\n"
      << "               [--model mrt2_small|mrt2_base] [--model-path PATH]\n"
      << "               [--resources-path PATH]\n"
      << "  mrt serve [--host 127.0.0.1] [--port 8765]\n"
      << "            [--model mrt2_small|mrt2_base] [--model-path PATH]\n"
      << "            [--resources-path PATH]\n"
      << "  mrt info [--model-path PATH] [--resources-path PATH]\n\n"
      << "Model path precedence: --model-path, MRT_MODEL_PATH, project models/.\n";
}

std::string nextValue(int& index, int argc, char** argv, const std::string& flag) {
  if (++index >= argc) throw std::invalid_argument(flag + " requires a value");
  return argv[index];
}

Options parse(int argc, char** argv) {
  if (argc < 2) throw std::invalid_argument("missing command");
  Options options;
  options.command = argv[1];
  if (options.command == "--help" || options.command == "-h") return options;
  if (options.command != "generate" && options.command != "serve" &&
      options.command != "info") {
    throw std::invalid_argument("unknown command: " + options.command);
  }
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { options.command = "--help"; break; }
    if (arg == "--prompt") options.prompt = nextValue(i, argc, argv, arg);
    else if (arg == "--duration") options.duration = std::stof(nextValue(i, argc, argv, arg));
    else if (arg == "--output") options.output = nextValue(i, argc, argv, arg);
    else if (arg == "--model-path") options.model_path = nextValue(i, argc, argv, arg);
    else if (arg == "--resources-path") options.resources_path = nextValue(i, argc, argv, arg);
    else if (arg == "--model") {
      options.model = nextValue(i, argc, argv, arg);
      if (options.model != "mrt2_small" && options.model != "mrt2_base")
        throw std::invalid_argument("--model must be mrt2_small or mrt2_base");
    } else if (arg == "--host") options.host = nextValue(i, argc, argv, arg);
    else if (arg == "--port") options.port = std::stoi(nextValue(i, argc, argv, arg));
    else throw std::invalid_argument("unknown option: " + arg);
  }
  if (options.port < 1 || options.port > 65535) throw std::invalid_argument("port must be 1..65535");
  return options;
}

void printMissingModel(const std::string& model) {
  std::cerr << "MRT2 model not found.\n\nSpecify one using:\n\n"
            << "./scripts/download_models.sh " << model << "\n\nor:\n\n"
            << "mrt serve --model-path /path/to/" << model << ".mlxfn\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    auto options = parse(argc, argv);
    if (options.command == "--help") { usage(std::cout); return 0; }
    const auto model_root = configuredPath("", "MRT_MODEL_ROOT", MRT_LOCAL_DEFAULT_MODEL_ROOT);
    options.model_path = configuredPath(options.model_path, "MRT_MODEL_PATH",
        model_root + "/models/" + options.model + "/" + options.model + ".mlxfn");
    options.resources_path = configuredPath(options.resources_path, "MRT_RESOURCES_PATH",
        model_root + "/resources");

    if (options.command == "info") {
      std::cout << "MRT Local\n\nPlatform: macOS arm64\nBackend: MLX\n"
                << "Model: " << options.model << "\nModel path: " << options.model_path
                << "\nResources path: " << options.resources_path
                << "\nSample rate: 48000\n";
      return 0;
    }
    if (!std::filesystem::exists(options.model_path)) { printMissingModel(options.model); return 2; }
    mrt_local::MrtEngine engine({options.model, options.model_path, options.resources_path, 48000});
    engine.load();
    if (options.command == "serve") return mrt_local::runServer(engine, options.host, options.port);
    if (options.prompt.empty()) throw std::invalid_argument("generate requires --prompt");

    auto generated = engine.generate(options.prompt, options.duration);
    mrt_local::writeWav(options.output, generated);
    std::cout << "Model: " << options.model << "\nPrompt: " << options.prompt
              << "\nDuration: " << options.duration << "s\n\nGenerated: "
              << options.output.string() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    usage(std::cerr);
    return 1;
  }
}
