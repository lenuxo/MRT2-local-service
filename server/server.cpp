#include "server/server.h"

#include "core/wav.h"
#include "server/openapi.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <iostream>

namespace mrt_local {
namespace {
void jsonResponse(httplib::Response& response, const nlohmann::json& body,
                  int status = 200) {
  response.status = status;
  response.set_content(body.dump(2) + "\n", "application/json");
}
}  // namespace

int runServer(MrtEngine& engine, const std::string& host, int port) {
  httplib::Server server;
  server.set_payload_max_length(64 * 1024);

  server.Get("/health", [&engine](const httplib::Request&, httplib::Response& res) {
    jsonResponse(res, {{"status", "ok"}, {"model", engine.config().model_name},
                       {"loaded", engine.isLoaded()}});
  });
  server.Get("/info", [&engine](const httplib::Request&, httplib::Response& res) {
    jsonResponse(res, {{"model", engine.config().model_name}, {"backend", "mlx"},
                       {"sampleRate", engine.config().sample_rate},
                       {"platform", "macos"}, {"architecture", "arm64"}});
  });
  server.Get("/openapi.json", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(kOpenApiJson, "application/json");
  });
  server.Get("/docs", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(kApiDocsHtml, "text/html; charset=utf-8");
  });
  server.Post("/generate", [&engine](const httplib::Request& req,
                                      httplib::Response& res) {
    try {
      if (!req.has_header("Content-Type") ||
          req.get_header_value("Content-Type").find("application/json") != 0) {
        jsonResponse(res, {{"error", "Content-Type must be application/json"}}, 415);
        return;
      }
      const auto input = nlohmann::json::parse(req.body);
      if (!input.is_object() || !input.contains("prompt") ||
          !input["prompt"].is_string()) {
        throw std::invalid_argument("prompt must be a string");
      }
      for (const auto& [key, value] : input.items()) {
        if (key != "prompt" && key != "duration") {
          throw std::invalid_argument("unknown property: " + key);
        }
      }
      float duration = 10.0f;
      if (input.contains("duration")) {
        if (!input["duration"].is_number()) {
          throw std::invalid_argument("duration must be a number");
        }
        duration = input["duration"].get<float>();
      }
      // MrtEngine serializes the complete stateful generation lifecycle.
      auto wav = encodeWav(engine.generate(input["prompt"].get<std::string>(), duration));
      res.set_content(reinterpret_cast<const char*>(wav.data()), wav.size(), "audio/wav");
    } catch (const nlohmann::json::exception& e) {
      jsonResponse(res, {{"error", std::string("invalid JSON: ") + e.what()}}, 400);
    } catch (const std::invalid_argument& e) {
      jsonResponse(res, {{"error", e.what()}}, 400);
    } catch (const std::exception& e) {
      std::cerr << "Generation error: " << e.what() << '\n';
      jsonResponse(res, {{"error", "audio generation failed"}}, 500);
    }
  });
  server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (res.status == 404) jsonResponse(res, {{"error", "not found"}}, 404);
  });

  if (!server.bind_to_port(host, port)) {
    std::cerr << "Failed to bind http://" << host << ':' << port << '\n';
    return 1;
  }
  std::cout << "MRT Local Server\n\n"
            << "Model: " << engine.config().model_name
            << "\nBackend: MLX\nSample rate: 48000\n\n"
            << "Listening:\nhttp://" << host << ':' << port << "\n";
  return server.listen_after_bind() ? 0 : 1;
}

}  // namespace mrt_local
