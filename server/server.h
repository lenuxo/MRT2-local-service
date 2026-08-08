#pragma once

#include "core/engine.h"

#include <string>

namespace mrt_local {

int runServer(MrtEngine& engine, const std::string& host, int port);

}  // namespace mrt_local
