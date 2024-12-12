#pragma once

#if __EMSCRIPTEN__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnontrivial-memaccess"
#endif

#include <imgui.h>
#include <imgui_internal.h>

#if __EMSCRIPTEN__
#pragma clang diagnostic pop
#endif