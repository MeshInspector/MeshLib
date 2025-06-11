#pragma once

#include "MRPch/MRSuppressWarning.h"

MR_SUPPRESS_WARNING_PUSH
#if __EMSCRIPTEN__
#pragma clang diagnostic ignored "-Wnontrivial-memaccess"
#endif
#if __clang_major__ == 20
#pragma clang diagnostic ignored "-Wnontrivial-memcall"
#endif

#include <imgui.h>
#include <imgui_internal.h>

MR_SUPPRESS_WARNING_POP
