#pragma once
#include <future>

namespace MR
{
// return std::launch::deferred for emscripten build, std::launch::async otherwise
inline std::launch getAsyncLaunchType()
{
#ifdef __EMSCRIPTEN__
    return std::launch::deferred;
#else
    return std::launch::async;
#endif
}

}