

#include "MRChrono.h"


// Those two functions are a chain of fallbacks in case the system doesn't provide `localtime_r` or `localtime_s`.
// They are intentionally in the global namespace to make sure there's no shadowing going on.
// `auto...` reduces the function priority to make sure the proper function is called if it exists.

[[maybe_unused]] static std::tm *localtime_s(const std::time_t *timer, std::tm* buf, auto...)
{
    (void)buf;
    return localtime(timer);
}

// Using `(...)` just in case `localtime_r` is a macro. It isn't on my system, but libfmt protects against that, so maybe it is a macro on some systems.
[[maybe_unused]] static std::tm *(localtime_r)(const std::time_t *timer, std::tm* buf, auto...)
{
    // Do we need `__STDC_WANT_LIB_EXT1__` for this? I think on MSVC it works fine even without that macro.
    // If we decide to add it, it probably needs to be duplicated to the PCH as well.
    return localtime_s(timer, buf);
}

namespace MR
{

std::optional<std::tm> Localtime(std::time_t time)
{
    std::tm ret{};
    if (auto ptr = localtime_r(&time, &ret))
        return *ptr;
    else
        return {};
}

}
