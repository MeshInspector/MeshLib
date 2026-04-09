#pragma once

#include <string>
#include <vector>

namespace MR::Locale::detail
{

/// \brief Returns a list of preferred locales on macOS.
/// \note The locales are in the BCP 47 format.
std::vector<std::string> getMacosLocales();

} // namespace MR::Locale::detail
