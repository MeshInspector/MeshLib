#pragma once

#include "exports.h"

#include <string_view>

namespace MR::Locale
{

/// \brief Checks that the translation can be passed to the same fmt::format or printf-like call as the original string.
/// The translation of a format string of fmt::format must have the same fields with the same format specs in any order,
/// and of a format string of printf - the same conversion specifications in the same order.
/// The original without fmt fields and without printf conversions is not a format string, and any translation suits it.
[[nodiscard]] MRVIEWER_API bool isFormatSafeTranslation( std::string_view original, std::string_view translation );

} // namespace MR::Locale
