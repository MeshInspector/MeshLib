#pragma once

#include "exports.h"

#include <string_view>

namespace MR::Locale
{

/// \brief Checks that the translation can be passed to the same fmt::format or printf-like call as the original string.
/// The translation must not use the arguments that the original does not use, or format them as other types;
/// otherwise fmt::format throws and printf has undefined behavior.
/// The original without fmt fields and without printf conversions is not a format string, and any translation suits it.
[[nodiscard]] MRVIEWER_API bool isFormatSafeTranslation( std::string_view original, std::string_view translation );

} // namespace MR::Locale
