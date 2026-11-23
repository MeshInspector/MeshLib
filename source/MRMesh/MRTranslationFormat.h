#pragma once
#include "MRMeshFwd.h"
#include <string_view>

namespace MR::Locale
{

/// checks that the translation can be passed to the same fmt::format or printf-like call as the original string:
/// it must not use the arguments that the original does not use, or format them as other types;
/// a translation failing this check throws from fmt::format or has undefined behavior in printf
[[nodiscard]] MRMESH_API bool isFormatSafeTranslation( std::string_view original, std::string_view translation );

} // namespace MR::Locale
