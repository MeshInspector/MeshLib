#pragma once
#include "MRMeshFwd.h"
#include <string>
#include <typeinfo>

namespace MR
{

/** 
 * Finds the substring in the string.
 * \return position, npos if not found
 * \ingroup BasicGroup
 */
[[nodiscard]] MRMESH_API size_t findSubstringCaseInsensitive( const std::string& string, const std::string& substring );

/**
 * Calculates Damerau-Levenshtein distance between to strings
 * \ingroup BasicGroup
 */
[[nodiscard]] MRMESH_API int calcDamerauLevenshteinDistance( const std::string& stringA, const std::string& stringB );

/**
 * Splits given string by delimiter.
 * \return vector of split strings
 * \ingroup BasicGroup
 */
[[nodiscard]] MRMESH_API std::vector<std::string> split( const std::string& string, const std::string& delimiter );

}
