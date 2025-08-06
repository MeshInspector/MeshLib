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
 * \param outLeftRightAddition if provided return amount of insertions to the left and to the right
 * \ingroup BasicGroup
 */
[[nodiscard]] MRMESH_API int calcDamerauLevenshteinDistance( const std::string& stringA, const std::string& stringB,
    bool caseSensitive = true, int* outLeftRightAddition = nullptr );

/**
 * Splits given string by delimiter.
 * \return vector of split strings
 * \ingroup BasicGroup
 */
[[nodiscard]] MRMESH_API std::vector<std::string> split( const std::string& string, const std::string& delimiter );

// This version of `split()` passes the segments to a callback, to avoid heap allocations.
// If the callback returns true, stops immediately and also returns true.
template <typename F>
bool split( std::string_view str, std::string_view sep, F&& func )
{
    std::size_t index = 0;

    while ( true )
    {
        std::size_t newIndex = str.find( sep, index );
        if ( func( str.substr( index, newIndex - index ) ) )
            return true;
        if ( newIndex == std::string_view::npos )
            break;
        index = newIndex + sep.size();
    }
    return false;
}

/// Returns \param target with all \param from replaced with \param to, zero or more times.
[[nodiscard]] MRMESH_API std::string replace( std::string target, std::string_view from, std::string_view to );

/// Replaces \param from with \param to in \param target (in-place), zero or more times.
MRMESH_API void replaceInplace( std::string& target, std::string_view from, std::string_view to );

/// Removes all whitespace character (detected by std::isspace) at the beginning and the end of string view
[[nodiscard]] MRMESH_API std::string_view trim( std::string_view str );

/// Removes all whitespace character (detected by std::isspace) at the beginning of string view
[[nodiscard]] MRMESH_API std::string_view trimLeft( std::string_view str );

/// Removes all whitespace character (detected by std::isspace) at the end of string view
[[nodiscard]] MRMESH_API std::string_view trimRight( std::string_view str );

} //namespace MR
