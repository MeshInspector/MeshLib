#pragma once
#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"
#include <string>
#include <typeinfo>

namespace MR
{

/**
 * Finds the substring in the string.
 * \note only ASCII letters are compared case-insensitively, for other alphabets case-fold both strings first
 * \return position, npos if not found
 * \ingroup BasicGroup
 */
[[nodiscard]] MRMESH_API size_t findSubstringCaseInsensitive( const std::string& string, const std::string& substring );

/**
 * Calculates Damerau-Levenshtein distance between to strings
 * \param outLeftRightAddition if provided return amount of insertions to the left and to the right
 * \note if not caseSensitive, only ASCII letters are compared case-insensitively
 * \ingroup BasicGroup
 */
[[nodiscard]] MRMESH_API int calcDamerauLevenshteinDistance( const std::string& stringA, const std::string& stringB,
    bool caseSensitive = true, int* outLeftRightAddition = nullptr );

/**
 * Returns the simple Unicode case folding of a codepoint (per CaseFolding.txt, statuses C + S),
 * for case-insensitive comparison. Simple folding is 1:1, so string length is preserved.
 * Codepoints without a folding (and all case-less scripts) are returned unchanged.
 * \note self-contained (no locale/ICU dependency), so behaves identically on every platform.
 * \ingroup BasicGroup
 */
[[nodiscard]] MR_BIND_IGNORE MRMESH_API char32_t caseFold( char32_t ch );

/**
 * Calculates Damerau-Levenshtein distance between two UTF-32 strings, in codepoints
 * \param outLeftRightAddition if provided return amount of insertions to the left and to the right
 * \note if not caseSensitive, only ASCII letters are compared case-insensitively, for other alphabets case-fold both strings first
 * \ingroup BasicGroup
 */
[[nodiscard]] MR_BIND_IGNORE MRMESH_API int calcDamerauLevenshteinDistance( const std::u32string& stringA, const std::u32string& stringB,
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


/// Returns true if `str` has at least one `{...}` formatting placeholder.
[[nodiscard]] MRMESH_API bool hasFormatPlaceholders( std::string_view str );

} //namespace MR
