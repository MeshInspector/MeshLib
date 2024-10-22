#pragma once

#include <algorithm>
#include <compare>
#include <optional>
#include <string>
#include <vector>

namespace MR
{

/// \defgroup IOFiltersGroup IO Filters
/// \ingroup IOGroup
/// \{

struct IOFilter
{
    IOFilter() = default;
    IOFilter( std::string _name, std::string _ext )
        : name{ std::move( _name ) }
        , extensions{ std::move( _ext ) }
    {}

    std::string name;
    std::string extensions; // "*.ext" or "*.ext1;*.ext2;*.ext3"

    std::partial_ordering operator <=>( const IOFilter& ) const = default;

    [[nodiscard]] inline bool isSupportedExtension( const std::string& ext ) const
    {
        const auto pos = extensions.find( ext );
        if ( pos == std::string::npos )
            return false;
        // check full match of the extension
        const auto epos = pos + ext.size();
        assert( epos <= extensions.size() );
        return epos == extensions.size() || extensions[epos] == ';';
    }
};

using IOFilters = std::vector<IOFilter>;

inline IOFilters operator | ( const IOFilters& a, const IOFilters& b )
{
    IOFilters copy = a;
    for ( const auto& bElem : b )
    {
        if ( std::find_if( a.begin(), a.end(), [&] ( const IOFilter& aF )
        {
            return aF.extensions == bElem.extensions;
        } ) == a.end() )
            copy.push_back( bElem );
    }
    return copy;
}

/// find a corresponding filter for a given extension
inline std::optional<IOFilter> findFilter( const IOFilters& filters, const std::string& extension )
{
    const auto it = std::find_if( filters.begin(), filters.end(), [&extension] ( auto&& filter )
    {
        return filter.isSupportedExtension( extension );
    } );
    if ( it != filters.end() )
        return *it;
    else
        return std::nullopt;
}

/// \}

}
