#pragma once

#include <algorithm>
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
    IOFilter( const std::string& _name, const std::string& _ext ) :
        name{_name}, extension{_ext}{}
    std::string name;
    std::string extension; // "*.ext"
};

using IOFilters = std::vector<IOFilter>;

inline IOFilters operator | ( const IOFilters& a, const IOFilters& b )
{
    IOFilters copy = a;
    for ( const auto& bElem : b )
    {
        if ( std::find_if( a.begin(), a.end(), [&] ( const IOFilter& aF )
        {
            return aF.extension == bElem.extension;
        } ) == a.end() )
            copy.push_back( bElem );
    }
    return copy;
}

/// \}

}
