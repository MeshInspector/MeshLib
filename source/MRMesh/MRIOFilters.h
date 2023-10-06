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
        name{_name}, extensions{_ext}{}
    std::string name;
    std::string extensions; // "*.ext" or "*.ext1;*.ext2;*.ext3"
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

/// \}

}
