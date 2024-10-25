#include "MRIOFilters.h"
#include <algorithm>
#include <cassert>

namespace MR
{

bool IOFilter::isSupportedExtension( const std::string& ext ) const
{
    const auto pos = extensions.find( ext );
    if ( pos == std::string::npos )
        return false;
    // check full match of the extension
    const auto epos = pos + ext.size();
    assert( epos <= extensions.size() );
    return epos == extensions.size() || extensions[epos] == ';';
}

IOFilters operator | ( const IOFilters& a, const IOFilters& b )
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

std::optional<IOFilter> findFilter( const IOFilters& filters, const std::string& extension )
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

} //namespace MR
