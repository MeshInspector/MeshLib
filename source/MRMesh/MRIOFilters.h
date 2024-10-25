#pragma once

#include "MRMeshFwd.h"
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

    [[nodiscard]] MRMESH_API bool isSupportedExtension( const std::string& ext ) const;
};

using IOFilters = std::vector<IOFilter>;

/// returns union of input filters
[[nodiscard]] MRMESH_API IOFilters operator | ( const IOFilters& a, const IOFilters& b );

/// find a corresponding filter for a given extension
[[nodiscard]] MRMESH_API std::optional<IOFilter> findFilter( const IOFilters& filters, const std::string& extension );

/// \}

}
