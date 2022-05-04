#pragma once

#include "MRAffineXf.h"
#include <optional>

namespace MR
{

/// optional T-object container, which stores a transformation as well for which the object is valid
/// \ingroup DataModelGroup
template<class T>
class XfBasedCache
{
public:
    /// returns stored object only if requested transformation is the same as stored one
    const std::optional<T> & get( const AffineXf3f & xf ) const
    {
        if ( cache_ && xf == xf_ )
            return cache_;
        static const std::optional<T> empty;
        return empty;
    }
    /// sets new transformation and the object
    void set( const AffineXf3f & xf, T t )
    {
        xf_ = xf;
        cache_ = std::move( t );
    }
    /// clears stored object
    void reset()
    {
        cache_.reset();
    }

private:
    AffineXf3f xf_;
    std::optional<T> cache_;
};

} // namespace MR
