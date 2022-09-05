#pragma once

#include "MRViewportId.h"
#include <map>

namespace MR
{

/// \defgroup ViewportGroup Viewport group
/// \ingroup DataModelGroup
/// \{

/// storage of some viewport-dependent property,
/// which has some default value for all viewports and special values for some specific viewports
template<typename T>
class ViewportProperty
{
public:
    /// sets default property value
    void set( T def ) { def_ = std::move( def ); }
    /// gets default property value
    const T & get() const { return def_; }
          T & get()       { return def_; }

    /// sets specific property value for given viewport
    void set( ViewportId id, T v ) { assert( id ); map_[id] = std::move( v ); }
    /// gets property value for given viewport: specific if available otherwise default one;
    /// \param isDef receives true if this viewport does not have specific value and default one is returned
    const T & get( ViewportId id, bool * isDef = nullptr ) const
    { 
        assert( id );
        auto it = map_.find( id );
        if ( it != map_.end() )
        {
            if ( isDef )
                *isDef = false;
            return it->second;
        }
        if ( isDef )
            *isDef = true;
        return def_;
    }
    T & get( ViewportId id, bool * isDef = nullptr )
    { 
        return const_cast<T&>( const_cast<const ViewportProperty<T>*>( this )->get( id, isDef ) );
    }
    /// forgets specific property value for given viewport
    void reset( ViewportId id ) { assert( id ); map_.erase( id ); }
    /// forgets specific property value for all viewports
    void reset() { map_.clear(); }

private:
    T def_{};
    std::map<ViewportId, T> map_;
};

/// \}

} // namespace MR
