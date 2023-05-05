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
    ViewportProperty() = default;
    ViewportProperty( const T& def ) :def_{ def } {}
    /// sets default property value
    void set( T def ) { def_ = std::move( def ); }
    /// gets default property value
    const T & get() const { return def_; }
          T & get()       { return def_; }

    /// returns direct access to value associated with given viewport (or default value if !id)
    T & operator[]( ViewportId id )
    {
        return id ? map_[id] : def_;
    }
    /// sets specific property value for given viewport (or default value if !id)
    void set( T v, ViewportId id )
    { 
        (*this)[id] = std::move( v );
    }
    /// gets property value for given viewport: specific if available otherwise default one;
    /// \param isDef receives true if this viewport does not have specific value and default one is returned
    const T & get( ViewportId id, bool * isDef = nullptr ) const
    { 
        if ( id )
        {
            auto it = map_.find( id );
            if ( it != map_.end() )
            {
                if ( isDef )
                    *isDef = false;
                return it->second;
            }
        }
        if ( isDef )
            *isDef = true;
        return def_;
    }
    /// forgets specific property value for given viewport (or all viewports if !id);
    /// returns true if any specific value was removed
    bool reset( ViewportId id )
    {
        if ( id )
            return map_.erase( id ) > 0;
        if ( map_.empty() )
            return false;
        map_.clear();
        return true;
    }
    /// forgets specific property value for all viewports;
    /// returns true if any specific value was removed
    bool reset()
    {
        if ( map_.empty() )
            return false;
        map_.clear();
        return true;
    }

private:
    T def_{};
    std::map<ViewportId, T> map_;
};

/// \}

} // namespace MR
