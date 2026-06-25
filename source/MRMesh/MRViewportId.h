#pragma once

#include "MRMeshFwd.h"
#include <cassert>

namespace MR
{

/// \defgroup ViewportGroup Viewport group
/// \ingroup DataModelGroup
/// \{

/// stores unique identifier of a viewport, which is power of two;
/// id=0 has a special meaning of default viewport in some contexts
class ViewportId
{
public:
    ViewportId() noexcept = default;
    explicit constexpr ViewportId( unsigned i ) noexcept : id_( i ) { }

    constexpr unsigned value() const { return id_; }
    bool valid() const { return id_ > 0; }
    explicit operator bool() const { return id_ > 0; }

    bool operator == ( ViewportId b ) const { return id_ == b.id_; }
    bool operator != ( ViewportId b ) const { return id_ != b.id_; }
    bool operator <  ( ViewportId b ) const { return id_ <  b.id_; }

    ViewportId next() const { return ViewportId{ id_ << 1 }; }
    ViewportId prev() const { return ViewportId{ id_ >> 1 }; }

// Allocating IDs on the heap in C is insufferable, so instead we bind them as single-member structs (via `--expose-as-struct`).
// But since our parser only tracks public members, we have to make it public for C.
#if !MR_PARSING_FOR_C_BINDINGS && !MR_COMPILING_C_BINDINGS
private:
#endif
    unsigned id_ = 0;
};

/// stores mask of viewport unique identifiers
class ViewportMask
{
public:
    ViewportMask() noexcept = default;
    explicit constexpr ViewportMask( unsigned i ) noexcept : mask_( i ) { }
    constexpr ViewportMask( ViewportId i ) noexcept : mask_( i.value() ) { }

    /// mask meaning all or any viewports
    static ViewportMask all() { return ViewportMask{ ~0u }; }
    static ViewportMask any() { return ViewportMask{ ~0u }; }

    constexpr unsigned value() const { return mask_; }
    constexpr bool empty() const { return mask_ == 0; }
    bool contains( ViewportId id ) const { assert( id.valid() ); return ( mask_ & id.value() ) != 0; }

    bool operator == ( ViewportMask b ) const { return mask_ == b.mask_; }
    bool operator != ( ViewportMask b ) const { return mask_ != b.mask_; }
    bool operator <  ( ViewportMask b ) const { return mask_ <  b.mask_; }

    ViewportMask operator~() const { ViewportMask res; res.mask_ = ~mask_; return res; }

    ViewportMask & operator &= ( ViewportMask b ) { mask_ &= b.mask_; return * this; }
    ViewportMask & operator |= ( ViewportMask b ) { mask_ |= b.mask_; return * this;  }
    ViewportMask & operator ^= ( ViewportMask b ) { mask_ ^= b.mask_; return * this;  }

    void set( ViewportId id, bool on = true ) { on ? ( mask_ |= id.value() ) : ( mask_ &= ~id.value() ); }

// Allocating IDs/masks on the heap in C is insufferable, so instead we bind them as single-member structs (via `--expose-as-struct`).
// But since our parser only tracks public members, we have to make it public for C.
#if !MR_PARSING_FOR_C_BINDINGS && !MR_COMPILING_C_BINDINGS
private:
#endif
    unsigned mask_ = 0;
};

inline ViewportMask operator & ( ViewportMask a, ViewportMask b ) { a &= b; return a; }
inline ViewportMask operator | ( ViewportMask a, ViewportMask b ) { a |= b; return a; }
inline ViewportMask operator ^ ( ViewportMask a, ViewportMask b ) { a ^= b; return a; }

/// iterates over all ViewportIds in given ViewportMask
class ViewportIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = ViewportId;

    /// constructs end iterator
    ViewportIterator() = default;
    /// constructs begin iterator
    ViewportIterator( ViewportMask mask )
        : mask_( mask )
    {
        findFirst_();
    }
    ViewportIterator & operator++( )
    {
        findNext_();
        return * this;
    }
    ViewportMask mask() const { return mask_; }
    ViewportId operator *() const { return id_; }

private:
    void findFirst_()
    {
        id_ = ViewportId{1};
        for( ; id_.valid() && !mask_.contains( id_ ); id_ = id_.next() );
    }
    void findNext_()
    {
        assert( id_.valid() );
        for( id_ = id_.next(); id_.valid() && !mask_.contains( id_ ); id_ = id_.next() );
    }

    ViewportId id_{0};
    ViewportMask mask_{0};
};

inline bool operator ==( const ViewportIterator & a, const ViewportIterator & b )
    { return *a == *b; }

inline auto begin( ViewportMask mask )
    { return ViewportIterator( mask ); }
inline auto end( ViewportMask )
    { return ViewportIterator(); }

/// \}

} // namespace MR
