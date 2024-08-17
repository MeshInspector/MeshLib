#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"

namespace MR
{

template <>
class Id<VoxelTag>
{
public:
    constexpr Id() noexcept : id_( ~size_t( 0 ) ) { }
    explicit Id( NoInit ) noexcept { }
    explicit constexpr Id( size_t i ) noexcept : id_( i ) { }
    explicit constexpr Id( int ) noexcept = delete;
    constexpr operator size_t() const { return id_; }
    constexpr bool valid() const { return id_ != ~size_t( 0 ); }
    explicit constexpr operator bool() const { return id_ != ~size_t( 0 ); }
    constexpr size_t& get() noexcept { return id_; }

    constexpr bool operator == (Id b) const { return id_ == b.id_; }
    constexpr bool operator != (Id b) const { return id_ != b.id_; }
    constexpr bool operator <  (Id b) const { return id_ <  b.id_; }

    template <typename U>
    bool operator == (Id<U> b) const = delete;
    template <typename U>
    bool operator != (Id<U> b) const = delete;
    template <typename U>
    bool operator < (Id<U> b) const = delete;

    constexpr Id & operator --() { --id_; return * this; }
    constexpr Id & operator ++() { ++id_; return * this; }

    constexpr Id operator --( int ) { auto res = *this; --id_; return res; }
    constexpr Id operator ++( int ) { auto res = *this; ++id_; return res; }

    constexpr Id & operator -=( size_t a ) { id_ -= a; return * this; }
    constexpr Id & operator +=( size_t a ) { id_ += a; return * this; }

private:
    size_t id_;
};

inline constexpr Id<VoxelTag> operator + ( Id<VoxelTag> id, size_t a ) { return Id<VoxelTag>{ ( size_t )id + a }; }
inline constexpr Id<VoxelTag> operator - ( Id<VoxelTag> id, size_t a ) { return Id<VoxelTag>{ ( size_t )id - a }; }

inline constexpr VoxelId operator "" _vox( unsigned long long i ) noexcept { return VoxelId{ size_t( i ) }; }

} // namespace MR
