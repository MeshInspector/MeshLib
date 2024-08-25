#pragma once

#include "MRMeshFwd.h"
#include <cassert>
#include <cstddef>

namespace MR
{

// stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
template <typename T>
class Id
{
public:
    constexpr Id() noexcept : id_( -1 ) { }
    explicit Id( NoInit ) noexcept { }
    explicit constexpr Id( int i ) noexcept : id_( i ) { }
    explicit constexpr Id( unsigned int i ) noexcept : id_( i ) { }
    explicit constexpr Id( size_t i ) noexcept : id_( int( i ) ) { }
    template <typename U> Id( Id<U> ) = delete;

    constexpr operator int() const { return id_; }
    constexpr bool valid() const { return id_ >= 0; }
    explicit constexpr operator bool() const { return id_ >= 0; }
    constexpr int & get() noexcept { return id_; }

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

    constexpr Id & operator -=( int a ) { id_ -= a; return * this; }
    constexpr Id & operator +=( int a ) { id_ += a; return * this; }

private:
    int id_;
};

template <>
class Id<MR::EdgeTag> // Need `MR::` here to simplify binding generation. See libclang bug: https://github.com/llvm/llvm-project/issues/92371
{
public:
    constexpr Id() noexcept : id_( -1 ) { }
    explicit Id( NoInit ) noexcept { }
    constexpr Id( UndirectedEdgeId u ) noexcept : id_( (int)u << 1 ) { assert( u.valid() ); }
    explicit constexpr Id( int i ) noexcept : id_( i ) { }
    explicit constexpr Id( unsigned int i ) noexcept : id_( i ) { }
    explicit constexpr Id( size_t i ) noexcept : id_( int( i ) ) { }
    constexpr operator int() const { return id_; }
    constexpr bool valid() const { return id_ >= 0; }
    explicit constexpr operator bool() const { return id_ >= 0; }
    constexpr int & get() noexcept { return id_; }

    // returns identifier of the edge with same ends but opposite orientation
    constexpr Id sym() const { assert( valid() ); return Id(id_ ^ 1); }
    // among each pair of sym-edges: one is always even and the other is odd
    constexpr bool even() const { assert( valid() ); return (id_ & 1) == 0; }
    constexpr bool odd() const { assert( valid() ); return (id_ & 1) == 1; }
    // returns unique identifier of the edge ignoring its direction
    constexpr UndirectedEdgeId undirected() const { assert( valid() ); return UndirectedEdgeId( id_ >> 1 ); }
    constexpr operator UndirectedEdgeId() const { return undirected(); }

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

    constexpr Id & operator -=( int a ) { id_ -= a; return * this; }
    constexpr Id & operator +=( int a ) { id_ += a; return * this; }

private:
    int id_;
};

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

template <typename T>
inline constexpr Id<T> operator + ( Id<T> id, int a )          { return Id<T>{ id.get() + a }; }
template <typename T>
inline constexpr Id<T> operator + ( Id<T> id, unsigned int a ) { return Id<T>{ id.get() + a }; }
template <typename T>
inline constexpr Id<T> operator + ( Id<T> id, size_t a )       { return Id<T>{ id.get() + a }; }

template <typename T>
inline constexpr Id<T> operator - ( Id<T> id, int a )          { return Id<T>{ id.get() - a }; }
template <typename T>
inline constexpr Id<T> operator - ( Id<T> id, unsigned int a ) { return Id<T>{ id.get() - a }; }
template <typename T>
inline constexpr Id<T> operator - ( Id<T> id, size_t a )       { return Id<T>{ id.get() - a }; }

inline constexpr FaceId operator "" _f( unsigned long long i ) noexcept { return FaceId{ (int)i }; }
inline constexpr VertId operator "" _v( unsigned long long i ) noexcept { return VertId{ (int)i }; }
inline constexpr EdgeId operator "" _e( unsigned long long i ) noexcept { return EdgeId{ (int)i }; }
inline constexpr UndirectedEdgeId operator "" _ue( unsigned long long i ) noexcept { return UndirectedEdgeId{ (int)i }; }
inline constexpr VoxelId operator "" _vox( unsigned long long i ) noexcept { return VoxelId{ size_t( i ) }; }

} //namespace MR
