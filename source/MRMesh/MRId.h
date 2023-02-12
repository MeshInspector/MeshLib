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
    Id() noexcept : id_( -1 ) { }
    explicit Id( NoInit ) noexcept { }
    explicit constexpr Id( int i ) noexcept : id_( i ) { }
    explicit constexpr Id( size_t i ) noexcept : id_( int( i ) ) { }
    template <typename U> Id( Id<U> ) = delete;

    operator int() const { return id_; }
    bool valid() const { return id_ >= 0; }
    explicit operator bool() const { return id_ >= 0; }
    constexpr int & get() noexcept { return id_; }

    bool operator == (Id b) const { return id_ == b.id_; }
    bool operator != (Id b) const { return id_ != b.id_; }
    bool operator <  (Id b) const { return id_ <  b.id_; }

    template <typename U> 
    bool operator == (Id<U> b) const = delete;
    template <typename U> 
    bool operator != (Id<U> b) const = delete;
    template <typename U> 
    bool operator < (Id<U> b) const = delete;

    Id & operator --() { --id_; return * this; }
    Id & operator ++() { ++id_; return * this; }

    Id operator --( int ) { auto res = *this; --id_; return res; }
    Id operator ++( int ) { auto res = *this; ++id_; return res; }

    Id & operator -=( int a ) { id_ -= a; return * this; }
    Id & operator +=( int a ) { id_ += a; return * this; }

private:
    int id_;
};

template <> 
class Id<EdgeTag>
{
public:
    Id() noexcept : id_( -1 ) { }
    explicit Id( NoInit ) noexcept { }
    Id( UndirectedEdgeId u ) noexcept : id_( (int)u << 1 ) { assert( u.valid() ); }
    explicit constexpr Id( int i ) noexcept : id_( i ) { }
    explicit constexpr Id( size_t i ) noexcept : id_( int( i ) ) { }
    operator int() const { return id_; }
    bool valid() const { return id_ >= 0; }
    explicit operator bool() const { return id_ >= 0; }
    constexpr int & get() noexcept { return id_; }

    // returns identifier of the edge with same ends but opposite orientation
    Id sym() const { assert( valid() ); return Id(id_ ^ 1); }
    // among each pair of sym-edges: one is always even and the other is odd
    bool even() const { assert( valid() ); return (id_ & 1) == 0; }
    bool odd() const { assert( valid() ); return (id_ & 1) == 1; }
    // returns unique identifier of the edge ignoring its direction
    UndirectedEdgeId undirected() const { assert( valid() ); return UndirectedEdgeId( id_ >> 1 ); }
    operator UndirectedEdgeId() const { return undirected(); }

    bool operator == (Id b) const { return id_ == b.id_; }
    bool operator != (Id b) const { return id_ != b.id_; }
    bool operator <  (Id b) const { return id_ <  b.id_; }

    template <typename U> 
    bool operator == (Id<U> b) const = delete;
    template <typename U> 
    bool operator != (Id<U> b) const = delete;
    template <typename U> 
    bool operator < (Id<U> b) const = delete;

    Id & operator --() { --id_; return * this; }
    Id & operator ++() { ++id_; return * this; }

    Id operator --( int ) { auto res = *this; --id_; return res; }
    Id operator ++( int ) { auto res = *this; ++id_; return res; }

    Id & operator -=( int a ) { id_ -= a; return * this; }
    Id & operator +=( int a ) { id_ += a; return * this; }

private:
    int id_;
};

template <typename T> 
inline Id<T> operator + ( Id<T> id, int a ) { return Id<T>{ (int)id + a }; }
template <typename T> 
inline Id<T> operator + ( Id<T> id, unsigned int a ) { return Id<T>{ (int)id + (int)a }; }
template <typename T> 
inline Id<T> operator - ( Id<T> id, int a ) { return Id<T>{ (int)id - a }; }

inline constexpr FaceId operator "" _f( unsigned long long i ) noexcept { return FaceId{ (int)i }; }
inline constexpr VertId operator "" _v( unsigned long long i ) noexcept { return VertId{ (int)i }; }
inline constexpr EdgeId operator "" _e( unsigned long long i ) noexcept { return EdgeId{ (int)i }; }
inline constexpr UndirectedEdgeId operator "" _ue( unsigned long long i ) noexcept { return UndirectedEdgeId{ (int)i }; }

} //namespace MR
