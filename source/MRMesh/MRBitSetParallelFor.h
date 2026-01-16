#pragma once

#include "MRBitSet.h"
#include "MRFunctional.h"
#include "MRParallel.h"
#include "MRProgressCallback.h"

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/// range of indices [beg, end)
template <typename Id>
struct IdRange
{
    Id beg, end;
    auto size() const { return end - beg; }
};

namespace BitSetParallel
{

template <typename IndexType>
inline auto blockRange( const IdRange<IndexType> & bitRange )
{
    const size_t beginBlock = bitRange.beg / BitSet::bits_per_block;
    const size_t endBlock = ( size_t( bitRange.end ) + BitSet::bits_per_block - 1 ) / BitSet::bits_per_block;
    return tbb::blocked_range<size_t>( beginBlock, endBlock );
}

template <typename BS>
inline auto blockRange( const BS & bs )
{
    const size_t endBlock = ( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    return tbb::blocked_range<size_t>( 0, endBlock );
}

template <typename BS>
inline auto bitRange( const BS & bs )
{
    return IdRange<typename BS::IndexType>{ bs.beginId(), bs.endId() };
}

template <typename IndexType>
auto bitSubRange( const IdRange<IndexType> & bitRange, const tbb::blocked_range<size_t> & range, const tbb::blocked_range<size_t> & subRange )
{
    return IdRange<IndexType>
    {
        .beg = subRange.begin() > range.begin() ? IndexType( subRange.begin() * BitSet::bits_per_block ) : bitRange.beg,
        .end = subRange.end() < range.end()     ? IndexType( subRange.end()   * BitSet::bits_per_block ) : bitRange.end
    };
}

using Range = tbb::blocked_range<size_t>;

MRMESH_API void forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range & )> f );

MRMESH_API void forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range &, void* )> f,
    FunctionRef<void* ()> ctx );

MRMESH_API bool forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range &, void* )> f,
    FunctionRef<void* ()> ctx, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 );

template <typename BS, typename F, typename ...Cb>
auto ForAllRanged( const BS & bs, F && f, Cb && ... cb )
{
    if constexpr ( sizeof...( cb ) == 0 )
    {
        return forAllRanged( { (size_t)bs.beginId(), (size_t)bs.endId() }, [&] ( size_t i, const Range& range )
        {
            using Id = typename BS::IndexType;
            std::forward<F>( f )( Id{ i }, IdRange<Id>{ Id{ range.begin() }, Id{ range.end() } } );
        } );
    }
    else
    {
        return forAllRanged( { (size_t)bs.beginId(), (size_t)bs.endId() }, [&] ( size_t i, const Range& range, void* )
        {
            using Id = typename BS::IndexType;
            std::forward<F>( f )( Id{ i }, IdRange<Id>{ Id{ range.begin() }, Id{ range.end() } } );
        }, [&]
        {
            return nullptr;
        }, std::forward<Cb>( cb )... );
    }
}

template <typename BS, typename L, typename F, typename ...Cb>
auto ForAllRanged( const BS & bs, tbb::enumerable_thread_specific<L>& e, F && f, Cb && ... cb )
{
    return forAllRanged( { (size_t)bs.beginId(), (size_t)bs.endId() }, [&] ( size_t i, const Range& range, void* ctx )
    {
        using Id = typename BS::IndexType;
        std::forward<F>( f )( Id{ i }, IdRange<Id>{ Id{ range.begin() }, Id{ range.end() } }, *(L*)ctx );
    }, [&]
    {
        return (void*)&e.local();
    }, std::forward<Cb>( cb )... );
}

} // namespace BitSetParallel

/// executes given function f( bit, subBitRange ) for each bit in bitRange in parallel threads,
/// where (subBitRange) are the bits that will be processed by the same thread;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename ...F>
inline auto BitSetParallelForAllRanged( const BS & bs, F &&... f )
{
    return BitSetParallel::ForAllRanged( bs, std::forward<F>( f )... );
}

/// executes given function f( bit, subBitRange, tls ) for each bit in IdRange or BitSet (bs) in parallel threads,
/// where subBitRange are the bits that will be processed by the same thread,
///       tls=e.local() (evaluated once for each subBitRange);
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename L, typename ...F>
inline auto BitSetParallelForAllRanged( const BS & bs, tbb::enumerable_thread_specific<L> & e, F &&... f )
{
    return BitSetParallel::ForAllRanged( bs, e, std::forward<F>( f )... );
}

/// executes given function f for each index in IdRange or BitSet (bs) in parallel threads;
/// it is guaranteed that every individual block in BitSet is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename F, typename ...Cb>
inline auto BitSetParallelForAll( const BS & bs, F && f, Cb&&... cb )
{
    return BitSetParallel::ForAllRanged( bs, [&] ( auto bit, auto && ) { std::forward<F>( f )( bit ); }, std::forward<Cb>( cb )... );
}

/// executes given function f for each index in IdRange or BitSet (bs) in parallel threads
/// passing e.local() (evaluated once for each sub-range) as the second argument to f;
/// it is guaranteed that every individual block in BitSet is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename L, typename F, typename ...Cb>
inline auto BitSetParallelForAll( const BS & bs, tbb::enumerable_thread_specific<L> & e, F && f, Cb&&... cb )
{
    return BitSetParallel::ForAllRanged( bs, e, [&] ( auto bit, auto &&, auto & tls ) { std::forward<F>( f )( bit, tls ); }, std::forward<Cb>( cb )... );
}

/// executes given function f for every _set_ bit in IdRange or BitSet (bs) in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename F, typename ...Cb>
inline auto BitSetParallelFor( const BS& bs, F && f, Cb&&... cb )
{
    return BitSetParallelForAll( bs, [&] ( auto bit ) { if ( bs.test( bit ) ) std::forward<F>( f )( bit ); }, std::forward<Cb>( cb )... );
}

/// executes given function f for every _set_ bit in bs IdRange or BitSet (bs) parallel threads,
/// passing e.local() (evaluated once for each sub-range) as the second argument to f;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename L, typename F, typename ...Cb>
inline auto BitSetParallelFor( const BS& bs, tbb::enumerable_thread_specific<L> & e, F && f, Cb&&... cb )
{
    return BitSetParallelForAll( bs, e, [&] ( auto bit, auto & tls ) { if ( bs.test( bit ) ) std::forward<F>( f )( bit, tls ); }, std::forward<Cb>( cb )... );
}

/// \}

} // namespace MR
