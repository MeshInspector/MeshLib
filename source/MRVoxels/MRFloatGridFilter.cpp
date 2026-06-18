#include "MRFloatGrid.h"
#include "MRVDBFloatGrid.h"
#include "MRVDBProgressInterrupter.h"

#include "MRMesh/MRTimer.h"

namespace MR
{

size_t countVoxelsWithValuePred( const FloatGrid& grid, const std::function<bool( float )>& pred )
{
    MR_TIMER;
    if ( !pred || !grid )
    {
        assert( false );
        return 0;
    }
    tbb::enumerable_thread_specific<size_t> tls( 0 );
    openvdb::tools::foreach( grid->cbeginValueAll(), [&] ( const openvdb::FloatGrid::ValueAllCIter& it )
    {
        if ( !pred( it.getValue() ) )
            return;
        auto& local = tls.local();
        local += it.getBoundingBox().volume();
    } );
    size_t res = 0;
    for ( const auto& t : tls )
        res += t;
    return res;
}

size_t countVoxelsWithValueLess( const FloatGrid& grid, float value )
{
    return countVoxelsWithValuePred( grid, [value] ( float v )
    {
        return v < value;
    } );
}

size_t countVoxelsWithValueGreater( const FloatGrid& grid, float value )
{
    return countVoxelsWithValuePred( grid, [value] ( float v )
    {
        return v > value;
    } );
}

void gaussianFilter( FloatGrid& grid, int width, int iters, ProgressCallback cb /*= {} */ )
{
    if ( !grid )
        return;
    // just grows to 100%
    // first grows fast, then slower
    ProgressCallback dummyProgressCb;
    float i = 1.0f;
    if ( cb )
        dummyProgressCb = [&] ( float )->bool
    {
        i += 1e-4f;
        return cb( 1.0f - 1.0f / std::sqrt( i ) );
    };
    ProgressInterrupter interrupter( dummyProgressCb );

    auto filter = openvdb::tools::Filter<openvdb::FloatGrid, openvdb::FloatGrid::ValueConverter<float>::Type, ProgressInterrupter>( ovdb( *grid ), &interrupter );
    filter.gaussian( width, iters );
}

FloatGrid gaussianFiltered( const FloatGrid& grid, int width, int iters, ProgressCallback cb /*= {} */ )
{
    if ( !grid )
        return {};
    auto dest = MakeFloatGrid( grid->deepCopy() );

    gaussianFilter( dest, width, iters, cb );
    return dest;
}

} //namespace MR
