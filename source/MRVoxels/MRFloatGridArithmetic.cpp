#include "MRFloatGrid.h"
#include "MRVDBFloatGrid.h"

#include "MRMesh/MRTimer.h"

namespace MR
{

FloatGrid operator += ( FloatGrid & a, FloatGrid&& b )
{
    MR_TIMER;
    openvdb::tools::csgUnion( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator+( const FloatGrid& a, const FloatGrid& b )
{
    MR_TIMER;
    return MakeFloatGrid( openvdb::tools::csgUnionCopy( ovdb( *a ), ovdb( *b ) ) );
}

FloatGrid operator -= ( FloatGrid & a, FloatGrid&& b )
{
    MR_TIMER;
    openvdb::tools::csgDifference( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator-( const FloatGrid& a, const FloatGrid& b )
{
    MR_TIMER;
    return MakeFloatGrid( openvdb::tools::csgDifferenceCopy( ovdb( *a ), ovdb( *b ) ) );
}

FloatGrid operator *= ( FloatGrid & a, FloatGrid&& b )
{
    MR_TIMER;
    openvdb::tools::csgIntersection( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator*( const FloatGrid& a, const FloatGrid& b )
{
    MR_TIMER;
    return MakeFloatGrid( openvdb::tools::csgIntersectionCopy( ovdb( *a ), ovdb( *b ) ) );
}

} //namespace MR
