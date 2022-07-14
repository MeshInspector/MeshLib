#ifndef __EMSCRIPTEN__
#include "MRBoolean.h"
#include "MRMesh.h"
#include "MRObjectMesh.h"
#include "MRFloatGrid.h"
#include "MRTimer.h"
#include "MRPch/MROpenvdb.h"
#include <filesystem>

namespace MR
{

FloatGrid MeshVoxelsConverter::operator() ( const ObjectMesh & obj ) const
{ 
    return meshToLevelSet( *obj.mesh(), obj.xf(), Vector3f::diagonal( voxelSize ), surfaceOffset, callBack );
}

Mesh MeshVoxelsConverter::operator() ( const FloatGrid & grid ) const
{ 
    return gridToMesh( grid, Vector3f::diagonal( voxelSize ), offsetVoxels, adaptivity, INT_MAX, callBack ).value();
}

FloatGrid operator += ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER
    openvdb::tools::csgUnion( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator -= ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER
    openvdb::tools::csgDifference( ovdb( *a ), ovdb( *b ) );
    return a;
}

FloatGrid operator *= ( FloatGrid & a, const FloatGrid & b )
{
    MR_TIMER
    openvdb::tools::csgIntersection( ovdb( *a ), ovdb( *b ) );
    return a;
}

} //namespace MR
#endif