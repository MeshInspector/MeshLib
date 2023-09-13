#include "MRBasinVolume.h"
#include "MRVector3.h"
#include "MRVector2.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include <algorithm>
#include <cassert>

namespace MR
{

bool BasinVolumeCalculator::addTerrainTri( Triangle3f t, float level )
{
    if ( t[0].z >= level && t[1].z >= level && t[2].z >= level )
        return false; // terrain triangle is fully above water level
    if ( t[0].z > level )
    {
        if ( t[1].z > level )
            std::rotate( t.data(), t.data() + 2, t.data() + 3 );
        else
            std::rotate( t.data(), t.data() + 1, t.data() + 3 );
    }
    else if ( t[1].z > level && t[2].z <= level )
        std::rotate( t.data(), t.data() + 2, t.data() + 3 );
    assert( t[0].z <= level );
    assert( t[1].z <= level || t[2].z > level );
    Vector3d ps[3] = { Vector3d( t[0] ), Vector3d( t[1] ), Vector3d( t[2] ) };
    if ( ps[2].z <= level )
    {
        // whole triangle below water level
        sum_ += ( level - ps[2].z ) * cross( Vector2d( ps[0] ), Vector2d( ps[1] ) )
            + dot( Vector2d( ps[2] ),
            ( level - ps[0].z ) * Vector2d( ps[1] ).perpendicular() - ( level - ps[1].z ) * Vector2d( ps[0] ).perpendicular() );
    }
    else if ( ps[1].z <= level )
    {
        // vertices 0,1 are below water level, and vertex 2 is above
        assert( ps[2].z > ps[0].z );
        assert( ps[2].z > ps[1].z );
        const auto a0 = ( level - ps[0].z ) / ( ps[2].z - ps[0].z );
        const auto x0 = ps[0] * ( 1 - a0 ) + ps[2] * a0;
        const auto a1 = ( level - ps[1].z ) / ( ps[2].z - ps[1].z );
        const auto x1 = ps[1] * ( 1 - a1 ) + ps[2] * a1;
        sum_ += ( level - ps[0].z ) * cross( Vector2d( x1 ), Vector2d( x0 ) )
            + dot( Vector2d( x1 ),
            ( level - ps[0].z ) * Vector2d( ps[1] ).perpendicular() - ( level - ps[1].z ) * Vector2d( ps[0] ).perpendicular() );
    }
    else
    {
        // vertex 0 below water level, and vertices 1,2 are above
        assert( ps[1].z > ps[0].z );
        assert( ps[2].z > ps[0].z );
        const auto a1 = ( level - ps[0].z ) / ( ps[1].z - ps[0].z );
        ps[1] = ps[0] * ( 1 - a1 ) + ps[1] * a1;
        const auto a2 = ( level - ps[0].z ) / ( ps[2].z - ps[0].z );
        ps[2] = ps[0] * ( 1 - a2 ) + ps[2] * a2;
        sum_ += ( level - ps[0].z ) * cross( Vector2d( ps[1] ), Vector2d( ps[2] ) );
    }
    return true;
}

double computeBasinVolume( const Mesh& mesh, const FaceBitSet& faces, float level )
{
    MR_TIMER
    BasinVolumeCalculator calc;
    for ( auto f : faces )
        calc.addTerrainTri( mesh.getTriPoints( f ), level );
    return calc.getVolume();
}

} //namespace MR
