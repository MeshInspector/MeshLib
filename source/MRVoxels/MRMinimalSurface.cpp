#include "MRMinimalSurface.h"

#include <MRMesh/MRVector3.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRMesh.h"
#include <MRMesh/MRMeshBoolean.h>

#include <MRVoxels/MRMarchingCubes.h>

#include <vector>
#include <string>

namespace MR
{

std::vector<std::string> getTPMSTypeNames()
{
    return {
        "Schwartz-P",
        "Double Schwartz-P",
        "Gyroid",
        "Double Gyroid"
    };
}

namespace TPMSFunctions
{

float SchwartzP( const Vector3f& p )
{
    return cos( p.x ) + cos( p.y ) + cos( p.z );
}
float DoubleSchwartzP( const Vector3f& p )
{
    return cos( p.x )*cos( p.y ) + cos( p.y )*cos( p.z ) + cos( p.x )*cos( p.z ) + 0.35f*(cos( 2*p.x ) + cos( 2*p.y ) + cos( 2*p.z ));
}

float Gyroid( const Vector3f& p )
{
    return cos( p.x )*sin( p.y ) + cos( p.y )*sin( p.x ) + cos( p.z )*sin( p.x );
}
float DoubleGyroid( const Vector3f& p )
{
    return 2.75f * ( sin(2*p.x)*sin(p.z)*cos(p.y) + sin(2*p.y)*sin(p.x)*cos(p.z) + sin(2*p.z)*sin(p.y)*cos(p.x) )
           - ( cos(2*p.x)*cos(2*p.y) + cos(2*p.y)*cos(2*p.z) + cos(2*p.z)*cos(2*p.x) );
}

};


TPMSFunction getTPMSFunction( TPMSType type )
{
    switch ( type )
    {
        case TPMSType::SchwartzP:
            return TPMSFunctions::SchwartzP;
        case TPMSType::DoubleSchwartzP:
            return TPMSFunctions::DoubleSchwartzP;
        case TPMSType::Gyroid:
            return TPMSFunctions::Gyroid;
        case TPMSType::DoubleGyroid:
            return TPMSFunctions::DoubleGyroid;
        default:
            assert( false );
            return TPMSFunctions::SchwartzP;
    }
}

FunctionVolume buildTPMSVolume( TPMSType type, const Vector3f& size, float frequency, float resolution )
{
    const auto N = frequency * size;            // number of repetitions (for each axis)
    const auto dimsF = resolution * N;          // float-dimensions: number of voxels per repetition times the number of repetitions
    const auto voxelSize = div( size, dimsF );  // voxel-size: size divided by the number of voxels
    const Vector3i dims( std::ceil( dimsF.x ), std::ceil( dimsF.y ), std::ceil( dimsF.z ) );

    return {
        .data = [frequency, voxelSize, func = getTPMSFunction( type )] ( const Vector3i& pv )
        {
            const float w = 2.f * PI_F * frequency;
            const Vector3f p = w * mult( voxelSize, Vector3f( pv ) + Vector3f::diagonal( 0.5f ) );
            return func( p );
        },
        .dims = dims,
        .voxelSize = voxelSize,
    };
}


Expected<Mesh> buildTPMSSurface( TPMSType type, const Vector3f& size, float frequency, float resolution, float iso )
{
    return marchingCubes( buildTPMSVolume( type, size, frequency, resolution ), { .iso = iso } );
}

Expected<Mesh> buildTPMSSurface( TPMSType type, const Mesh& mesh, float frequency, float resolution, float iso )
{
    // first construct a surface by the bounding box of the mesh
    const auto extraStep = Vector3f::diagonal( 1.f / frequency );
    auto sponge = buildTPMSSurface( type, mesh.getBoundingBox().size() + 1.5f*extraStep, frequency, resolution, iso );
    if ( !sponge )
        return sponge;

    // translation to mesh csys
    const auto xf = AffineXf3f::translation( mesh.getBoundingBox().min - 0.75f*extraStep );

    auto res = boolean( mesh, *sponge, BooleanOperation::Intersection, &xf );
    if ( !res )
        return unexpected( res.errorString );

    res.mesh.topology.flipOrientation();
    return res.mesh;
}




};
