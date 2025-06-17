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

using TPMSFunction = float(*)( const Vector3f& );
namespace TPMSFunctions
{

float SchwartzP( const Vector3f& p )
{
    return std::cos( p.x ) + std::cos( p.y ) + std::cos( p.z );
}
float DoubleSchwartzP( const Vector3f& p )
{
    return std::cos( p.x )*std::cos( p.y ) + std::cos( p.y )*std::cos( p.z ) + std::cos( p.x )*std::cos( p.z ) + 0.35f*(std::cos( 2*p.x ) + std::cos( 2*p.y ) + std::cos( 2*p.z ));
}

float Gyroid( const Vector3f& p )
{
    return std::cos( p.x )*std::sin( p.y ) + std::cos( p.y )*std::sin( p.x ) + std::cos( p.z )*std::sin( p.x );
}
float DoubleGyroid( const Vector3f& p )
{
    return 2.75f * ( std::sin(2*p.x)*std::sin(p.z)*std::cos(p.y) + std::sin(2*p.y)*std::sin(p.x)*std::cos(p.z) + std::sin(2*p.z)*std::sin(p.y)*std::cos(p.x) )
           - ( std::cos(2*p.x)*std::cos(2*p.y) + std::cos(2*p.y)*std::cos(2*p.z) + std::cos(2*p.z)*std::cos(2*p.x) );
}

}
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


namespace
{

struct DimsAndSize
{
    Vector3i dims;
    Vector3f size;
};
DimsAndSize getDimsAndSize( const Vector3f& size, float frequency, float resolution )
{
    const auto N = frequency * size;            // number of repetitions (for each axis)
    const auto dimsF = resolution * N;          // float-dimensions: number of voxels per repetition times the number of repetitions
    const auto voxelSize = div( size, dimsF );  // voxel-size: size divided by the number of voxels
    const Vector3i dims( (int)std::ceil( dimsF.x ), (int)std::ceil( dimsF.y ), (int)std::ceil( dimsF.z ) );
    return { dims, voxelSize };
}

}


FunctionVolume buildTPMSVolume( TPMSType type, const Vector3f& size, float frequency, float resolution )
{
    const auto [dims, voxelSize] = getDimsAndSize( size, frequency, resolution );
    return {
        .data = [frequency, voxelSizeCapture = voxelSize, func = getTPMSFunction( type )] ( const Vector3i& pv )
        {
            const float w = 2.f * PI_F * frequency;
            const Vector3f p = w * mult( voxelSizeCapture, Vector3f( pv ) + Vector3f::diagonal( 0.5f ) );
            return func( p );
        },
        .dims = dims,
        .voxelSize = voxelSize,
    };
}


Expected<Mesh> buildTPMS( TPMSType type, const Vector3f& size, float frequency, float resolution, float iso, ProgressCallback cb )
{
    return marchingCubes( buildTPMSVolume( type, size, frequency, resolution ), { .cb = cb, .iso = iso } );
}

Expected<Mesh> fillWithTPMS( TPMSType type, const Mesh& mesh, float frequency, float resolution, float iso, ProgressCallback cb )
{
    // first construct a surface by the bounding box of the mesh
    const auto extraStep = Vector3f::diagonal( 1.f / frequency );
    auto sponge = buildTPMS( type, mesh.getBoundingBox().size() + 1.5f*extraStep, frequency, resolution, iso, subprogress( cb, 0.f, 0.9f ) );
    if ( !sponge )
        return sponge;

    // translation to mesh csys
    const auto xf = AffineXf3f::translation( mesh.getBoundingBox().min - 0.75f*extraStep );

    auto res = boolean( mesh, *sponge, BooleanOperation::Intersection, &xf, nullptr, subprogress( cb, 0.9f, 1.f ) );
    if ( !res )
        return unexpected( res.errorString );

    res.mesh.topology.flipOrientation();
    return std::move( res.mesh );
}

size_t getNumberOfVoxelsForTPMS( const Mesh& mesh, float frequency, float resolution )
{
    const auto extraStep = Vector3f::diagonal( 1.f / frequency );
    const auto dims = getDimsAndSize( mesh.getBoundingBox().size() + 1.5f*extraStep, frequency, resolution ).dims;
    return (size_t)dims.x * (size_t)dims.y * (size_t)dims.z;
}


}
