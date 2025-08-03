#include "MRMinimalSurface.h"

#include <MRMesh/MRVector3.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRMesh.h"
#include <MRMesh/MRMeshBoolean.h>

#include <MRVoxels/MRMarchingCubes.h>

#include <vector>
#include <string>
#include <map>

namespace MR::TPMS
{

std::vector<std::string> getTypeNames()
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
TPMSFunction getTPMSFunction( Type type )
{
    switch ( type )
    {
        case Type::SchwartzP:
            return TPMSFunctions::SchwartzP;
        case Type::DoubleSchwartzP:
            return TPMSFunctions::DoubleSchwartzP;
        case Type::Gyroid:
            return TPMSFunctions::Gyroid;
        case Type::DoubleGyroid:
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


std::map<float, float> reverseMap( const std::map<float, float>& other )
{
    std::map<float, float> res;
    for ( auto [k, v] : other )
        res[v] = k;
    return res;
}

const std::vector<std::map<float, float>> density2iso = {
    {
        {0.788975, -1},
        {0.759864, -0.9},
        {0.730905, -0.8},
        {0.701789, -0.7},
        {0.67295, -0.6},
        {0.644014, -0.5},
        {0.615147, -0.4},
        {0.586371, -0.3},
        {0.557551, -0.2},
        {0.52876, -0.1},
        {0.5, 0},
        {0.47124, 0.1},
        {0.442449, 0.2},
        {0.413629, 0.3},
        {0.384853, 0.4},
        {0.355985, 0.5},
        {0.32705, 0.6},
        {0.29821, 0.7},
        {0.269095, 0.8},
        {0.240136, 0.9},
        {0.211025, 1},
    },
    {
        {0.997467, -1},
        {0.970437, -0.9},
        {0.917148, -0.8},
        {0.798708, -0.7},
        {0.679275, -0.6},
        {0.598016, -0.5},
        {0.53185, -0.4},
        {0.475072, -0.3},
        {0.423839, -0.2},
        {0.376103, -0.1},
        {0.331773, 0},
        {0.295088, 0.1},
        {0.267647, 0.2},
        {0.244183, 0.3},
        {0.223739, 0.4},
        {0.205919, 0.5},
        {0.190089, 0.6},
        {0.175405, 0.7},
        {0.161777, 0.8},
        {0.149396, 0.9},
        {0.138025, 1},
    },
    {
        {0.866612, -1},
        {0.833128, -0.9},
        {0.798331, -0.8},
        {0.763666, -0.7},
        {0.728875, -0.6},
        {0.693088, -0.5},
        {0.656571, -0.4},
        {0.619761, -0.3},
        {0.581685, -0.2},
        {0.541816, -0.1},
        {0.500168, 0},
        {0.458244, 0.1},
        {0.418315, 0.2},
        {0.380271, 0.3},
        {0.343429, 0.4},
        {0.306904, 0.5},
        {0.271125, 0.6},
        {0.236334, 0.7},
        {0.201697, 0.8},
        {0.166872, 0.9},
        {0.133388, 1},
    },
    {
        {0.594462, -1},
        {0.576542, -0.9},
        {0.559056, -0.8},
        {0.542165, -0.7},
        {0.525182, -0.6},
        {0.508817, -0.5},
        {0.492698, -0.4},
        {0.476833, -0.3},
        {0.463394, -0.2},
        {0.449116, -0.1},
        {0.435347, 0},
        {0.42125, 0.1},
        {0.406499, 0.2},
        {0.392565, 0.3},
        {0.378933, 0.4},
        {0.365132, 0.5},
        {0.353343, 0.6},
        {0.340634, 0.7},
        {0.32839, 0.8},
        {0.316597, 0.9},
        {0.304421, 1},
    }
};

const std::vector<std::map<float, float>> iso2density = {
    reverseMap( density2iso[0] ),
    reverseMap( density2iso[1] ),
    reverseMap( density2iso[2] ),
    reverseMap( density2iso[3] ),
};


float interpolateMap( const std::map<float, float>& map, float key )
{
    auto itUp = map.upper_bound( key );
    if ( itUp == map.end() )
        return map.rbegin()->second;
    if ( itUp == map.begin() )
        return map.begin()->second;
    return ( std::prev( itUp )->second + itUp->second ) / 2.f;
}

}


FunctionVolume buildVolume( Type type, const Vector3f& size, float frequency, float resolution )
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


Expected<Mesh> build( Type type, const Vector3f& size, float frequency, float resolution, float iso, ProgressCallback cb )
{
    return marchingCubes( buildVolume( type, size, frequency, resolution ), { .cb = cb, .iso = iso } );
}

Expected<Mesh> fill( Type type, const Mesh& mesh, float frequency, float resolution, float iso, ProgressCallback cb )
{
    // first construct a surface by the bounding box of the mesh
    const auto extraStep = Vector3f::diagonal( 1.f / frequency );
    auto sponge = build( type, mesh.getBoundingBox().size() + 1.5f*extraStep, frequency, resolution, iso, subprogress( cb, 0.f, 0.9f ) );
    if ( !sponge )
        return sponge;

    // translation to mesh csys
    const auto xf = AffineXf3f::translation( mesh.getBoundingBox().min - 0.75f*extraStep );
//  this looks like a boolean bug
//    auto res = boolean( mesh, *sponge, BooleanOperation::Intersection, &xf, nullptr, subprogress( cb, 0.9f, 1.f ) );

    sponge->transform( xf );
    auto res = boolean( mesh, *sponge, BooleanOperation::Intersection, nullptr, nullptr, subprogress( cb, 0.9f, 1.f ) );
    if ( !res )
        return unexpected( res.errorString );

    return std::move( res.mesh );
}

size_t getNumberOfVoxels( const Mesh& mesh, float frequency, float resolution )
{
    const auto extraStep = Vector3f::diagonal( 1.f / frequency );
    const auto dims = getDimsAndSize( mesh.getBoundingBox().size() + 1.5f*extraStep, frequency, resolution ).dims;
    return (size_t)dims.x * (size_t)dims.y * (size_t)dims.z;
}

size_t getNumberOfVoxels( const Vector3f& size, float frequency, float resolution )
{
    const auto dims = getDimsAndSize( size, frequency, resolution ).dims;
    return (size_t)dims.x * (size_t)dims.y * (size_t)dims.z;
}

float estimateIso( Type type, float targetDensity )
{
    int itype = static_cast<int>( type );
    assert( itype < density2iso.size() );
    return interpolateMap( density2iso[itype], targetDensity );
}

float estimateDensity( Type type, float targetIso )
{
    int itype = static_cast<int>( type );
    assert( itype < iso2density.size() );
    return interpolateMap( iso2density[itype], targetIso );
}

}
