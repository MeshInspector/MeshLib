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


float interpolateMap( const std::map<float, float>& map, float key )
{
    auto itUp = map.upper_bound( key );
    if ( itUp == map.end() )
        return map.rbegin()->second;
    if ( itUp == map.begin() )
        return map.begin()->second;
    return ( std::prev( itUp )->second + itUp->second ) / 2.f;
}


enum class InterpolateDensityAndIsoDirection
{
    density2iso, iso2density
};
float interpolateDensityAndIso( InterpolateDensityAndIsoDirection direction, Type type, float key )
{
    static std::map<float, float> density2iso[(int)Type::Count] =
    {
        {
            { 0.788975f, -1.f },
            { 0.759864f, -0.9f },
            { 0.730905f, -0.8f },
            { 0.701789f, -0.7f },
            { 0.67295f,  -0.6f },
            { 0.644014f, -0.5f },
            { 0.615147f, -0.4f },
            { 0.586371f, -0.3f },
            { 0.557551f, -0.2f },
            { 0.52876f,  -0.1f },
            { 0.5f,       0.f },
            { 0.47124f,   0.1f },
            { 0.442449f,  0.2f },
            { 0.413629f,  0.3f },
            { 0.384853f,  0.4f },
            { 0.355985f,  0.5f },
            { 0.32705f,   0.6f },
            { 0.29821f,   0.7f },
            { 0.269095f,  0.8f },
            { 0.240136f,  0.9f },
            { 0.211025f,  1.f },
        },
        {
            { 0.997467f, -1.f },
            { 0.970437f, -0.9f },
            { 0.917148f, -0.8f },
            { 0.798708f, -0.7f },
            { 0.679275f, -0.6f },
            { 0.598016f, -0.5f },
            { 0.53185f,  -0.4f },
            { 0.475072f, -0.3f },
            { 0.423839f, -0.2f },
            { 0.376103f, -0.1f },
            { 0.331773f,  0.f },
            { 0.295088f,  0.1f },
            { 0.267647f,  0.2f },
            { 0.244183f,  0.3f },
            { 0.223739f,  0.4f },
            { 0.205919f,  0.5f },
            { 0.190089f,  0.6f },
            { 0.175405f,  0.7f },
            { 0.161777f,  0.8f },
            { 0.149396f,  0.9f },
            { 0.138025f,  1.f },
        },
        {
            { 0.866612f, -1.f },
            { 0.833128f, -0.9f },
            { 0.798331f, -0.8f },
            { 0.763666f, -0.7f },
            { 0.728875f, -0.6f },
            { 0.693088f, -0.5f },
            { 0.656571f, -0.4f },
            { 0.619761f, -0.3f },
            { 0.581685f, -0.2f },
            { 0.541816f, -0.1f },
            { 0.500168f,  0.f },
            { 0.458244f,  0.1f },
            { 0.418315f,  0.2f },
            { 0.380271f,  0.3f },
            { 0.343429f,  0.4f },
            { 0.306904f,  0.5f },
            { 0.271125f,  0.6f },
            { 0.236334f,  0.7f },
            { 0.201697f,  0.8f },
            { 0.166872f,  0.9f },
            { 0.133388f,  1.f },
        },
        {
            { 0.594462f, -1.f },
            { 0.576542f, -0.9f },
            { 0.559056f, -0.8f },
            { 0.542165f, -0.7f },
            { 0.525182f, -0.6f },
            { 0.508817f, -0.5f },
            { 0.492698f, -0.4f },
            { 0.476833f, -0.3f },
            { 0.463394f, -0.2f },
            { 0.449116f, -0.1f },
            { 0.435347f,  0.f },
            { 0.42125f,   0.1f },
            { 0.406499f,  0.2f },
            { 0.392565f,  0.3f },
            { 0.378933f,  0.4f },
            { 0.365132f,  0.5f },
            { 0.353343f,  0.6f },
            { 0.340634f,  0.7f },
            { 0.32839f,   0.8f },
            { 0.316597f,  0.9f },
            { 0.304421f,  1.f },
        }
    };

    static std::map<float, float> iso2density[(int)Type::Count] =
    {
        reverseMap( density2iso[0] ),
        reverseMap( density2iso[1] ),
        reverseMap( density2iso[2] ),
        reverseMap( density2iso[3] ),
    };

    const auto& map = direction == InterpolateDensityAndIsoDirection::iso2density ? iso2density : density2iso;
    const int itype = static_cast<int>( type );
    assert( itype < (int)Type::Count );
    return interpolateMap( map[itype], key );
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
    return interpolateDensityAndIso( InterpolateDensityAndIsoDirection::density2iso, type, targetDensity );
}

float estimateDensity( Type type, float targetIso )
{
    return interpolateDensityAndIso( InterpolateDensityAndIsoDirection::iso2density, type, targetIso );
}

}
