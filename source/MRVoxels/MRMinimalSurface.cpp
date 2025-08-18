#include "MRMinimalSurface.h"

#include <MRMesh/MRVector3.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshDecimate.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshComponents.h"
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
        "Rectangular",
        "Thick Rectangular",
        "Double Gyroid",
        "Thick Gyroid"
    };
}

std::vector<std::string> getTypeTooltips()
{
    return {
        "Schwartz-P surface.",
        "Schwartz-P with thick walls.",
        "",
        ""
    };
}

bool isThick( Type type )
{
    return type == Type::ThickSchwartzP || type == Type::ThickGyroid;
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
float ThickSchwartzP( const Vector3f& p )
{
    return std::abs( SchwartzP( p ) );
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
float ThickGyroid( const Vector3f& p )
{
    return std::abs( Gyroid( p ) );
}

}
TPMSFunction getTPMSFunction( Type type )
{
    switch ( type )
    {
        case Type::SchwartzP:
            return TPMSFunctions::SchwartzP;
        case Type::ThickSchwartzP:
            return TPMSFunctions::ThickSchwartzP;
        case Type::DoubleGyroid:
            return TPMSFunctions::DoubleGyroid;
        case Type::ThickGyroid:
            return TPMSFunctions::ThickGyroid;
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
        // SchwartzP
        {
            {0.788112, -1},
            {0.77365, -0.95},
            {0.759149, -0.9},
            {0.744631, -0.85},
            {0.730192, -0.8},
            {0.715688, -0.75},
            {0.701236, -0.7},
            {0.686835, -0.65},
            {0.672414, -0.6},
            {0.657971, -0.55},
            {0.643578, -0.5},
            {0.629213, -0.45},
            {0.614841, -0.4},
            {0.600454, -0.35},
            {0.586076, -0.3},
            {0.57173, -0.25},
            {0.557386, -0.2},
            {0.543039, -0.15},
            {0.528688, -0.1},
            {0.514339, -0.05},
            {0.499995, 0},
            {0.485654, 0.05},
            {0.471314, 0.1},
            {0.456975, 0.15},
            {0.442625, 0.2},
            {0.42826, 0.25},
            {0.413898, 0.3},
            {0.399542, 0.35},
            {0.385189, 0.4},
            {0.370805, 0.45},
            {0.356396, 0.5},
            {0.341994, 0.55},
            {0.327614, 0.6},
            {0.313206, 0.65},
            {0.298727, 0.7},
            {0.284284, 0.75},
            {0.269869, 0.8},
            {0.255334, 0.85},
            {0.240855, 0.9},
            {0.226326, 0.95},
            {0.211894, 1},
        },
        // ThickSchwartzP
        {
            {0, 0},
            {0.000585522, 0.05},
            {0.0102221, 0.1},
            {0.0604986, 0.15},
            {0.112434, 0.2},
            {0.143471, 0.25},
            {0.172179, 0.3},
            {0.200912, 0.35},
            {0.229652, 0.4},
            {0.258408, 0.45},
            {0.287183, 0.5},
            {0.315977, 0.55},
            {0.344799, 0.6},
            {0.373628, 0.65},
            {0.40251, 0.7},
            {0.431404, 0.75},
            {0.460324, 0.8},
            {0.489297, 0.85},
            {0.518295, 0.9},
            {0.547325, 0.95},
            {0.576218, 1},
        },
        // DoubleGyroid
        {
            {0.59417, -1},
            {0.584924, -0.95},
            {0.575915, -0.9},
            {0.56694, -0.85},
            {0.558064, -0.8},
            {0.549413, -0.75},
            {0.541021, -0.7},
            {0.532707, -0.65},
            {0.524562, -0.6},
            {0.516584, -0.55},
            {0.508716, -0.5},
            {0.500908, -0.45},
            {0.493201, -0.4},
            {0.485642, -0.35},
            {0.478079, -0.3},
            {0.470608, -0.25},
            {0.463317, -0.2},
            {0.456122, -0.15},
            {0.448956, -0.1},
            {0.441967, -0.05},
            {0.435017, 0},
            {0.428076, 0.05},
            {0.421149, 0.1},
            {0.414261, 0.15},
            {0.407567, 0.2},
            {0.401135, 0.25},
            {0.394467, 0.3},
            {0.387791, 0.35},
            {0.381257, 0.4},
            {0.374834, 0.45},
            {0.368415, 0.5},
            {0.362159, 0.55},
            {0.355885, 0.6},
            {0.349656, 0.65},
            {0.34338, 0.7},
            {0.337281, 0.75},
            {0.331181, 0.8},
            {0.325112, 0.85},
            {0.319143, 0.9},
            {0.313066, 0.95},
            {0.306966, 1},
        },
        // ThickGyroid
        {
            {0, 0},
            {0.00715582, 0.05},
            {0.0320288, 0.1},
            {0.0854733, 0.15},
            {0.145352, 0.2},
            {0.194796, 0.25},
            {0.236609, 0.3},
            {0.275085, 0.35},
            {0.312034, 0.4},
            {0.348159, 0.45},
            {0.384236, 0.5},
            {0.419774, 0.55},
            {0.455202, 0.6},
            {0.490089, 0.65},
            {0.524965, 0.7},
            {0.559502, 0.75},
            {0.593942, 0.8},
            {0.628168, 0.85},
            {0.662269, 0.9},
            {0.696453, 0.95},
            {0.730294, 1},
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


FunctionVolume buildVolume( const Vector3f& size, const VolumeParams& params )
{
    const auto [dims, voxelSize] = getDimsAndSize( size, params.frequency, params.resolution );
    return {
        .data = [frequency = params.frequency, voxelSizeCapture = voxelSize, func = getTPMSFunction( params.type )] ( const Vector3i& pv )
        {
            const float w = 2.f * PI_F * frequency;
            const Vector3f p = w * mult( voxelSizeCapture, Vector3f( pv ) + Vector3f::diagonal( 0.5f ) );
            return func( p );
        },
        .dims = dims,
        .voxelSize = voxelSize,
    };
}


Expected<Mesh> build( const Vector3f& size, const MeshParams& params, ProgressCallback cb )
{
    MR_TIMER;
    ProgressCallback mcProgress, decProgress;
    if ( params.decimate )
    {
        mcProgress = subprogress( cb, 0.f, 0.8f );
        decProgress = subprogress( cb, 0.8f, 1.f );
    }
    else
    {
        mcProgress = cb;
    }

    auto res = marchingCubes( buildVolume( size, params ), { .cb = mcProgress, .iso = params.iso } );
    if ( !res )
        return res;
    if ( isThick( params.type ) )
        res->topology.flipOrientation();

    if ( params.decimate )
    {
        const auto voxelSize = getDimsAndSize( size, params.frequency, params.resolution ).size;
        decimateMesh( *res, DecimateSettings{ .maxError = std::min( voxelSize.x, std::min( voxelSize.y, voxelSize.z ) ), .progressCallback = decProgress } );
    }
    return res;
}

Expected<Mesh> fill( const Mesh& mesh, const MeshParams& params, ProgressCallback cb )
{
    MR_TIMER;
    // first construct a surface by the bounding box of the mesh
    const auto extraStep = Vector3f::diagonal( 1.f / params.frequency );
    auto sponge = build( mesh.getBoundingBox().size() + 1.5f*extraStep, params, subprogress( cb, 0.f, 0.9f ) );
    if ( !sponge )
        return sponge;

    // translation to mesh csys
    const auto xf = AffineXf3f::translation( mesh.getBoundingBox().min - 0.75f*extraStep );

    BooleanOperation booleanOp = isThick( params.type ) ? BooleanOperation::OutsideB : BooleanOperation::Union;
    auto res = boolean( mesh, *sponge, booleanOp, &xf, nullptr, subprogress( cb, 0.9f, 1.f ) );
    if ( !res )
        return unexpected( res.errorString );

    if ( params.type != Type::ThickGyroid ) // this surface inherently has disconnected components by the layers
    {
        auto largestComponents = MeshComponents::getNLargeByAreaComponents( MeshPart{ *res }, { .maxLargeComponents = isThick( params.type ) ? 2 : 1 } );
        FaceBitSet principalSurface;
        for ( const auto& c : largestComponents )
            principalSurface |= c;
        res->deleteFaces( res->topology.getValidFaces() - principalSurface );
    }
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

float getMinimalResolution( Type type, float iso )
{
    // voxel size == 1 / (res * freq) <= delta
    float delta = 1.f;
    const auto k = 2 * PI_F;
//    const auto w = 2 * PI_F * frequency;
    switch ( type )
    {
        case Type::ThickSchwartzP:
            delta = 2 * std::asin( iso / 2.f ) / k; //  / w
            break;
        case Type::ThickGyroid:
            delta = 2 * std::asin( iso / 4.f ) / k; // / w
            break;
        case Type::SchwartzP:
            delta = std::acos( iso ) / k; // / w
            break;
        case Type::DoubleGyroid:
            delta = 1.f; // it seems that 5 is always enough for a double gyroid
            break;
        default:
            assert( false );
            delta = 1.f;
    }

    // 1 / (res * freq) <= delta => res >= 1 / (delta * freq)
    return std::max( 5.f, 1.f / delta );
}


}
