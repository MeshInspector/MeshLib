#include "MRFillingSurface.h"

#include <MRMesh/MRVector3.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshDecimate.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRBestFitPolynomial.h"
#include "MRMesh/MRCube.h"
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRConstants.h>
#include <MRPch/MRFmt.h>

#include <MRVoxels/MRMarchingCubes.h>

#include <vector>
#include <string>
#include <map>


// Question-mark operator for Expected, inspired from Rust.
#define QME( __dest, __val_expr ) \
    if ( auto __val = (__val_expr); !(__val) ) \
        return unexpected( (__val).error() ); \
    else \
        (__dest) = std::move( __val.value() );

namespace MR::FillingSurface
{

namespace
{

struct SizeAndXf
{
    Vector3f size;
    AffineXf3f xf;
};

/// Given mesh, returns the size of the filling surface (with padding) and its transform back to mesh
SizeAndXf getFillingSizeAndXf( const Mesh& mesh, float period )
{
    const auto extraStep = Vector3f::diagonal( period );

    SizeAndXf ret;
    ret.xf = AffineXf3f::translation( mesh.getBoundingBox().min - 0.75f*extraStep );
    ret.size = mesh.getBoundingBox().size() + 1.5f*extraStep;
    return ret;
}

}

namespace TPMS
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

Polynomialf<1> inverseLinearFunc( const Polynomialf<1>& f )
{
    // y = a + b*x => x = ( y - a ) / b
    assert( f.a[1] != 0 );
    return { { -f.a[0] / f.a[1], 1.f / f.a[1] } };
}

enum class InterpolateDensityAndIsoDirection
{
    density2iso, iso2density
};
float interpolateDensityAndIso( InterpolateDensityAndIsoDirection direction, Type type, float key )
{
    static const Polynomialf<1> iso2density[(int)Type::Count] =
    {
        Polynomialf<1>{ { 0.5f, -0.286268f } }, // SchwartzP
        Polynomialf<1>{ { 0.f, 0.573504f } },   // ThickSchwartzP
        Polynomialf<1>{ { 0.441534f, -0.139001f } },   // DoubleGyroid
        Polynomialf<1>{ { 0.014612f, 0.719944f } },   // ThickGyroid
    };
    static const Polynomialf<1> density2iso[(int)Type::Count] =
    {
        inverseLinearFunc( iso2density[0] ),
        inverseLinearFunc( iso2density[1] ),
        inverseLinearFunc( iso2density[2] ),
        inverseLinearFunc( iso2density[3] ),
    };

    return ( direction == InterpolateDensityAndIsoDirection::density2iso ? density2iso : iso2density )[(int)type]( key );
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
    auto [size, xf] = getFillingSizeAndXf( mesh, 1.f / params.frequency );

    auto sponge = build( size, params, subprogress( cb, 0.f, 0.9f ) );
    if ( !sponge )
        return sponge;

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
    const auto dims = getDimsAndSize( getFillingSizeAndXf( mesh, 1.f / frequency ).size, frequency, resolution ).dims;
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

} // namespace TPMS


namespace CellularSurface
{

std::vector<std::string> getTypeNames()
{
    return { "Cylinder", "Rectangle" };
}

namespace
{

struct AbsentTip
{
    int dir = 1; int ax = 0;
    bool operator==( const AbsentTip& ) const = default;
    auto operator<=>( const AbsentTip& ) const = default;
};
using AbsentTips = std::vector<AbsentTip>;


std::vector<AbsentTip> getAbsentTips( const Vector3i& idx, const Vector3i& size )
{
    std::vector<AbsentTip> res;
    for ( int i = 0; i < 3; ++i )
    {
        if ( idx[i] == 0 )
            res.push_back( { -1, i } );
        if ( idx[i] == size[i] )
            res.push_back( { 1, i } );
    }
    return res;
}

Mesh makeBarForBaseElement( Type type, float width, float length, int res )
{
    switch ( type )
    {
        case Type::Cylinder:
            return makeCylinder( width / 2.f, length, res );
        case Type::Rect:
            return makeCube( { width, width, length }, Vector3f{ -width, -width, 0.f } / 2.f );
        default:
            assert( false );
            return {};
    }
}

constexpr float normalEps = 1e-5f;
constexpr float decimateEps = 1e-3f;

Expected<Mesh> makeBaseElement( const Params& params, const AbsentTips& absentTips )
{
    Mesh baseElement;
    if ( params.r > std::sqrt( 3.f ) * std::min( params.width.x, std::min( params.width.y, params.width.z ) ) / 2.f  )
        baseElement.addMesh( makeSphere( { .radius = params.r, .numMeshVertices = params.highRes ? 500 : 100 } ) );

    for ( int ax = 0; ax < 3; ++ax )
    {
        Vector3f s;
        std::vector<int> dirs{ -1, 1 };
        s[ax] = -params.period[ax] / 2.f;
        for ( const auto& tip : absentTips )
        {
            if ( tip.ax == ax )
            {
                erase_if( dirs, std::bind_front( std::equal_to{}, tip.dir ) );
                if ( tip.dir == -1 )
                    s[ax] += params.period[ax] / 2.f;
                if ( tip.dir == 1 )
                    s[ax] -= decimateEps / 10.f; // should 0 but causes boolean failure when sphere radius is 0 !
            }
        }

        float l;
        if ( dirs.size() == 2 )
            l = params.period[ax];
        else if ( dirs.size() == 1 )
            l = params.period[ax] / 2.f;
        else
            continue;
        auto bar = makeBarForBaseElement( params.type, params.width[ax], l, params.highRes ? 64 : 16 );

        FaceBitSet cylToDel;
        for ( auto f : bar.topology.getValidFaces() )
        {
            auto n =  bar.normal( f );
            for ( int d : dirs )
            {
                if ( std::abs( n.z - (float)d ) < normalEps )
                    cylToDel.autoResizeSet( f, true );
            }
        }
        bar.deleteFaces( cylToDel );

        AffineXf3f tr;
        if ( ax == 0 )
            tr.A = Matrix3f::rotation( Vector3f::plusY(), PI2_F );
        if ( ax == 1 )
            tr.A = Matrix3f::rotation( Vector3f::plusX(), -PI2_F );

        tr = AffineXf3f::translation( s ) * tr;
        bar.transform( tr );
        auto r = boolean( baseElement, bar, BooleanOperation::Union );
        if ( !r )
            return unexpected( r.errorString );
        baseElement = std::move( r.mesh );
    }

    decimateMesh( baseElement, { .maxError = decimateEps, .stabilizer = 1e-5f, .touchNearBdEdges = false, .touchBdVerts = false   } );
    return baseElement;
}
}

Expected<Mesh> build( const Vector3f& size, const Params& params, const ProgressCallback& cb )
{
    MR_TIMER;

    const auto delta = params.period - params.width;
    if ( delta.x <= 0 || delta.y <= 0 || delta.z <= 0 )
        return unexpected( "Period must be larger than width" );

    auto getAbsentTipsIfNeeded = [&params] ( const Vector3i& idx, const Vector3i& size ) -> AbsentTips
    {
        if ( params.preserveTips )
            return {};
        return getAbsentTips( idx, size );
    };

    reportProgress( cb, 0.f );
    const Vector3i dims( int( size.x / params.period.x ), int( size.y / params.period.y ), int( size.z / params.period.z ) );
    std::map<AbsentTips, Mesh> baseElements;
    if ( params.preserveTips )
    {
        auto absentTips = getAbsentTips( {1, 1, 1}, {2, 2, 2} );
        QME( baseElements[absentTips], makeBaseElement( params, absentTips ) )
    }
    else
    {
        // find axes of dims that are 1
        std::vector<int> flatAxes;
        for ( int i = 0; i < 3; ++i )
            if ( dims[i] == 1 )
                flatAxes.push_back( i );

        Vector3i baseDims{ 3, 3, 3 };
        for ( int fax : flatAxes )
            baseDims[fax] = 1;

        for ( int x = 0; x < baseDims.x; ++x )
            for ( int y = 0; y < baseDims.y; ++y )
                for ( int z = 0; z < baseDims.z; ++z )
                {
                    auto absentTips = getAbsentTips( {x, y, z}, baseDims - Vector3i::diagonal( 1 ) );
                    QME( baseElements[absentTips], makeBaseElement( params, absentTips ) )
                }

        assert( ( flatAxes.size() == 0 && baseElements.size() == 27 ) ||
                ( flatAxes.size() == 1 && baseElements.size() == 9 )  ||
                ( flatAxes.size() == 2 && baseElements.size() == 3 ) ||
                ( flatAxes.size() == 3 && baseElements.size() == 1 ) );
    }

    if ( !reportProgress( cb, 0.2f ) )
        return unexpectedOperationCanceled();

    auto sp = subprogress( cb, 0.2f, 1.f );
    Mesh result;
    for ( int x = 0; x < dims.x; ++x )
    {
        for ( int y = 0; y < dims.y; ++y )
        {
            for ( int z = 0; z < dims.z; ++z )
            {
                auto key = getAbsentTipsIfNeeded( {x, y, z}, dims - Vector3i::diagonal( 1 ) );
                assert( baseElements.contains( key ) );
                auto mesh = baseElements[key];
                mesh.transform( AffineXf3f::translation( mult( Vector3f( (float)x, (float)y, (float)z ), params.period ) ) );
                result.addMesh( mesh, {}, true );
            }
        }
        if ( !reportProgress( sp, (float)x * params.period.x / float( size.x ) ) )
            return unexpectedOperationCanceled();
    }

    MeshBuilder::uniteCloseVertices( result, decimateEps );
    if ( MeshComponents::getNumComponents( result ) != 1 )
        return unexpected( "Failed to unify result" );

    return result;
}

Expected<Mesh> fill( const Mesh& mesh, const Params& params, const ProgressCallback& cb )
{
    auto [size, xf] = getFillingSizeAndXf( mesh, std::max( params.period.x, std::max( params.period.y, params.period.z ) ) );
    auto filling = build( size, params, subprogress( cb, 0.f, 0.2f ) );
    if ( !filling )
        return filling;

    auto res = boolean( mesh, *filling, BooleanOperation::Union, &xf, {}, subprogress( cb, 0.2f, 1.f ) );
    if ( !res )
        return unexpected( res.errorString );
    return *res;
}

float estimateDensity( float T, float width, float R )
{
    const auto cr = width / 2.f;

    // usefull link: https://en.wikipedia.org/wiki/Steinmetz_solid
    auto Vbase = [cr] ( float T )
    {
        return cr*cr * ( 3.f * PI_F * T - 8.f*std::sqrt( 2.f )*cr );
    };

    if ( R <= std::sqrt( 3.f ) * cr )
    {
        return Vbase( T ) / ( T*T*T );
    }
    else
    {
        const auto t = std::sqrt( R*R - cr*cr );
        const auto third = 1.f / 3.f;
        const auto Vhat = PI_F*( 2.f*third*R*R*R - t*( 2*R*R + cr*cr )*third );
        const auto Vsphere = 4.f*third*PI_F*R*R*R;

        const auto internalBase = Vbase( 2*t );
        const auto fullBase = Vbase( T );

        return ( Vsphere - internalBase - 6.f*Vhat + fullBase ) / ( T*T*T );
    }
}

std::optional<float> estimateWidth( float T, float R, float d )
{
    // first guess R <= std::sqrt( 3.f ) * cr
    Polynomial<float, 3> p1{ { T*T*T*d, 0, -3.f*PI_F*T, 8.f*std::sqrt( 2.f ) } };
    float sol = -1.f;
    for ( float x : p1.solve( 1e-3f ) )
    {
        if ( x > 0 && 2.f*x < T && R <= std::sqrt( 3.f ) * x )
        {
            sol = x;
            break;
        }
    }
    if ( sol > 0 )
        return sol * 2.f;


    sol = -1.f;
    const auto alpha = d*T*T*T - (4.f / 3.f)*PI_F*R*R*R + 4.f*PI_F*R*R*R;
    const auto beta = -3.f*PI_F*T;
    p1 = Polynomial<float, 3>{ { sqr(alpha) - sqr(4*PI_F*R*R*R), 48.f*sqr(PI_F*R*R) + 2.f*alpha*beta, sqr(beta) - 48.f*sqr(PI_F*R), 16.f*sqr(PI_F) } };
    for ( float v : p1.solve( 1e-3f ) )
    {
        if ( v < 0 )
            continue;
        auto x = std::sqrt(v);
        if ( 2.f*x < T && R > std::sqrt( 3.f ) * x )
        {
            sol = 2.f*x;
            break;
        }
    }
    return sol > 0 ? std::optional{ sol } : std::nullopt;
}

} // namespace CellularSurface

std::vector<std::string> getKindNames()
{
    return { "TPMS", "Cellular" };
}

Expected<Mesh> build( const Vector3f& size, ConstMeshParamsRef params, ProgressCallback cb )
{
    return std::visit( overloaded{
        [&size, &cb] ( const TPMS::MeshParams& params ) { return TPMS::build( size, params, cb ); },
        [&size, &cb] ( const CellularSurface::Params& params ) { return CellularSurface::build( size, params, cb ); }
    }, params );
}

Expected<Mesh> fill( const Mesh& mesh, ConstMeshParamsRef params, ProgressCallback cb )
{
    return std::visit( overloaded{
        [&mesh, &cb] ( const TPMS::MeshParams& params ) { return TPMS::fill( mesh, params, cb ); },
        [&mesh, &cb] ( const CellularSurface::Params& params ) { return CellularSurface::fill( mesh, params, cb ); }
    }, params );
}



} // namespace FillingSurface
