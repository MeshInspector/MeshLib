#include "MRAlignContoursToMesh.h"
#include "MR2DContoursTriangulation.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRMeshMetrics.h"
#include "MRMeshFillHole.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRMeshIntersect.h"
#include "MRBox.h"
#include "MRMeshComponents.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include <cmath>

namespace MR
{

void addBaseToPlanarMesh( Mesh& mesh, float zOffset )
{
    MR_TIMER;
    mesh.pack(); // for some hard fonts with duplicated points (if triangulated contours have same points, duplicates are not used)
    // it's important to have all vertices valid:
    // first half is upper points of text and second half is lower points of text

    Mesh mesh2 = mesh;
    for ( auto& p : mesh2.points )
        p.z += zOffset;

    mesh2.topology.flipOrientation();

    mesh.addMesh( mesh2 );

    auto edges = mesh.topology.findHoleRepresentiveEdges();
    for ( int bi = 0; bi < edges.size() / 2; ++bi )
    {
        StitchHolesParams stitchParams;
        stitchParams.metric = getVerticalStitchMetric( mesh, Vector3f::plusZ() );
        buildCylinderBetweenTwoHoles( mesh, edges[bi], edges[edges.size() / 2 + bi], stitchParams );
    }
}

Expected<Mesh> alignContoursToMesh( const Mesh& mesh, const Contours2f& contours, const ContoursMeshAlignParams& params )
{
    MR_TIMER;
    auto contoursMesh = PlanarTriangulation::triangulateContours( contours );
    auto bbox = contoursMesh.computeBoundingBox();
    if ( !bbox.valid() )
        return unexpected( "Contours mesh is empty" );

    const float cStartDepth = bbox.diagonal() * 0.05f; // use relative depth to avoid floating errors
    addBaseToPlanarMesh( contoursMesh, -cStartDepth );

    auto diagonal = bbox.size(); diagonal.z = cStartDepth;
    AffineXf3f transform;

    const auto& vecx = params.xDirection.normalized();
    const auto norm = params.zDirection != nullptr ? *params.zDirection : mesh.pseudonormal( params.meshPoint );
    const auto vecy = cross( vecx, -norm ).normalized();

    const Vector3f pivotCoord{ bbox.min.x + diagonal.x * params.pivotPoint.x,
                               bbox.min.y + diagonal.y * params.pivotPoint.y,
                               0.0f };

    auto rotQ = Quaternionf( Vector3f::plusX(), vecx );
    // handle degenerated case
    auto newY = rotQ( Vector3f::plusY() );
    auto dotY = dot( newY, vecy );
    if ( std::abs( std::abs( dotY ) - 1.0f ) < 10.0f * std::numeric_limits<float>::epsilon() )
    {
        if ( dotY < 0.0f )
            rotQ = Quaternionf( vecx, PI_F ) * rotQ;
    }
    else
        rotQ = Quaternionf( newY, vecy ) * rotQ;
    AffineXf3f rot = AffineXf3f::linear( rotQ );

    auto translation = mesh.triPoint( params.meshPoint );

    transform =
        AffineXf3f::translation( translation ) *
        rot
        * AffineXf3f::translation( -pivotCoord );

    auto& contoursMeshPoints = contoursMesh.points;
    for ( auto& p : contoursMeshPoints )
        p = transform( p );

    auto plusOffsetDir = norm * std::abs( params.extrusion );
    auto minusOffsetDir = norm * ( cStartDepth - std::abs( params.extrusion ) );
    const auto maxMovement = std::max( 0.0f, params.maximumShift );
    for ( int i = 0; i < contoursMeshPoints.size() / 2; ++i )
    {
        PointOnFace hit;
        auto inter = rayMeshIntersect( mesh, Line3f{ contoursMeshPoints[VertId( i )] + norm * bbox.size().y, -norm } );
        if ( !inter )
            return unexpected( std::string( "Cannot align contours" ) );
        hit = inter.proj;

        auto coords = hit.point;
        auto dir = coords - contoursMeshPoints[VertId( i )];
        auto movement = dir.length();
        if ( movement > maxMovement )
            dir = ( maxMovement / movement ) * dir;

        contoursMeshPoints[VertId( i )] += dir + plusOffsetDir;
        contoursMeshPoints[VertId( i + contoursMeshPoints.size() / 2 )] += dir + minusOffsetDir;
    }
    return contoursMesh;

}

Expected<Mesh> bendContoursAlongCurve( const Contours2f& contours, const CurveFunc& curve, const BendContoursAlongCurveParams& params )
{
    MR_TIMER;
    auto contoursMesh = PlanarTriangulation::triangulateContours( contours );
    auto bbox = contoursMesh.computeBoundingBox();
    if ( !bbox.valid() )
        return unexpected( "Contours mesh is empty" );

    const float cStartDepth = bbox.diagonal() * 0.05f; // use relative depth to avoid floating errors
    addBaseToPlanarMesh( contoursMesh, -cStartDepth );
    contoursMesh.invalidateCaches();
    auto diagonal = bbox.size();
    diagonal.z = cStartDepth;
    const float pivotInBoxX = lerp( bbox.min.x, bbox.max.x, params.pivotBoxPoint.x );
    const float pivotInBoxY = lerp( bbox.min.y, bbox.max.y, params.pivotBoxPoint.y );

    const float plusOffset = std::abs( params.extrusion );
    const float minusOffset = cStartDepth - std::abs( params.extrusion );

    auto& contoursMeshPoints = contoursMesh.points;
    VertId firstBottomVert( contoursMeshPoints.size() / 2 );

    const auto components = MeshComponents::getAllComponents( contoursMesh );
    const auto stretchMod = curve.totalLength / diagonal.x;
    const auto startCurvePos = params.pivotCurveTime * curve.totalLength;
    // independently for each component of contoursMesh
    ParallelFor( components, [&]( size_t icomp )
    {
        const auto & compFaces = components[icomp];
        const auto compCenter = contoursMesh.computeBoundingBox( &compFaces ).center();
        float xInBox = compCenter.x - pivotInBoxX;
        if ( params.stretch )
            xInBox *= stretchMod;

        float curveTime = startCurvePos + xInBox;
        if ( params.periodicCurve )
            curveTime = std::fmodf( curveTime, curve.totalLength );
        const auto pos = curve.func( curveTime );

        const auto vecx = pos.dir;
        const auto norm = pos.snorm;
        const auto vecy = cross( vecx, -norm ).normalized();

        auto rotQ = Quaternionf( Vector3f::plusX(), vecx );
        // handle degenerated case
        auto newY = rotQ( Vector3f::plusY() );
        auto dotY = dot( newY, vecy );
        if ( std::abs( std::abs( dotY ) - 1.0f ) < 10.0f * std::numeric_limits<float>::epsilon() )
        {
            if ( dotY < 0.0f )
                rotQ = Quaternionf( vecx, PI_F ) * rotQ;
        }
        else
            rotQ = Quaternionf( newY, vecy ) * rotQ;
        AffineXf3f rot = AffineXf3f::linear( rotQ );

        const AffineXf3f transformTop =
            AffineXf3f::translation( pos.pos ) *
            rot
            * AffineXf3f::translation( Vector3f{ -compCenter.x, -pivotInBoxY, plusOffset } );

        const AffineXf3f transformBottom =
            AffineXf3f::translation( pos.pos ) *
            rot
            * AffineXf3f::translation( Vector3f{ -compCenter.x, -pivotInBoxY, minusOffset } );

        for ( auto v : getIncidentVerts( contoursMesh.topology, compFaces ) )
        {
            if ( v < firstBottomVert )
                contoursMeshPoints[v] = transformTop( contoursMeshPoints[v] );
            else
                contoursMeshPoints[v] = transformBottom( contoursMeshPoints[v] );
        }
    } );
    return contoursMesh;
}

Expected<Mesh> bendContoursAlongSurfacePath( const Contours2f& contours, const Mesh& mesh, const MeshTriPoint & start, const SurfacePath& path, const MeshTriPoint & end,
    const BendContoursAlongCurveParams& params )
{
    MR_TIMER;
    return curveFromPoints( meshPathCurvePoints( mesh, start, path, end ) )
        .and_then( [&]( auto && curve ) { return bendContoursAlongCurve( contours, curve, params ); } );
}

Expected<Mesh> bendContoursAlongSurfacePath( const Contours2f& contours, const Mesh& mesh, const SurfacePath& path,
    const BendContoursAlongCurveParams& params )
{
    MR_TIMER;
    return curveFromPoints( meshPathCurvePoints( mesh, path ) )
        .and_then( [&]( auto && curve ) { return bendContoursAlongCurve( contours, curve, params ); } );
}

Expected<std::vector<float>> findPartialLens( const CurvePoints& cp, float * outCurveLen )
{
    MR_TIMER;
    if ( cp.size() < 2 )
    {
        assert( false );
        return unexpected( "Curve is too short" );
    }

    std::vector<float> lens;
    lens.reserve( cp.size() );
    lens.push_back( 0 );
    for ( int i = 0; i + 1 < cp.size(); ++i )
        lens.push_back( lens.back() + distance( cp[i].pos, cp[i+1].pos ) );
    assert( lens.size() == cp.size() );
    if ( outCurveLen )
        *outCurveLen = lens.back();
    if ( lens.back() <= 0 )
        return unexpected( "curve has zero length" );

    return lens;
}

CurvePoint getCurvePoint( const CurvePoints& cp, const std::vector<float> & lens, float p )
{
    assert( cp.size() == lens.size() );
    assert( cp.size() >= 2 );
    CurvePoint res;
    if ( p <= lens.front() )
    {
        // extrapolate
        res = cp.front();
        res.pos += ( p - lens.front() ) * res.dir;
        return res;
    }
    if ( p >= lens.back() )
    {
        // extrapolate
        res = cp.back();
        res.pos += ( p - lens.back() ) * res.dir;
        return res;
    }
    // interpolate
    auto i = std::lower_bound( lens.begin(), lens.end(), p ) - lens.begin();
    assert( lens[i] >= p );
    if ( lens[i] == p )
        return cp[i];
    assert( lens[i-1] < p );
    auto f = ( p - lens[i-1] ) / ( lens[i] - lens[i-1] );
    res = CurvePoint
    {
        .pos = lerp( cp[i-1].pos, cp[i].pos, f ),
        .dir = lerp( cp[i-1].dir, cp[i].dir, f ),
        .snorm = lerp( cp[i-1].snorm, cp[i].snorm, f )
    };
    return res;
}

Expected<CurveFunc> curveFromPoints( const CurvePoints& cp, float* outCurveLen )
{
    MR_TIMER;
    auto maybeLens = findPartialLens( cp, outCurveLen );
    if ( !maybeLens )
        return unexpected( std::move( maybeLens.error() ) );

    CurveFunc res;
    res.totalLength = maybeLens->back();
    res.func = [&cp, lens = std::move( *maybeLens )] ( float p )
    {
        return getCurvePoint( cp, lens, p );
    };
    return res;
}

Expected<CurveFunc> curveFromPoints( CurvePoints&& cp, float * outCurveLen )
{
    MR_TIMER;
    auto maybeLens = findPartialLens( cp, outCurveLen );
    if ( !maybeLens )
        return unexpected( std::move( maybeLens.error() ) );

    CurveFunc res;
    res.totalLength = maybeLens->back();
    res.func = [&cp, lens = std::move( *maybeLens )] ( float p )
    {
        return getCurvePoint( cp, lens, p );
    };
    return res;
}

CurvePoints meshPathCurvePoints( const Mesh& mesh, const MeshTriPoint & start, const SurfacePath& path, const MeshTriPoint & end )
{
    MR_TIMER;
    CurvePoints cp;
    cp.reserve( path.size() + 2 );
    cp.push_back( { .pos = mesh.triPoint( start ), .snorm = mesh.normal( start ) } );
    for ( const auto & ep : path )
        cp.push_back( { .pos = mesh.triPoint( ep ), .snorm = mesh.normal( ep ) } );
    cp.push_back( { .pos = mesh.triPoint( end ), .snorm = mesh.normal( end ) } );
    assert( cp.size() == path.size() + 2 );

    cp[0].dir = ( cp[1].pos - cp[0].pos ).normalized();
    for ( int i = 1; i + 1 < cp.size(); ++i )
        cp[i].dir = ( cp[i + 1].pos - cp[i - 1].pos ).normalized();
    cp.back().dir = ( cp[cp.size() - 1].pos - cp[cp.size() - 2].pos ).normalized();
    return cp;
}

CurvePoints meshPathCurvePoints( const Mesh& mesh, const SurfacePath& path )
{
    MR_TIMER;
    CurvePoints cp;
    cp.reserve( path.size() );
    for ( const auto & ep : path )
        cp.push_back( { .pos = mesh.triPoint( ep ), .snorm = mesh.normal( ep ) } );
    assert( cp.size() == path.size() );

    cp[0].dir = ( cp[1].pos - cp[0].pos ).normalized();
    for ( int i = 1; i + 1 < cp.size(); ++i )
        cp[i].dir = ( cp[i + 1].pos - cp[i - 1].pos ).normalized();
    cp.back().dir = ( cp[cp.size() - 1].pos - cp[cp.size() - 2].pos ).normalized();
    return cp;
}

} //namespace MR
