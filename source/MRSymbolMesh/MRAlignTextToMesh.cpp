#include "MRAlignTextToMesh.h"
#include "MRMesh/MRAlignContoursToMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRMatrix2.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRTimer.h"
#include <algorithm>

namespace MR
{

namespace
{

Box2f findContoursBBox( const Contours2f& conts )
{
    MR_TIMER;
    tbb::enumerable_thread_specific<Box2f> tls;
    ParallelFor( conts, [&] ( size_t i )
    {
        ParallelFor( conts[i], [&] ( size_t j )
        {
            tls.local().include( conts[i][j] );
        } );
    } );
    Box2f bbox;
    for ( const auto& tlsBox : tls )
        bbox.include( tlsBox );
    return bbox;
}

void scaleContours( Contours2f& conts, float scale )
{
    MR_TIMER;
    ParallelFor( conts, [&] ( size_t i )
    {
        ParallelFor( conts[i], [&] ( size_t j )
        {
            conts[i][j] = scale * conts[i][j];
        } );
    } );
}

} //anonymous namespace

Expected<Mesh> alignTextToMesh(
    const Mesh& mesh, const TextMeshAlignParams& params )
{
    MR_TIMER;
    auto contoursOrError = createSymbolContours( params );
    if ( !contoursOrError.has_value() )
        return unexpected( std::move( contoursOrError.error() ) );

    auto& conts = *contoursOrError;

    auto bbox = findContoursBBox( conts );
    if ( !bbox.valid() )
        return unexpected( "Symbols mesh is empty" );

    int numLines = 1;
    for ( auto c : params.text )
        if ( c == '\n' )
            ++numLines;

    auto diagonal = bbox.size();

    const float symbolDependentMultiplier = params.fontBasedSizeCalc ? diagonal.y / params.MaxGeneratedFontHeight / numLines : 1.0f;
    auto scale = symbolDependentMultiplier * ( params.fontHeight * numLines * ( 1.0f + params.symbolsDistanceAdditionalOffset.y ) ) / diagonal.y;
    scaleContours( conts, scale );

    Vector2f relPivot = params.pivotPoint;
    if ( params.fontBasedSizeCalc )
    {
        float absYPivot =
            ( 1 - numLines ) * params.MaxGeneratedFontHeight * ( 1 - params.pivotPoint.y ) + params.MaxGeneratedFontHeight * params.pivotPoint.y;
        relPivot.y = ( absYPivot - bbox.min.y ) / diagonal.y;
    }

    return alignContoursToMesh( mesh, conts, {
        .meshPoint = params.startPoint, 
        .pivotPoint = relPivot, 
        .xDirection = params.direction, 
        .zDirection = params.textNormal, 
        .extrusion = params.surfaceOffset, 
        .maximumShift = params.textMaximumMovement
        } );
}

Expected<Mesh> bendTextAlongCurve( const CurveFunc& curve, const BendTextAlongCurveParams& params )
{
    MR_TIMER;
    if ( !curve )
    {
        assert( false );
        return unexpected( "No curve provided" );
    }

    auto contoursOrError = createSymbolContours( params.symbolMesh );
    if ( !contoursOrError.has_value() )
        return unexpected( std::move( contoursOrError.error() ) );

    auto& conts = *contoursOrError;

    auto bbox = findContoursBBox( conts );
    if ( !bbox.valid() )
        return unexpected( "Symbols mesh is empty" );

    int numLines = 1;
    for ( auto c : params.symbolMesh.text )
        if ( c == '\n' )
            ++numLines;

    auto diagonal = bbox.size();

    const float symbolDependentMultiplier = params.fontBasedSizeCalc ? diagonal.y / params.symbolMesh.MaxGeneratedFontHeight / numLines : 1.0f;
    auto scale = symbolDependentMultiplier * ( params.fontHeight * numLines * ( 1.0f + params.symbolMesh.symbolsDistanceAdditionalOffset.y ) ) / diagonal.y;
    scaleContours( conts, scale );

    Vector2f relPivot = params.pivotBoxPoint;
    if ( params.fontBasedSizeCalc )
    {
        float absYPivot =
            ( 1 - numLines ) * params.symbolMesh.MaxGeneratedFontHeight * ( 1 - params.pivotBoxPoint.y ) + params.symbolMesh.MaxGeneratedFontHeight * params.pivotBoxPoint.y;
        relPivot.y = ( absYPivot - bbox.min.y ) / diagonal.y;
    }

    return bendContoursAlongCurve( conts, {
        .pivotCurveTime = params.pivotCurveTime,
        .pivotBoxPoint = relPivot,
        .curve = curve,
        .periodicCurve = params.periodicCurve,
        .stretch = params.stretch,
        .extrusion = params.surfaceOffset
        } );
}

Expected<Mesh> bendTextAlongCurve( const CurvePoints& cp, const BendTextAlongCurveParams& params0 )
{
    MR_TIMER;

    float curveLen = 0;
    auto maybeCurveFunc = curveFromPoints( cp, params0.stretch, &curveLen );
    if ( !maybeCurveFunc )
        return unexpected( std::move( maybeCurveFunc.error() ) );
    assert( curveLen > 0 );

    BendTextAlongCurveParams params = params0;
    if ( !params.stretch )
    {
        // pivotCurveTime from relative [0,1] into actual [0,len]
        params.pivotCurveTime *= curveLen;
    }

    return bendTextAlongCurve( *maybeCurveFunc, params );
}

Expected<Mesh> bendTextAlongSurfacePath( const Mesh& mesh,
    const MeshTriPoint & start, const SurfacePath& path, const MeshTriPoint & end, const BendTextAlongCurveParams& params )
{
    MR_TIMER;
    CurvePoints curve;
    curve.reserve( path.size() + 2 );
    curve.push_back( { .pos = mesh.triPoint( start ), .snorm = mesh.normal( start ) } );
    for ( const auto & ep : path )
        curve.push_back( { .pos = mesh.triPoint( ep ), .snorm = mesh.normal( ep ) } );
    curve.push_back( { .pos = mesh.triPoint( end ), .snorm = mesh.normal( end ) } );
    assert( curve.size() == path.size() + 2 );

    curve[0].dir = ( curve[1].pos - curve[0].pos ).normalized();
    for ( int i = 1; i + 1 < curve.size(); ++i )
        curve[i].dir = ( curve[i + 1].pos - curve[i - 1].pos ).normalized();
    curve.back().dir = ( curve[curve.size() - 1].pos - curve[curve.size() - 2].pos ).normalized();

    return bendTextAlongCurve( curve, params );
}

Expected<Mesh> bendTextAlongSurfacePath( const Mesh& mesh,
    const SurfacePath& path, const BendTextAlongCurveParams& params )
{
    MR_TIMER;
    if ( path.size() < 2 )
        return unexpected( "too short path" );
    CurvePoints curve;
    curve.reserve( path.size() );
    for ( const auto & ep : path )
        curve.push_back( { .pos = mesh.triPoint( ep ), .snorm = mesh.normal( ep ) } );
    assert( curve.size() == path.size() );

    curve[0].dir = ( curve[1].pos - curve[0].pos ).normalized();
    for ( int i = 1; i + 1 < curve.size(); ++i )
        curve[i].dir = ( curve[i + 1].pos - curve[i - 1].pos ).normalized();
    curve.back().dir = ( curve[curve.size() - 1].pos - curve[curve.size() - 2].pos ).normalized();

    return bendTextAlongCurve( curve, params );
}

} //namespace MR
