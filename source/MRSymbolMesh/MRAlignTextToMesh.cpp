#include "MRAlignTextToMesh.h"
#include "MRMesh/MRAlignContoursToMesh.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRMatrix2.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRTimer.h"

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
        return unexpected( "no curve provided" );

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

    auto pivotY = params.pivotY;
    if ( params.fontBasedSizeCalc )
    {
        float absYPivot =
            ( 1 - numLines ) * params.MaxGeneratedFontHeight * ( 1 - params.pivotY ) + params.MaxGeneratedFontHeight * params.pivotY;
        pivotY = ( absYPivot - bbox.min.y ) / diagonal.y;
    }

    return bendContoursAlongCurve( conts, {
        .pivotY = pivotY,
        .curve = curve,
        .extrusion = params.surfaceOffset
        } );
}

Expected<Mesh> bendTextAlongCurve( const CurvePoints& curve, const BendTextAlongCurveParams& params )
{
    MR_TIMER;
    if ( curve.empty() )
        return unexpected( "no curve provided" );
    assert( std::is_sorted( curve.begin(), curve.end(), []( const auto& a, const auto& b ) { return a.time < b.time; } ) );
    if ( curve.front().time > 0 || curve.back().time < 1 )
        return unexpected( "curve does not include [0,1] interval" );

    std::vector<float> len;
    len.reserve( curve.size() );
}

} //namespace MR
