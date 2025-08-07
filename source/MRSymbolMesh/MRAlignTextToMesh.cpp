#include "MRAlignTextToMesh.h"
#include "MRMesh/MRAlignContoursToMesh.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRMatrix2.h"
#include "MRMesh/MRParallelFor.h"

namespace MR
{

Expected<Mesh> alignTextToMesh( 
    const Mesh& mesh, const TextMeshAlignParams& params )
{
    auto contoursOrError = createSymbolContours( params );
    if ( !contoursOrError.has_value() )
    {
        return unexpected( std::move( contoursOrError.error() ) );
    }

    auto& conts = *contoursOrError;

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

    if ( !bbox.valid() )
        return unexpected( "Symbols mesh is empty" );

    int numLines = 1;
    for ( auto c : params.text )
        if ( c == '\n' )
            ++numLines;

    auto diagonal = bbox.size();

    const float symbolDependentMultiplier = params.fontBasedSizeCalc ? diagonal.y / params.MaxGeneratedFontHeight / numLines : 1.0f;
    auto scaleXf = Matrix2f::scale( symbolDependentMultiplier * ( params.fontHeight * numLines * ( 1.0f + params.symbolsDistanceAdditionalOffset.y ) ) / diagonal.y );

    ParallelFor( conts, [&] ( size_t i )
    {
        ParallelFor( conts[i], [&] ( size_t j )
        {
            conts[i][j] = scaleXf * conts[i][j];
        } );
    } );

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

}
