#include "MRAlignTextToMesh.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRQuaternion.h"

namespace MR
{

Expected<Mesh>  alignTextToMesh( 
    const Mesh& mesh, const TextMeshAlignParams& params )
{
    auto meshOrError = createSymbolsMesh( params );

    if ( !meshOrError.has_value() )
    {
        return unexpected( std::move( meshOrError.error() ) );
    }

    auto& textMesh = meshOrError.value();

    auto bbox = textMesh.computeBoundingBox();
    if ( !bbox.valid() )
        return unexpected( "Symbols mesh is empty" );

    int numLines = 1;
    for ( auto c : params.text )
        if ( c == '\n' )
            ++numLines;
    auto diagonal = bbox.size();
    AffineXf3f transform;

    const auto& vecx = params.direction.normalized();
    const auto norm = params.textNormal != nullptr ? *params.textNormal : mesh.pseudonormal( params.startPoint );
    const auto vecy = cross( vecx, -norm ).normalized();

    const Vector3f pivotCoord{ bbox.min.x + diagonal.x * params.pivotPoint.x,
                               params.fontBasedSizeCalc ? ( 1 - numLines ) * params.MaxGeneratedFontHeight * ( 1 - params.pivotPoint.y ) + params.MaxGeneratedFontHeight * params.pivotPoint.y :
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

    const float symbolDependentMultiplier = params.fontBasedSizeCalc ? diagonal.y / params.MaxGeneratedFontHeight / numLines : 1.0f;
    float scale = symbolDependentMultiplier * ( params.fontHeight * numLines * ( 1.0f + params.symbolsDistanceAdditionalOffset.y ) ) / diagonal.y;
    auto translation = mesh.triPoint( params.startPoint );

    transform = 
        AffineXf3f::translation( translation ) *
        AffineXf3f::linear( Matrix3f::scale( scale ) ) * rot
        * AffineXf3f::translation( -pivotCoord );

    auto& textMeshPoints = textMesh.points;
    for ( auto& p : textMeshPoints )
        p = transform( p );
    
    auto plusOffsetDir = norm * params.surfaceOffset;
    auto minusOffsetDir = norm * ( diagonal.z*scale - params.surfaceOffset );
    const auto maxMovement = std::max( 0.0f, params.textMaximumMovement );
    for ( int i = 0; i < textMeshPoints.size() / 2; ++i )
    {
        PointOnFace hit;
        auto inter = rayMeshIntersect( mesh, Line3f{ textMeshPoints[VertId( i )] + norm * bbox.size().y, -norm});
        if ( !inter )
            return unexpected( std::string( "Cannot align text" ) );
        hit = inter.proj;

        auto coords = hit.point;
        auto dir = coords - textMeshPoints[VertId( i )];
        auto movement = dir.length();
        if ( movement > maxMovement )
            dir = ( maxMovement / movement ) * dir;

        textMeshPoints[VertId( i )] += dir + plusOffsetDir;
        textMeshPoints[VertId( i + textMeshPoints.size() / 2 )] += dir + minusOffsetDir;
    }
    return std::move( textMesh );
}

}
