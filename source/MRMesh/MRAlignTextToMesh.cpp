#ifndef MRMESH_NO_LABEL
#include "MRAlignTextToMesh.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRAffineXf3.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRQuaternion.h"

namespace MR
{

tl::expected<Mesh, std::string>  alignTextToMesh( 
    const Mesh& mesh, const TextMeshAlignParams& params )
{
    Mesh textMesh = createSymbolsMesh( params );

    auto bbox = textMesh.computeBoundingBox();
    if ( !bbox.valid() )
        return tl::make_unexpected( "Symbols mesh is empty" );

    auto diagonal = bbox.size();
    const auto diagonalLength = diagonal.length();
    AffineXf3f transform;

    const auto& vecx = params.direction.normalized();
    const auto norm = params.textNormal != nullptr ? *params.textNormal : mesh.pseudonormal( params.startPoint );
    const auto vecy = cross( vecx, -norm ).normalized();

    const Vector3f pivotCoord{ bbox.min.x + diagonal.x * params.pivotPoint.x,
                               bbox.min.y + diagonal.y * params.pivotPoint.y,
                               0.0f };

    auto rot1Q = Quaternionf( Vector3f::plusX(), vecx );
    // handle degenerated case
    auto newY = rot1Q( Vector3f::plusY() );
    auto dotY = dot( newY, vecy );
    if ( std::abs( std::abs( dotY ) - 1.0f ) < 10.0f * std::numeric_limits<float>::epsilon() )
    {
        if ( dotY < 0.0f )
            newY = -vecy;
        else
            newY = vecy;
    }
    auto rotQ = Quaternionf( newY, vecy ) * rot1Q;
    AffineXf3f rot = AffineXf3f::linear( rotQ );
    float scale = params.fontHeight / diagonal.y;
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
        auto inter = rayMeshIntersect( mesh, Line3f{ textMeshPoints[VertId( i )] + norm * diagonalLength, -norm } );
        if ( !inter )
            return tl::make_unexpected( std::string( "Cannot align text" ) );
        hit = inter->proj;

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
#endif
