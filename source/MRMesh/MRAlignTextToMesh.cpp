#ifndef MRMESH_NO_LABEL
#include "MRAlignTextToMesh.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRAffineXf3.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"

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

    const Vector3f pivotCoord{ bbox.min.x + (bbox.max.x - bbox.min.x) * params.pivotPoint.x,
                               bbox.min.y + (bbox.max.y - bbox.min.y) * params.pivotPoint.y,
                               0.0f };

    AffineXf3f rot1 = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusX(), vecx ) );
    AffineXf3f rot2 = AffineXf3f::linear( Matrix3f::rotation( rot1( Vector3f::plusY() ), vecy ) );
    float scale = params.fontHeight / diagonal.y;
    auto translation = mesh.triPoint( params.startPoint );

    transform = 
        AffineXf3f::translation( translation ) * rot2 * rot1 
        * AffineXf3f::linear( Matrix3f::scale( scale ) )
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
