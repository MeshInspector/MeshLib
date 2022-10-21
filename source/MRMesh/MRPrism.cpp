#include "MRPrism.h"
#include "MRMesh.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"
#include "MRVector2.h"

namespace MR
{
Mesh makePrism( float c, float alp, float bet, float height )
{
    const float gam = PI_F - alp - bet;
    const float b = c * sin( bet ) / sin( gam );
    const float a = c * sin( alp ) / sin( gam );

    const float ah = ( b * b - a * a + c * c ) / ( 2 * c );
    const float ch = sqrt( b * b - ah * ah );

    Vector2f points[3] =
    {
        { -c / 2 , 0 },
        { ah - c / 2, ch },
        { c / 2, 0}
    };
    // all triangles (8)
    Triangulation t
    {
        { VertId{0}, VertId{1}, VertId{2} },
        { VertId{3}, VertId{5}, VertId{4} },
        { VertId{0}, VertId{3}, VertId{1} },
        { VertId{1}, VertId{3}, VertId{4} },
        { VertId{1}, VertId{4}, VertId{5} },
        { VertId{1}, VertId{5}, VertId{2} },
        { VertId{0}, VertId{2}, VertId{5} },
        { VertId{0}, VertId{5}, VertId{3} }
    };

    Mesh meshObj;
    meshObj.topology = MeshBuilder::fromTriangles( t );
    meshObj.points.reserve( 6 );
    meshObj.points.emplace_back( points[0].x, points[0].y, -height * 0.5f ); // VertId{0}
    meshObj.points.emplace_back( points[1].x, points[1].y, -height * 0.5f ); // VertId{1}
    meshObj.points.emplace_back( points[2].x, points[2].y, -height * 0.5f ); // VertId{2}
    meshObj.points.emplace_back( points[0].x, points[0].y, height * 0.5f ); // VertId{3}
    meshObj.points.emplace_back( points[1].x, points[1].y, height * 0.5f ); // VertId{4}
    meshObj.points.emplace_back( points[2].x, points[2].y, height * 0.5f ); // VertId{5}

    return meshObj;
}
}