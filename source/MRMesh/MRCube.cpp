#include "MRCube.h"
#include "MRMesh.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"
#include "MRBox.h"

namespace MR
{

MeshTopology makeCubeTopology()
{
    // all triangles (12)
    Triangulation t{
        { VertId{0}, VertId{1}, VertId{2} },
        { VertId{2}, VertId{3}, VertId{0} },
        { VertId{0}, VertId{4}, VertId{5} },
        { VertId{5}, VertId{1}, VertId{0} },
        { VertId{0}, VertId{3}, VertId{7} },
        { VertId{7}, VertId{4}, VertId{0} },
        { VertId{6}, VertId{5}, VertId{4} },
        { VertId{4}, VertId{7}, VertId{6} },
        { VertId{1}, VertId{5}, VertId{6} },
        { VertId{6}, VertId{2}, VertId{1} },
        { VertId{6}, VertId{7}, VertId{3} },
        { VertId{3}, VertId{2}, VertId{6} }
    };
    return MeshBuilder::fromTriangles( t );
}

Mesh makeCube( const Vector3f& size, const Vector3f& base)
{
    Mesh res;
    res.topology = makeCubeTopology();
    res.points.reserve( 8 );
    res.points.emplace_back( base.x, base.y, base.z ); // VertId{0}
    res.points.emplace_back( base.x, base.y + size.y, base.z) ; // VertId{1}
    res.points.emplace_back( base.x + size.x, base.y + size.y, base.z ); // VertId{2}
    res.points.emplace_back( base.x + size.x, base.y, base.z ); // VertId{3}
    res.points.emplace_back( base.x, base.y, base.z + size.z ); // VertId{4}
    res.points.emplace_back( base.x, base.y + size.y, base.z + size.z ); // VertId{5}
    res.points.emplace_back( base.x + size.x, base.y + size.y, base.z + size.z ); // VertId{6}
    res.points.emplace_back( base.x + size.x, base.y, base.z + size.z ); // VertId{7}

    return res;
}

Mesh makeParallelepiped(const Vector3f side[3], const Vector3f & corner)
{
    Mesh res;
    res.topology = makeCubeTopology();
    res.points.reserve( 8 );
    res.points.emplace_back( corner.x, corner.y, corner.z ); //base // VertId{0}
    res.points.emplace_back( corner.x + side[1].x, corner.y + side[1].y, corner.z + side[1].z ); //base+b // VertId{1}
    res.points.emplace_back( corner.x + side[0].x + side[1].x, corner.y + side[0].y + side[1].y, corner.z + side[0].z + side[1].z ); //+b+a // VertId{2}
    res.points.emplace_back( corner.x + side[0].x, corner.y + side[0].y, corner.z + side[0].z ); //base+a // VertId{3}
    res.points.emplace_back( corner.x + side[2].x, corner.y + side[2].y, corner.z + side[2].z ); //base+c // VertId{4}
    res.points.emplace_back( corner.x + side[1].x + side[2].x, corner.y + side[1].y + side[2].y, corner.z + side[1].z + side[2].z ); //base+b+c // VertId{5}
    res.points.emplace_back( corner.x + side[0].x + side[1].x + side[2].x, corner.y + side[0].y + side[1].y + side[2].y, corner.z + side[0].z + side[1].z + side[2].z ); //base+a+b+c // VertId{6}
    res.points.emplace_back( corner.x + side[0].x + side[2].x, corner.y + side[0].y + side[2].y, corner.z + side[0].z + side[2].z ); //base+a+c // VertId{7}

    return res;
}

Mesh makeBoxMesh( const Box3f& box )
{
    Mesh res;
    res.topology = makeCubeTopology();
    res.points.reserve( 8 );
    res.points.push_back( box.corner( { 0, 0, 0 } ) ); // VertId{0}
    res.points.push_back( box.corner( { 0, 1, 0 } ) ); // VertId{1}
    res.points.push_back( box.corner( { 1, 1, 0 } ) ); // VertId{2}
    res.points.push_back( box.corner( { 1, 0, 0 } ) ); // VertId{3}
    res.points.push_back( box.corner( { 0, 0, 1 } ) ); // VertId{4}
    res.points.push_back( box.corner( { 0, 1, 1 } ) ); // VertId{5}
    res.points.push_back( box.corner( { 1, 1, 1 } ) ); // VertId{6}
    res.points.push_back( box.corner( { 1, 0, 1 } ) ); // VertId{7}
    return res;
}

} // namespace MR
