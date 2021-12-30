#include "MRCube.h"
#include "MRMesh.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"

namespace MR
{

MR::Mesh makeCube( const Vector3f& size, const Vector3f& base)
{
	// all triangles (12)
	std::vector<VertId> v{
	VertId{0}, VertId{1}, VertId{2},
	VertId{2}, VertId{3}, VertId{0},
	VertId{0}, VertId{4}, VertId{5},
	VertId{5}, VertId{1}, VertId{0},
	VertId{0}, VertId{3}, VertId{7},
	VertId{7}, VertId{4}, VertId{0},
	VertId{6}, VertId{5}, VertId{4},
	VertId{4}, VertId{7}, VertId{6},
	VertId{1}, VertId{5}, VertId{6},
	VertId{6}, VertId{2}, VertId{1},
	VertId{6}, VertId{7}, VertId{3},
	VertId{3}, VertId{2}, VertId{6}
	};

	// create object Mesh cube
	Mesh meshObj;
	meshObj.topology = MeshBuilder::fromVertexTriples(v);
	meshObj.points.emplace_back(base.x, base.y, base.z); // VertId{0}
	meshObj.points.emplace_back(base.x, base.y + size.y, base.z); // VertId{1}
	meshObj.points.emplace_back(base.x + size.x, base.y + size.y, base.z); // VertId{2}
	meshObj.points.emplace_back(base.x + size.x, base.y, base.z); // VertId{3}
	meshObj.points.emplace_back(base.x, base.y, base.z + size.z); // VertId{4}
	meshObj.points.emplace_back(base.x, base.y + size.y, base.z + size.z); // VertId{5}
	meshObj.points.emplace_back(base.x + size.x, base.y + size.y, base.z + size.z); // VertId{6}
	meshObj.points.emplace_back(base.x + size.x, base.y, base.z + size.z); // VertId{7}

    return meshObj;
}

Mesh makeParallelepiped(const Vector3f side[3], const Vector3f & corner) {
	// all triangles (12)
	std::vector<VertId> v{
	VertId{0}, VertId{1}, VertId{2},
	VertId{2}, VertId{3}, VertId{0},
	VertId{0}, VertId{4}, VertId{5},
	VertId{5}, VertId{1}, VertId{0},
	VertId{0}, VertId{3}, VertId{7},
	VertId{7}, VertId{4}, VertId{0},
	VertId{6}, VertId{5}, VertId{4},
	VertId{4}, VertId{7}, VertId{6},
	VertId{1}, VertId{5}, VertId{6},
	VertId{6}, VertId{2}, VertId{1},
	VertId{6}, VertId{7}, VertId{3},
	VertId{3}, VertId{2}, VertId{6}
	};

	// create object Mesh parallelepiped
	Mesh meshObj;
	meshObj.topology = MeshBuilder::fromVertexTriples(v);
	meshObj.points.emplace_back(corner.x, corner.y, corner.z); //base // VertId{0}
	meshObj.points.emplace_back(corner.x + side[1].x, corner.y + side[1].y, corner.z + side[1].z); //base+b // VertId{1}
	meshObj.points.emplace_back(corner.x + side[0].x + side[1].x, corner.y + side[0].y + side[1].y, corner.z + side[0].z + side[1].z); //+b+a // VertId{2}
	meshObj.points.emplace_back(corner.x + side[0].x, corner.y + side[0].y, corner.z + side[0].z); //base+a // VertId{3}
	meshObj.points.emplace_back(corner.x + side[2].x, corner.y + side[2].y, corner.z + side[2].z); //base+c // VertId{4}
	meshObj.points.emplace_back(corner.x + side[1].x + side[2].x, corner.y + side[1].y + side[2].y, corner.z + side[1].z + side[2].z); //base+b+c // VertId{5}
	meshObj.points.emplace_back(corner.x + side[0].x + side[1].x + side[2].x, corner.y + side[0].y + side[1].y + side[2].y, corner.z + side[0].z + side[1].z + side[2].z); //base+a+b+c // VertId{6}
	meshObj.points.emplace_back(corner.x + side[0].x + side[2].x, corner.y + side[0].y + side[2].y, corner.z + side[0].z + side[2].z); //base+a+c // VertId{7}

	return meshObj;
}

}
