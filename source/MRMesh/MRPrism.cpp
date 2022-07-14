#include "MRPrism.h"
#include "MRMesh.h"
#include "MRConstants.h"
#include "MRMeshBuilder.h"
#include "MRVector2.h"

namespace MR
{
    Mesh makePrism( const std::array<MR::Vector2f, 3>& points, float height )
    {
		// all triangles (8)
		std::vector<VertId> v
		{
		VertId{0}, VertId{1}, VertId{2},
		VertId{3}, VertId{5}, VertId{4},
		VertId{0}, VertId{3}, VertId{1},
		VertId{1}, VertId{3}, VertId{4},
		VertId{1}, VertId{4}, VertId{5},
		VertId{1}, VertId{5}, VertId{2},
		VertId{0}, VertId{2}, VertId{5},
		VertId{0}, VertId{5}, VertId{3}
		};

		Mesh meshObj;
		meshObj.topology = MeshBuilder::fromVertexTriples( v );
		meshObj.points.emplace_back( points[0].x, points[0].y, -height * 0.5f ); // VertId{0}
		meshObj.points.emplace_back( points[1].x, points[1].y, -height * 0.5f ); // VertId{1}
		meshObj.points.emplace_back( points[2].x, points[2].y, -height * 0.5f ); // VertId{2}
		meshObj.points.emplace_back( points[0].x, points[0].y, height * 0.5f ); // VertId{3}
		meshObj.points.emplace_back( points[1].x, points[1].y, height * 0.5f ); // VertId{4}
		meshObj.points.emplace_back( points[2].x, points[2].y, height * 0.5f ); // VertId{5}

		return meshObj;
    }
}