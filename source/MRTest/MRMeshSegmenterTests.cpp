#include <MRMesh/MRMeshSegmenter.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MREdgeMetric.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, SegmentMesh )
{
    const Mesh cube = makeCube();
    const auto order = segmentMesh( cube, edgeDihedralAngleMetric( cube ) );
    ASSERT_TRUE( order.has_value() );
    EXPECT_EQ( order->size(), 11 );
    // first the two triangles of every cube side are merged
    EXPECT_EQ( findSegmentBoundaries( cube.topology, *order, 6 ).count(), 12 );
}

TEST( MRMesh, MeshSegmenter )
{
    const Mesh cube = makeCube();

    struct SegmentData
    {
        int numFaces = 0;
        SegmentData& operator +=( const SegmentData& r ) { numFaces += r.numFaces; return *this; }
        static SegmentData fromTri( const Mesh&, FaceId ) { return { .numFaces = 1 }; }
    };
    struct SegmentBdData
    {
        bool sharp = false;
        SegmentBdData& operator +=( const SegmentBdData& r ) { sharp = sharp || r.sharp; return *this; }
        static SegmentBdData fromUndirectedEdge( const Mesh& mesh, const EdgeMetric&, UndirectedEdgeId ue )
        {
            return { .sharp = std::abs( mesh.dihedralAngle( ue ) ) > 0.1f };
        }
    };
    // never merge over sharp edges
    auto mergePenalty = []( const SegmentData& a, const SegmentData& b, const SegmentBdData& bd )
    {
        return bd.sharp ? prohibitMergePenalty : double( a.numFaces + b.numFaces );
    };

    const auto order = MeshSegmenter<SegmentData, SegmentBdData, decltype( mergePenalty )>::run( cube, {}, mergePenalty );
    ASSERT_TRUE( order.has_value() );
    EXPECT_EQ( order->size(), 6 );
    EXPECT_EQ( findSegmentBoundaries( cube.topology, *order, 6 ).count(), 12 );
}

} //namespace MR
