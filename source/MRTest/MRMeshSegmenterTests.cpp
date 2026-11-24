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

    // never merge over sharp edges
    struct Policy
    {
        const Mesh& mesh;

        struct SegmentData
        {
            int numFaces = 0;
            SegmentData& operator +=( const SegmentData& r ) { numFaces += r.numFaces; return *this; }
        };
        struct SegmentBdData
        {
            bool sharp = false;
            SegmentBdData& operator +=( const SegmentBdData& r ) { sharp = sharp || r.sharp; return *this; }
        };

        SegmentData fromTri( FaceId ) const { return { .numFaces = 1 }; }
        SegmentBdData fromUndirectedEdge( UndirectedEdgeId ue ) const { return { .sharp = std::abs( mesh.dihedralAngle( ue ) ) > 0.1f }; }
        double mergePenalty( const SegmentData& a, const SegmentData& b, const SegmentBdData& bd ) const
        {
            return bd.sharp ? prohibitMergePenalty : double( a.numFaces + b.numFaces );
        }
    };

    const auto order = MeshSegmenter<Policy>::run( cube.topology, Policy{ cube } );
    ASSERT_TRUE( order.has_value() );
    EXPECT_EQ( order->size(), 6 );
    EXPECT_EQ( findSegmentBoundaries( cube.topology, *order, 6 ).count(), 12 );
}

} //namespace MR
