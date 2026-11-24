#include <MRMesh/MRMeshSegmenter.h>
#include <MRMesh/MRNarrowNeckSegmentPolicy.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMakePlane.h>
#include <MRMesh/MREdgeIterator.h>
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

        SegmentData segmentFromTri( FaceId ) const { return { .numFaces = 1 }; }
        SegmentBdData bdFromEdge( UndirectedEdgeId ue ) const { return { .sharp = std::abs( mesh.dihedralAngle( ue ) ) > 0.1f }; }
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

TEST( MRMesh, NarrowNeckSegmentPolicy )
{
    // without holes the result is the same as with SimpleSegmentPolicy
    const Mesh cube = makeCube();
    const auto curvMetric = edgeDihedralAngleMetric( cube );
    const auto simpleOrder = MeshSegmenter<SimpleSegmentPolicy>::run( cube.topology, { cube, curvMetric } );
    const auto narrowNeckOrder = MeshSegmenter<NarrowNeckSegmentPolicy>::run( cube.topology, { cube, curvMetric, 100 } );
    ASSERT_TRUE( simpleOrder.has_value() );
    ASSERT_TRUE( narrowNeckOrder.has_value() );
    EXPECT_EQ( *simpleOrder, *narrowNeckOrder );

    // two triangles, each with two boundary edges, sharing the diagonal
    const Mesh plane = makePlane();
    ASSERT_EQ( plane.topology.numValidFaces(), 2 );
    const EdgeMetric unitMetric = []( EdgeId ) { return 1.0f; };
    double diagonal = 0;
    for ( auto ue : undirectedEdges( plane.topology ) )
        if ( plane.topology.left( ue ) && plane.topology.right( ue ) )
            diagonal = plane.edgeLength( ue );
    ASSERT_GT( diagonal, 0 );

    const NarrowNeckSegmentPolicy policy{ plane, unitMetric };
    const auto d0 = policy.segmentFromTri( 0_f );
    EXPECT_NEAR( d0.holeBdLength, diagonal * std::sqrt( 2.0 ), 1e-6 );

    // the diagonal is not short: merge
    auto order = MeshSegmenter<NarrowNeckSegmentPolicy>::run( plane.topology, { plane, unitMetric, diagonal / 2 } );
    ASSERT_TRUE( order.has_value() );
    EXPECT_EQ( order->size(), 1 );

    // the diagonal is short, and shorter than the hole boundary of each triangle: no merge
    order = MeshSegmenter<NarrowNeckSegmentPolicy>::run( plane.topology, { plane, unitMetric, diagonal * 2 } );
    ASSERT_TRUE( order.has_value() );
    EXPECT_EQ( order->size(), 0 );

    auto largest = findLargestSegment( cube, curvMetric, 100 );
    ASSERT_TRUE( largest.has_value() );
    EXPECT_EQ( largest->count(), 12 );

    largest = findLargestSegment( plane, unitMetric, diagonal / 2 );
    ASSERT_TRUE( largest.has_value() );
    EXPECT_EQ( largest->count(), 2 );

    largest = findLargestSegment( plane, unitMetric, diagonal * 2 );
    ASSERT_TRUE( largest.has_value() );
    EXPECT_EQ( largest->count(), 1 );
}

} //namespace MR
