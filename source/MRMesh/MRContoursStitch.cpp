#include "MRContoursStitch.h"
#include "MRMesh.h"
#include "MREdgePaths.h"
#include "MRCube.h"
#include "MRRingIterator.h"
#include "MRGTest.h"

namespace MR
{

void stitchContours( MeshTopology & topology, const EdgePath & c0, const EdgePath & c1 )
{
    assert( c0.size() == c1.size() );
    const auto sz = c0.size();

    // delete vertices from c1, make multiple edges
    for ( size_t i = 0; i < sz; ++i )
    {
        auto e0 = c0[i];
        assert( !topology.left( e0 ) );
        auto e1 = c1[i];
        assert( !topology.right( e1 ) );
        assert( e0 != e1 );

        if ( topology.org( e0 ) != topology.org( e1 ) )
        {
            topology.setOrg( e1, VertId{} );
            topology.splice( e0, topology.prev( e1 ) );
        }
        if ( topology.org( e0.sym() ) != topology.org( e1.sym() ) )
        {
            topology.setOrg( e1.sym(), VertId{} );
            topology.splice( topology.prev( e0.sym() ), e1.sym() );
        }
        assert( topology.next( e0 ) == e1 );
        assert( topology.next( e1.sym() ) == e0.sym() );
    }

    // delete edges from c1
    for ( size_t i = 0; i < sz; ++i )
    {
        auto e0 = c0[i];
        auto e1 = c1[i];
        if ( topology.next( e0 ) == e1 )
            topology.splice( e0, e1 );
        if ( topology.next( e1.sym() ) == e0.sym() )
            topology.splice( topology.prev( e1.sym() ), e1.sym() );

        assert( topology.isLoneEdge( e1 ) );
    }
}

EdgeLoop cutAlongEdgeLoop( MeshTopology & topology, const EdgeLoop & c0 )
{
    EdgePath c1;
    if ( !isEdgeLoop( topology, c0 ) )
    {
        assert( false );
        return c1;
    }
    const auto sz = c0.size();
    c1.reserve( sz );

    EdgeId last0 = c0.back().sym();

    // introduce multiple edge for each edge from c0
    for ( size_t i = 0; i < sz; ++i )
    {
        const auto e0 = c0[i];
        const auto e1 = topology.makeEdge();
        c1.push_back( e1 );
        topology.splice( e0, e1 );
        topology.splice( topology.prev( e0.sym() ), e1.sym() );
        assert( topology.fromSameOriginRing( e0, e1 ) );
        assert( topology.fromSameOriginRing( e0.sym(), e1.sym() ) );
        assert( !topology.left( e0 ) );
        assert( !topology.right( e1 ) );
    }

    // split vertices
    for ( size_t i = 0; i < sz; ++i )
    {
        auto e0 = c0[i];
        assert( !topology.left( e0 ) );
        auto e1 = c1[i];
        assert( !topology.right( e1 ) );
        assert( e0 != e1 );

        assert( topology.fromSameOriginRing( e0, e1 ) );
        assert( topology.fromSameOriginRing( e0, last0 ) );
        topology.splice( e0, topology.prev( last0 ) );
        assert( !topology.fromSameOriginRing( e0, e1 ) );
        if ( topology.org( e0 ) )
            topology.setOrg( e1, topology.addVertId() );
        last0 = e0.sym();
    }

    assert( isEdgePath( topology, c1 ) );
    return c1;
}

EdgeLoop cutAlongEdgeLoop( Mesh& mesh, const EdgeLoop& c0 )
{
    const auto res = cutAlongEdgeLoop( mesh.topology, c0 );
    mesh.points.reserve( mesh.points.size() + res.size() );

    for ( size_t i = 0; i < c0.size(); ++i )
    {
        mesh.points.autoResizeSet( mesh.topology.org( res[i] ), mesh.orgPnt( c0[i] ) );
    }
    return res;
}

TEST(MRMesh, cutAlongEdgeLoop)
{
    Mesh mesh = makeCube();
    auto & topology = mesh.topology;
    const auto ueCntA = topology.computeNotLoneUndirectedEdges();

    EdgeLoop c0;
    for ( auto e : leftRing( mesh.topology, 0_f ) )
        c0.push_back( e );
    auto c1 = cutAlongEdgeLoop( mesh.topology, c0 );
    const auto ueCntB = topology.computeNotLoneUndirectedEdges();
    ASSERT_EQ( ueCntB, ueCntA + 3 );

    stitchContours( mesh.topology, c0, c1 );
    const auto ueCntC = topology.computeNotLoneUndirectedEdges();
    ASSERT_EQ( ueCntC, ueCntA );
}

} //namespace MR
