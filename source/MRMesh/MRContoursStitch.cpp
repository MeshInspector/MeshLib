#include "MRContoursStitch.h"
#include "MRMeshTopology.h"

namespace MR
{

void stitchContours( MeshTopology & topology, const std::vector<EdgeId> & c0, const std::vector<EdgeId> & c1 )
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

} //namespace MR
