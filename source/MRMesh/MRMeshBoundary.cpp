#include "MRMeshBoundary.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRTriMath.h"
#include "MRTimer.h"
#include <algorithm>
#include <array>

namespace MR
{

void straightenBoundary( Mesh & mesh, EdgeId bd, float minNeiNormalsDot, float maxTriAspectRatio, FaceBitSet* newFaces )
{
    MR_TIMER;
    MR_WRITER( mesh );

    assert( !mesh.topology.left( bd ).valid() );

    auto addFaceId = [&]()
    {
        auto nf = mesh.topology.addFaceId();
        if ( newFaces )
            newFaces->autoResizeSet( nf );
        return nf;
    };

    EdgeId e = bd;
    do
    {
        if ( mesh.topology.isLeftTri( e ) )
        {
            // create final triangle that closes the hole
            mesh.topology.setLeft( e, addFaceId() );
            return;
        }

        EdgeId e1 = mesh.topology.prev( e.sym() );
        if ( mesh.topology.right( e ) != mesh.topology.right( e1 ) )
        {
            const auto & ap = mesh.orgPnt( e );
            const auto & bp = mesh.orgPnt( e1 );
            const auto & cp = mesh.destPnt( e1 );
            auto newNormal = cross( bp - ap, cp - ap ).normalized();
            if ( triangleAspectRatio( ap, bp, cp ) <= maxTriAspectRatio
              && dot( newNormal, mesh.leftNormal( e.sym() ) ) >= minNeiNormalsDot 
              && dot( newNormal, mesh.leftNormal( e1.sym() ) ) >= minNeiNormalsDot )
            {
                // create new triangle
                EdgeId x = mesh.topology.makeEdge();
                mesh.topology.splice( e, x );
                mesh.topology.splice( mesh.topology.prev( e1.sym() ), x.sym() );
                mesh.topology.setLeft( e, addFaceId() );
                if ( e == bd || e1 == bd )
                {
                    bd = mesh.topology.next( x ).sym();
                }
                e1 = x;
            }
        }
        e = e1;
    } while ( e != bd );
}

bool sameBoundary( const Mesh & a, const Mesh & b )
{
    MR_TIMER;

    // coordinates of origin and destination of every boundary edge having the hole to the left, sorted
    using Segm = std::array<float, 6>;
    auto getSegms = []( const Mesh & m )
    {
        std::vector<Segm> res;
        for ( EdgeId e : undirectedEdges( m.topology ) )
        {
            const bool hasLeft = m.topology.left( e ).valid();
            if ( hasLeft == m.topology.right( e ).valid() )
                continue; // inner or lone edge
            if ( hasLeft )
                e = e.sym();
            const auto o = m.orgPnt( e );
            const auto d = m.destPnt( e );
            res.push_back( { o.x, o.y, o.z, d.x, d.y, d.z } );
        }
        std::sort( res.begin(), res.end() );
        return res;
    };

    return getSegms( a ) == getSegms( b );
}

} //namespace MR
