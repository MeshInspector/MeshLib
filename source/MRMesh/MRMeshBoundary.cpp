#include "MRMeshBoundary.h"
#include "MRMesh.h"
#include "MRTriMath.h"
#include "MRTimer.h"

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

} //namespace MR
