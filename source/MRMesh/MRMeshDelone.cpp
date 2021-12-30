#include "MRMeshDelone.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

namespace MR
{

bool checkDeloneQuadrangle( const Vector3d& a, const Vector3d& b, const Vector3d& c, const Vector3d& d )
{
    auto dir = []( const auto& p, const auto& q, const auto& r )
    {
        return cross( q - p, r - p );
    };
    auto dirABD = dir( a, b, d );
    auto dirDBC = dir( d, b, c );

    if ( dot( dirABD, dirDBC ) < 0 )
        return true; // flipping of given edge will create two faces with opposite normals

    auto metricAC = std::max( circumcircleDiameter( a, c, d ), circumcircleDiameter( c, a, b ) );
    auto metricBD = std::max( circumcircleDiameter( b, d, a ), circumcircleDiameter( d, b, c ) );
    return metricAC <= metricBD;
}

bool checkDeloneQuadrangle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d )
{
    return checkDeloneQuadrangle( Vector3d{a}, Vector3d{b}, Vector3d{c}, Vector3d{d} );
}

bool checkDeloneQuadrangleInMesh( const Mesh & mesh, EdgeId edge, float maxDeviationAfterFlip, const FaceBitSet * region )
{
    if ( !mesh.topology.isInnerEdge( edge, region ) )
        return true; // consider condition satisfied for not inner edges

    VertId a, b, c, d;
    mesh.topology.getLeftTriVerts( edge, a, c, d );
    assert( a != c );
    b = mesh.topology.dest( mesh.topology.prev( edge ) );
    if( b == d )
        return true; // consider condition satisfied to avoid creation of loop edges

    bool edgeIsMultiple = false;
    for ( auto e : orgRing0( mesh.topology, edge ) )
    {
        if ( mesh.topology.dest( e ) == c )
        {
            edgeIsMultiple = true;
            break;
        }
    }

    bool flipEdgeWillBeMultiple = false;
    for ( auto e : orgRing( mesh.topology, mesh.topology.next( edge ).sym()  ) )
    {
        assert( mesh.topology.org( e ) == d );
        if ( mesh.topology.dest( e ) == b )
        {
            flipEdgeWillBeMultiple = true;
            break;
        }
    }

    if ( edgeIsMultiple && !flipEdgeWillBeMultiple )
        return false;
    if ( !edgeIsMultiple && flipEdgeWillBeMultiple )
        return true;

    auto ap = mesh.points[a];
    auto bp = mesh.points[b];
    auto cp = mesh.points[c];
    auto dp = mesh.points[d];

    if ( maxDeviationAfterFlip < FLT_MAX )
    {
        // two possible diagonals in the quadrangle
        auto diag0 = cp - ap;
        auto diag1 = dp - bp;
        // distance between them
        double dist = fabs( dot( cross( diag0, diag1 ).normalized(), bp - ap ) );
        if ( dist > maxDeviationAfterFlip )
            return true; // flipping of given edge will change the surface shape too much
    }

    return checkDeloneQuadrangle( ap, bp, cp, dp );
}

int makeDeloneEdgeFlips( Mesh & mesh, int numIters, float maxDeviationAfterFlip, const FaceBitSet * region )
{
    if ( numIters <= 0 )
        return 0;
    MR_TIMER;
    MR_MESH_WRITER( mesh );

    int flipsDone = 0;
    for ( int iter = 0; iter < numIters; ++iter )
    {
        int flipsDoneBeforeThisIter = flipsDone;
        for ( UndirectedEdgeId e : undirectedEdges( mesh.topology ) )
        {
            if ( checkDeloneQuadrangleInMesh( mesh, e, maxDeviationAfterFlip, region ) )
                continue;

            mesh.topology.flipEdge( e );
            ++flipsDone;
        }
        if ( flipsDoneBeforeThisIter == flipsDone )
            break; 
    }
    return flipsDone;
}

void makeDeloneOriginRing( Mesh & mesh, EdgeId e, float maxDeviationAfterFlip, const FaceBitSet * region )
{
    MR_MESH_WRITER( mesh );
    const EdgeId e0 = e;
    for (;;)
    {
        auto testEdge = mesh.topology.prev( e.sym() );
        if ( !mesh.topology.left( testEdge ).valid() || !mesh.topology.right( testEdge ).valid() 
            || checkDeloneQuadrangleInMesh( mesh, testEdge, maxDeviationAfterFlip, region ) )
        {
            e = mesh.topology.next( e );
            if ( e == e0 )
                break; // full ring has been inspected
            continue;
        }
        mesh.topology.flipEdge( testEdge );
    } 
}

} //namespace MR
