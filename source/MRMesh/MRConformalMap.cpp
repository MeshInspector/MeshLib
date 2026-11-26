#include "MRConformalMap.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRRegionBoundary.h"
#include "MRBestFit.h"
#include "MRDirMaxBruteForce.h"
#include "MRReducePath.h"
#include "MRVector2.h"
#include "MRTimer.h"
#include "MRPch/MREigenSparseCore.h"
#include <Eigen/SparseCholesky>

namespace MR
{

Expected<VertUVCoords> leastSquaresConformalMap( const MeshPart& mp )
{
    MR_TIMER;
    const auto& topology = mp.mesh.topology;
    const auto& faces = topology.getFaceIds( mp.region );
    const auto verts = getIncidentVerts( topology, faces );
    if ( verts.count() < 3 )
        return unexpected( "Too few vertices" );

    // pin two vertices most distant along the main direction: pa in (0,0) and pb in (1,0)
    PointAccumulator acc;
    for ( auto v : verts )
        acc.addPoint( mp.mesh.points[v] );
    const auto mm = findDirMinMaxBruteForce( acc.getBestLinef().d, mp.mesh.points, &verts );
    const VertId pa = mm.minArg, pb = mm.maxArg;
    if ( !pa || !pb )
        return unexpected( "Not finite coordinates" );
    // the unknowns are u coordinates of not-pinned vertices in columns col[v], and their v coordinates in columns col[v] + numFree
    auto freeVerts = verts;
    freeVerts.reset( pa );
    freeVerts.reset( pb );
    const auto col = makeVectorWithSeqNums( freeVerts );
    const int numFree = int( freeVerts.count() );

    std::vector<Eigen::Triplet<double>> trips;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero( 2 * int( faces.count() ) );
    int row = 0;
    for ( auto f : faces )
    {
        const auto t = topology.getTriVerts( f );
        const Vector3d p0( mp.mesh.points[t[0]] );
        const Vector3d e1 = Vector3d( mp.mesh.points[t[1]] ) - p0, e2 = Vector3d( mp.mesh.points[t[2]] ) - p0;
        const double l1 = e1.length();
        // the triangle in the plane with the vertices in (0,0), (l1,0) and q[2] to the left
        const Vector2d q[3] = { {}, { l1, 0 }, unfoldOnPlane( e1, e2, Vector2d( l1, 0 ), true ) };
        const double ar = l1 * q[2].y; // twice the area
        const double w = ar > 0 ? 1 / std::sqrt( ar ) : 0; // zero rows for degenerate triangles, the coordinates are finite here
        for ( int j = 0; j < 3; ++j )
        {
            const int a = ( j + 1 ) % 3, b = ( j + 2 ) % 3;
            const double wr = ( q[b].x - q[a].x ) * w, wi = ( q[b].y - q[a].y ) * w;
            if ( const int c = col[t[j]]; c >= 0 )
            {
                trips.emplace_back( row, c, wr );
                trips.emplace_back( row, c + numFree, -wi );
                trips.emplace_back( row + 1, c, wi );
                trips.emplace_back( row + 1, c + numFree, wr );
            }
            else if ( t[j] == pb ) // pinned pa = (0,0) adds nothing
            {
                rhs[row] -= wr;
                rhs[row + 1] -= wi;
            }
        }
        row += 2;
    }

    Eigen::SparseMatrix<double> m( rhs.size(), 2 * numFree );
    m.setFromTriplets( trips.begin(), trips.end() );
    // without non-degenerate triangles the matrix is zero and its factorization fails
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt( m.transpose() * m );
    if ( ldlt.info() != Eigen::Success )
        return unexpected( "Degenerate system" );
    const Eigen::VectorXd sol = ldlt.solve( m.transpose() * rhs );
    if ( !sol.allFinite() )
        return unexpected( "Degenerate system" );

    VertUVCoords res( topology.vertSize() );
    res[pb].x = 1;
    for ( auto v : freeVerts )
        res[v] = UVCoord( float( sol[col[v]] ), float( sol[col[v] + numFree] ) );
    return res;
}

} //namespace MR
