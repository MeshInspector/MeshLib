#include "MRConformalMap.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRRegionBoundary.h"
#include "MRBestFit.h"
#include "MRDirMaxBruteForce.h"
#include "MRVector2.h"
#include "MRTimer.h"
#include "MRPch/MREigenCore.h"
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
    const auto mm = findDirMinMaxBruteForce( Vector3f( acc.getBestLine().d ), mp.mesh.points, &verts );
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
    std::vector<double> rhsVals;
    int row = 0;
    for ( auto f : faces )
    {
        const auto t = topology.getTriVerts( f );
        const Vector3d p0( mp.mesh.points[t[0]] );
        const Vector3d e1 = Vector3d( mp.mesh.points[t[1]] ) - p0, e2 = Vector3d( mp.mesh.points[t[2]] ) - p0;
        const double l1 = e1.length();
        const Vector3d nrm = cross( e1, e2 );
        const double ar = nrm.length();
        if ( !( l1 > 0 ) || !( ar > 0 ) )
            continue;
        // the triangle in its own orthonormal frame
        const Vector3d ex = e1 / l1, ey = cross( nrm / ar, ex );
        const double x[3] = { 0, l1, dot( e2, ex ) }, y[3] = { 0, 0, dot( e2, ey ) };
        const double w = 1 / std::sqrt( ar );
        rhsVals.resize( row + 2, 0.0 );
        for ( int j = 0; j < 3; ++j )
        {
            const int a = ( j + 1 ) % 3, b = ( j + 2 ) % 3;
            const double wr = ( x[b] - x[a] ) * w, wi = ( y[b] - y[a] ) * w;
            if ( const int c = col[t[j]]; c >= 0 )
            {
                trips.emplace_back( row, c, wr );
                trips.emplace_back( row, c + numFree, -wi );
                trips.emplace_back( row + 1, c, wi );
                trips.emplace_back( row + 1, c + numFree, wr );
            }
            else if ( t[j] == pb ) // pinned pa = (0,0) adds nothing
            {
                rhsVals[row] -= wr;
                rhsVals[row + 1] -= wi;
            }
        }
        row += 2;
    }
    if ( row == 0 )
        return unexpected( "No non-degenerate triangles" );

    Eigen::SparseMatrix<double> m( row, 2 * numFree );
    m.setFromTriplets( trips.begin(), trips.end() );
    const Eigen::VectorXd rhs = Eigen::Map<const Eigen::VectorXd>( rhsVals.data(), row );
    const Eigen::SparseMatrix<double> mtm = m.transpose() * m;
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt( mtm );
    if ( ldlt.info() != Eigen::Success )
        return unexpected( "Degenerate system" );
    const Eigen::VectorXd sol = ldlt.solve( m.transpose() * rhs );

    VertUVCoords res( topology.vertSize() );
    res[pb].x = 1;
    for ( auto v : freeVerts )
        res[v] = UVCoord( float( sol[col[v]] ), float( sol[col[v] + numFree] ) );
    return res;
}

} //namespace MR
