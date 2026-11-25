#include "MRConformalMap.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRRegionBoundary.h"
#include "MRSymMatrix3.h"
#include "MRMatrix3.h"
#include "MRVector2.h"
#include "MRTimer.h"
#include "MRPch/MREigenCore.h"
#include <Eigen/SparseCholesky>

namespace MR
{

Expected<VertCoords2> leastSquaresConformalMap( const MeshPart& mp )
{
    MR_TIMER;
    const auto& topology = mp.mesh.topology;
    const auto& faces = topology.getFaceIds( mp.region );
    const auto verts = getIncidentVerts( topology, faces );
    const int n = int( verts.count() );
    if ( n < 3 )
        return unexpected( "Too few vertices" );

    // pin two vertices most distant along the main direction
    Vector3d mean;
    for ( auto v : verts )
        mean += Vector3d( mp.mesh.points[v] );
    mean /= double( n );
    SymMatrix3d cov;
    for ( auto v : verts )
        cov += outerSquare( Vector3d( mp.mesh.points[v] ) - mean );
    Matrix3d vecs;
    cov.eigens( &vecs );
    const Vector3d axis = vecs.z;
    VertId pa, pb;
    for ( auto v : verts )
    {
        const double t = dot( Vector3d( mp.mesh.points[v] ), axis );
        if ( !pa || t < dot( Vector3d( mp.mesh.points[pa] ), axis ) )
            pa = v;
        if ( !pb || t > dot( Vector3d( mp.mesh.points[pb] ), axis ) )
            pb = v;
    }

    // the unknowns are u and v coordinates of not-pinned vertices
    Vector<int, VertId> uCol( topology.vertSize(), -1 ), vCol( topology.vertSize(), -1 );
    int numFree = 0;
    for ( auto v : verts )
        if ( v != pa && v != pb )
            uCol[v] = numFree++;
    for ( auto v : verts )
        if ( v != pa && v != pb )
            vCol[v] = numFree++;

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
            const VertId v = t[j];
            // the pinned vertices have (u,v) = (0,0) and (1,0)
            auto add = [&]( int r, int col, bool isU, double coeff )
            {
                if ( col >= 0 )
                    trips.emplace_back( r, col, coeff );
                else if ( isU && v == pb )
                    rhsVals[r] -= coeff;
            };
            add( row, uCol[v], true, wr );
            add( row, vCol[v], false, -wi );
            add( row + 1, uCol[v], true, wi );
            add( row + 1, vCol[v], false, wr );
        }
        row += 2;
    }
    if ( row == 0 )
        return unexpected( "No non-degenerate triangles" );

    Eigen::SparseMatrix<double> m( row, numFree );
    m.setFromTriplets( trips.begin(), trips.end() );
    const Eigen::VectorXd rhs = Eigen::Map<const Eigen::VectorXd>( rhsVals.data(), row );
    const Eigen::SparseMatrix<double> mtm = m.transpose() * m;
    const Eigen::VectorXd sol = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>( mtm ).solve( m.transpose() * rhs );

    VertCoords2 res( topology.vertSize() );
    for ( auto v : verts )
    {
        res[v].x = uCol[v] >= 0 ? float( sol[uCol[v]] ) : ( v == pb ? 1.0f : 0.0f );
        res[v].y = vCol[v] >= 0 ? float( sol[vCol[v]] ) : 0.0f;
        if ( !std::isfinite( res[v].x + res[v].y ) )
            return unexpected( "Degenerate system" );
    }
    return res;
}

} //namespace MR
