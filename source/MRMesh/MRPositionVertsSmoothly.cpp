#include "MRPositionVertsSmoothly.h"
#include "MRRingIterator.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <Eigen/SparseCholesky>

namespace MR
{

void positionVertsSmoothly( Mesh& mesh, const VertBitSet& verts,
    Laplacian::EdgeWeights egdeWeightsType,
    const VertBitSet * fixedSharpVertices )
{
    MR_TIMER

    Laplacian laplacian( mesh );
    laplacian.init( verts, egdeWeightsType, Laplacian::RememberShape::No );
    if ( fixedSharpVertices )
        for ( auto v : *fixedSharpVertices )
            laplacian.fixVertex( v, false );
    laplacian.apply();
}

void positionVertsSmoothlySharpBd( Mesh& mesh, const VertBitSet& verts )
{
    MR_TIMER
    assert( !MeshComponents::hasFullySelectedComponent( mesh, verts ) );

    const auto sz = verts.count();
    if ( sz <= 0 )
        return;

    // vertex id -> position in the matrix
    HashMap<VertId, int> vertToMatPos;
    int n = 0;
    for ( auto v : verts )
        vertToMatPos[v] = n++;

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( sz );
    n = 0;
    for ( auto v : verts )
    {
        double sumW = 0;
        for ( [[maybe_unused]] auto e : orgRing( mesh.topology, v ) )
            sumW += 1;
        mTriplets.emplace_back( n, n, sumW );
        Vector3d sumFixed;
        for ( auto e : orgRing( mesh.topology, v ) )
        {
            auto d = mesh.topology.dest( e );
            if ( auto it = vertToMatPos.find( d ); it != vertToMatPos.end() )
            {
                // free neighbor
                int di = it->second;
                mTriplets.emplace_back( n, di, -1 );
            }
            else
            {
                // fixed neighbor
                sumFixed += Vector3d( mesh.points[d] );
            }
        }
        for ( int i = 0; i < 3; ++i )
            rhs[i][n] = sumFixed[i];
        ++n;
    }

    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SparseMatrix A;
    A.resize( sz, sz );
    A.setFromTriplets( mTriplets.begin(), mTriplets.end() );
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute( A );

    Eigen::VectorXd sol[3];
    tbb::parallel_for( tbb::blocked_range<int>( 0, 3, 1 ), [&]( const tbb::blocked_range<int> & range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
            sol[i] = solver.solve( rhs[i] );
    } );

    // copy solution back into mesh points
    n = 0;
    for ( auto v : verts )
    {
        auto & pt = mesh.points[v];
        pt.x = (float) sol[0][n];
        pt.y = (float) sol[1][n];
        pt.z = (float) sol[2][n];
        ++n;
    }
}

} //namespace MR
