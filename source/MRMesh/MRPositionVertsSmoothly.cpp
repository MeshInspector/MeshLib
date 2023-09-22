#include "MRPositionVertsSmoothly.h"
#include "MRRingIterator.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRBitSetParallelFor.h"
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

void positionVertsSmoothlySharpBd( Mesh& mesh, const VertBitSet& verts, const Vector<Vector3f, VertId>* vertShifts )
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
        if ( vertShifts )
            sumFixed = sumW * Vector3d( (*vertShifts)[v] );
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

void inflate( Mesh& mesh, const VertBitSet& verts, const InflateSettings & settings )
{
    MR_TIMER
    if ( !verts.any() )
        return;
    if ( settings.preSmooth )
        positionVertsSmoothlySharpBd( mesh, verts );
    if ( settings.iterations <= 0 || settings.pressure == 0 )
        return;

    VertScalars a( verts.find_last() + 1 );
    BitSetParallelFor( verts, [&]( VertId v )
    {
        a[v] = mesh.dblArea( v );
    } );
    double sumArea = 0;
    for ( auto v : verts )
        sumArea += a[v];
    if ( sumArea <= 0 )
        return;
    float rAvgArea = float( 1 / sumArea );
    BitSetParallelFor( verts, [&]( VertId v )
    {
        a[v] *= rAvgArea;
    } );
    // a[v] contains relative area around vertex #v in the whole region, sum(a[v]) = 1

    Vector<Vector3f, VertId> vertShifts( a.size() );
    for ( int i = 0; i < settings.iterations; ++i )
    {
        const auto currPressure = settings.gradualPressureGrowth ?
            ( i + 1 ) * settings.pressure / settings.iterations : settings.pressure;
        BitSetParallelFor( verts, [&]( VertId v )
        {
            vertShifts[v] = currPressure * a[v] * mesh.normal( v );
        } );
        positionVertsSmoothlySharpBd( mesh, verts, &vertShifts );
    }
}

} //namespace MR
