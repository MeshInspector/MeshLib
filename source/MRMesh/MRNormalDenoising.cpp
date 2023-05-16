#include "MRNormalDenoising.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wunknown-warning-option" // for next one
#pragma clang diagnostic ignored "-Wunused-but-set-variable" // for newer clang
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma )
{
    MR_TIMER

    const auto sz = normals.size();
    assert( sz == mesh.topology.faceSize() );
    assert( v.size() == mesh.topology.undirectedEdgeSize() );
    if ( sz <= 0 )
        return;

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( sz );
    for ( auto f = 0_f; f < sz; ++f )
    {
        int n = 0;
        FaceId rf[3];
        float w[3];
        float sumLen = 0;
        if ( mesh.topology.hasFace( f ) )
        {
            for ( auto e : leftRing( mesh.topology, f ) )
            {
                assert( mesh.topology.left( e ) == f );
                const auto r = mesh.topology.right( e );
                if ( !r )
                    continue;
                auto len = mesh.edgeLength( e );
                assert( n < 3 );
                rf[n] = r;
                w[n] = gamma * len * sqr( v[e.undirected()] );
                sumLen += len;
                ++n;
            }
        }
        float centralWeight = 1;
        if ( sumLen > 0 )
        {
            for ( int i = 0; i < 3; ++i )
            {
                if ( !rf[i] )
                    break;
                float weight = w[i] / sumLen;
                centralWeight += weight;
                mTriplets.emplace_back( f, rf[i], -weight );
            }
        }
        mTriplets.emplace_back( f, f, centralWeight );
        const auto nm = normals[f];
        for ( int i = 0; i < 3; ++i )
            rhs[i][f] = nm[i];
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

    // copy solution back into normals
    ParallelFor( normals, [&]( FaceId f )
    {
        normals[f] = Vector3f(
            (float) sol[0][f],
            (float) sol[1][f],
            (float) sol[2][f] ).normalized();
    } );
}

} //namespace MR
