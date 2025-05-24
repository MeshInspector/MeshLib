#include "MRNormalDenoising.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRMeshNormals.h"
#include "MRNormalsToPoints.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include <limits>

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wunknown-warning-option" // for next one
#pragma clang diagnostic ignored "-Wunused-but-set-variable" // for newer clang
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma )
{
    MR_TIMER;

    const auto sz = normals.size();
    assert( (int)sz >= mesh.topology.lastValidFace() );
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
                // even if there is no right face (r), increment sumLen for better results on mesh boundary
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
                    continue;
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

constexpr float eps = 0.001f;

void updateIndicator( const Mesh & mesh, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma )
{
    MR_TIMER;

    const auto sz = v.size();
    assert( sz == mesh.topology.undirectedEdgeSize() );
    assert( (int)normals.size() >= mesh.topology.lastValidFace() );
    if ( sz <= 0 )
        return;

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs;
    rhs.resize( sz );
    const float rh = beta / ( 2 * eps );
    const float k = 2 * beta * eps;
    for ( auto ue = 0_ue; ue < sz; ++ue )
    {
        const EdgeId e = ue; // note that it can be lone edge
        float centralWeight = rh;
        const auto l = mesh.topology.left( e );
        const auto r = mesh.topology.right( e );
        if ( l && r )
            centralWeight += 2 * gamma * ( normals[l] - normals[r] ).lengthSq();
        const auto lenE = ( l || r ) ? mesh.edgeLength( e ) : 0.0f;
        if ( lenE > 0 )
        {
            if ( l )
            {
                const auto c = mesh.triCenter( l );
                {
                    const auto a = mesh.topology.next( e );
                    const auto lenL = ( c - mesh.orgPnt( e ) ).length();
                    const auto x = k * lenL / lenE;
                    centralWeight += x;
                    mTriplets.emplace_back( ue, a.undirected(), -x );
                }
                {
                    const auto b = mesh.topology.prev( e.sym() );
                    const auto lenL = ( c - mesh.destPnt( e ) ).length();
                    const auto x = k * lenL / lenE;
                    centralWeight += x;
                    mTriplets.emplace_back( ue, b.undirected(), -x );
                }
            }
            if ( r )
            {
                const auto c = mesh.triCenter( r );
                {
                    const auto a = mesh.topology.prev( e );
                    const auto lenL = ( c - mesh.orgPnt( e ) ).length();
                    const auto x = k * lenL / lenE;
                    centralWeight += x;
                    mTriplets.emplace_back( ue, a.undirected(), -x );
                }
                {
                    const auto b = mesh.topology.next( e.sym() );
                    const auto lenL = ( c - mesh.destPnt( e ) ).length();
                    const auto x = k * lenL / lenE;
                    centralWeight += x;
                    mTriplets.emplace_back( ue, b.undirected(), -x );
                }
            }
        }
        mTriplets.emplace_back( ue, ue, centralWeight );
        rhs[ue] = rh;
    }

    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SparseMatrix A;
    A.resize( sz, sz );
    A.setFromTriplets( mTriplets.begin(), mTriplets.end() );
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute( A );

    Eigen::VectorXd sol = solver.solve( rhs );

    // copy solution back into v
    ParallelFor( v, [&]( UndirectedEdgeId ue )
    {
        v[ue] = (float) sol[ue];
    } );
}

void updateIndicatorFast( const MeshTopology & topology, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma )
{
    MR_TIMER;

    assert( v.size() == topology.undirectedEdgeSize() );
    assert( (int)normals.size() >= topology.lastValidFace() );

    const float rh = beta / ( 2 * eps );
    ParallelFor( v, [&]( UndirectedEdgeId ue )
    {
        const EdgeId e = ue;
        const auto l = topology.left( e );
        const auto r = topology.right( e );
        if ( !l || !r )
        {
            v[ue] = 1;
            return;
        }
        v[ue] = rh / ( rh + 2 * gamma * ( normals[l] - normals[r] ).lengthSq() );
    } );
}

Expected<void> meshDenoiseViaNormals( Mesh & mesh, const DenoiseViaNormalsSettings & settings )
{
    MR_TIMER;
    if ( settings.normalIters <= 0 || settings.pointIters <= 0 )
    {
        assert( false );
        return unexpected( "Bad parameters" );
    }

    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    auto fnormals0 = computePerFaceNormals( mesh );
    Vector<float, UndirectedEdgeId> v( mesh.topology.undirectedEdgeSize(), 1 );

    if ( !reportProgress( settings.cb, 0.05f ) )
        return unexpectedOperationCanceled();

    auto sp = subprogress( settings.cb, 0.05f, 0.95f );
    FaceNormals fnormals;
    for ( int i = 0; i < settings.normalIters; ++i )
    {
        fnormals = fnormals0;
        denoiseNormals( mesh, fnormals, v, settings.gamma );
        if ( !reportProgress( sp, float( 2 * i ) / ( 2 * settings.normalIters ) ) )
            return unexpectedOperationCanceled();

        if ( settings.fastIndicatorComputation )
            updateIndicatorFast( mesh.topology, v, fnormals, settings.beta, settings.gamma );
        else
            updateIndicator( mesh, v, fnormals, settings.beta, settings.gamma );
        if ( !reportProgress( sp, float( 2 * i + 1 ) / ( 2 * settings.normalIters ) ) )
            return unexpectedOperationCanceled();
    }

    if ( settings.outCreases )
    {
        settings.outCreases->clear();
        settings.outCreases->resize( mesh.topology.undirectedEdgeSize() );
        BitSetParallelForAll( *settings.outCreases, [&]( UndirectedEdgeId ue )
        {
            if ( v[ue] < 0.5f )
                settings.outCreases->set( ue );
        } );
    }

    if ( !reportProgress( settings.cb, 0.95f ) )
        return unexpectedOperationCanceled();

    const auto guide = mesh.points;
    NormalsToPoints n2p;
    n2p.prepare( mesh.topology, settings.guideWeight );
    auto maxInitialDistSq = settings.limitNearInitial ? sqr( settings.maxInitialDist )
        : std::numeric_limits<float>::infinity();
    for ( int i = 0; i < settings.pointIters; ++i )
        n2p.run( guide, fnormals, mesh.points, maxInitialDistSq );

    reportProgress( settings.cb, 1.0f );
    return {};
}

} //namespace MR
