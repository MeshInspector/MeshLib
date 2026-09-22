#include "MRNormalDenoising.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRMeshNormals.h"
#include "MRMeshMath.h"
#include "MRNormalsToPoints.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRTimer.h"
#include <limits>
#include <tuple>

#include <MRPch/MREigenSparseCore.h>
#include <Eigen/SparseCholesky>

namespace MR
{

namespace
{

// the (topology, points) forms of denoiseNormals and computePerFaceNormals, kept file-local
// so that the public overload sets stay single-function and their C bindings keep their plain names
void denoiseNormals( const MeshTopology & topology, const VertCoords & points, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma )
{
    MR_TIMER;

    const auto sz = normals.size();
    assert( (int)sz >= topology.lastValidFace() );
    assert( v.size() == topology.undirectedEdgeSize() );
    if ( sz <= 0 )
        return;

    // perimeter of every face, also counting boundary edges for better results on mesh boundary
    Buffer<float, FaceId> perimeter( sz );
    BitSetParallelFor( topology.getValidFaces(), [&]( FaceId f )
    {
        float p = 0;
        for ( auto e : leftRing( topology, f ) )
            p += edgeLength( topology, points, e.undirected() );
        perimeter[f] = p;
    } );

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( sz );
    for ( auto f = 0_f; f < sz; ++f )
    {
        float centralWeight = 1;
        if ( topology.hasFace( f ) )
        {
            for ( auto e : leftRing( topology, f ) )
            {
                assert( topology.left( e ) == f );
                const auto r = topology.right( e );
                if ( !r )
                    continue;
                const auto sumPerimeter = perimeter[f] + perimeter[r];
                if ( sumPerimeter <= 0 )
                    continue;
                // the weight is symmetric in (f,r), so the matrix is symmetric positive definite as SimplicialLDLT requires
                const float weight = gamma * edgeLength( topology, points, e.undirected() ) * sqr( v[e.undirected()] ) * 2 / sumPerimeter;
                centralWeight += weight;
                mTriplets.emplace_back( f, r, -weight );
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

FaceNormals computePerFaceNormals( const MeshTopology & topology, const VertCoords & points )
{
    MR_TIMER;
    FaceNormals res( topology.faceSize() );
    BitSetParallelFor( topology.getValidFaces(), [&]( FaceId f )
    {
        res[f] = normal( topology, points, f );
    } );
    return res;
}

} //anonymous namespace

void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma )
{
    denoiseNormals( mesh.topology, mesh.points, normals, v, gamma );
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
    mesh.invalidateCaches();
    for ( int i = 0; i < settings.pointIters; ++i )
        n2p.run( guide, fnormals, mesh.points, maxInitialDistSq );

    reportProgress( settings.cb, 1.0f );
    return {};
}

void meshDenoiseWithCreases( Mesh & mesh, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings )
{
    mesh.invalidateCaches();
    meshDenoiseWithCreases( mesh.topology, mesh.points, creases, settings );
}

void meshDenoiseWithCreases( const MeshTopology & topology, VertCoords & points, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings )
{
    std::ignore = meshDenoiseWithCreases( topology, points, creases, settings, {} );
}

Expected<void> meshDenoiseWithCreases( Mesh & mesh, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings, const ProgressCallback & cb )
{
    mesh.invalidateCaches();
    return meshDenoiseWithCreases( mesh.topology, mesh.points, creases, settings, cb );
}

Expected<void> meshDenoiseWithCreases( const MeshTopology & topology, VertCoords & points, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings, const ProgressCallback & cb )
{
    MR_TIMER;

    if ( !reportProgress( cb, 0.0f ) )
        return unexpectedOperationCanceled();

    Vector<float, UndirectedEdgeId> v( topology.undirectedEdgeSize() );
    ParallelFor( v, [&]( UndirectedEdgeId ue )
    {
        v[ue] = creases.test( ue ) ? 0.0f : 1.0f;
    } );

    auto fnormals = computePerFaceNormals( topology, points );
    denoiseNormals( topology, points, fnormals, v, settings.gamma );
    if ( !reportProgress( cb, 0.5f ) )
        return unexpectedOperationCanceled();

    const auto guide = points;
    NormalsToPoints n2p;
    n2p.prepare( topology, settings.guideWeight );

    auto sp = subprogress( cb, 0.5f, 1.0f );
    for ( int i = 0; i < settings.pointIters; ++i )
    {
        if ( !reportProgress( sp, float( i ) / settings.pointIters ) )
            return unexpectedOperationCanceled();
        n2p.run( guide, fnormals, points );
    }

    reportProgress( cb, 1.0f );
    return {};
}

} //namespace MR
