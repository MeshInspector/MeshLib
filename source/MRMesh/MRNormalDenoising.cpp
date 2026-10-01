#include "MRNormalDenoising.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRMeshNormals.h"
#include "MRMeshMath.h"
#include "MRRegionBoundary.h"
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

void denoiseNormals( const Mesh & mesh, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma, const FaceBitSet * region )
{
    denoiseNormals( mesh.topology, mesh.points, normals, v, gamma, region );
}

void denoiseNormals( const MeshTopology & topology, const VertCoords & points, FaceNormals & normals, const Vector<float, UndirectedEdgeId> & v, float gamma, const FaceBitSet * region )
{
    MR_TIMER;

    assert( (int)normals.size() >= topology.lastValidFace() );
    assert( v.size() == topology.undirectedEdgeSize() );
    const auto & faces = topology.getFaceIds( region );

    // index of every face with unknown normal in the linear system, -1 for fixed faces;
    // a hash map for a region to avoid allocation for all mesh faces
    Vector<int, FaceId> face2idxVec;
    HashMap<FaceId, int> face2idxMap;
    int sz = 0;
    if ( region )
    {
        face2idxMap = makeHashMapWithSeqNums( faces );
        sz = (int)face2idxMap.size();
    }
    else
    {
        face2idxVec.resize( topology.faceSize(), -1 );
        for ( auto f : faces )
            face2idxVec[f] = sz++;
    }
    if ( sz <= 0 )
        return;
    const auto idxOf = [&]( FaceId f ) -> int
    {
        if ( !region )
            return face2idxVec[f];
        auto it = face2idxMap.find( f );
        return it != face2idxMap.end() ? it->second : -1;
    };

    // perimeter of a face, also counting boundary edges for better results on mesh boundary
    const auto computePerimeter = [&]( FaceId f )
    {
        float p = 0;
        for ( auto e : leftRing( topology, f ) )
            p += edgeLength( topology, points, e.undirected() );
        return p;
    };
    // precomputed for all faces without a region, and computed only when needed for a region
    Buffer<float, FaceId> perimeter( region ? 0 : topology.faceSize() );
    if ( !region )
        BitSetParallelFor( topology.getValidFaces(), [&]( FaceId f ) { perimeter[f] = computePerimeter( f ); } );
    const auto perimeterOf = [&]( FaceId f ) { return region ? computePerimeter( f ) : perimeter[f]; };

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( sz );
    for ( auto f : faces )
    {
        const int fi = idxOf( f );
        const float pf = perimeterOf( f );
        float centralWeight = 1;
        Vector3d rh( normals[f] );
        for ( auto e : leftRing( topology, f ) )
        {
            assert( topology.left( e ) == f );
            const auto r = topology.right( e );
            if ( !r )
                continue;
            const auto sumPerimeter = pf + perimeterOf( r );
            if ( sumPerimeter <= 0 )
                continue;
            // the weight is symmetric in (f,r), so the matrix is symmetric positive definite as SimplicialLDLT requires
            const float weight = gamma * edgeLength( topology, points, e.undirected() ) * sqr( v[e.undirected()] ) * 2 / sumPerimeter;
            centralWeight += weight;
            if ( const int ri = idxOf( r ); ri >= 0 )
                mTriplets.emplace_back( fi, ri, -weight );
            else
                rh += double( weight ) * Vector3d( normals[r] ); // fixed normal of a face outside the region
        }
        mTriplets.emplace_back( fi, fi, centralWeight );
        for ( int i = 0; i < 3; ++i )
            rhs[i][fi] = rh[i];
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
    BitSetParallelFor( faces, [&]( FaceId f )
    {
        const int fi = idxOf( f );
        normals[f] = Vector3f(
            (float) sol[0][fi],
            (float) sol[1][fi],
            (float) sol[2][fi] ).normalized();
    } );
}

constexpr float eps = 0.001f;

void updateIndicator( const MeshPart & mp, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma )
{
    MR_TIMER;
    const auto & mesh = mp.mesh;
    const auto * region = mp.region;

    assert( v.size() == mesh.topology.undirectedEdgeSize() );
    assert( (int)normals.size() >= mesh.topology.lastValidFace() );

    // the edges with unknown indicator in the linear system, the indicator of all other edges is fixed
    UndirectedEdgeBitSet regionEdges;
    HashMap<UndirectedEdgeId, int> edge2idx;
    if ( region )
    {
        regionEdges = getIncidentEdges( mesh.topology, *region );
        edge2idx = makeHashMapWithSeqNums( regionEdges );
    }
    const int sz = region ? (int)edge2idx.size() : (int)v.size();
    if ( sz <= 0 )
        return;
    // index of given edge in the linear system, -1 for fixed edges
    const auto idxOf = [&]( UndirectedEdgeId ue ) -> int
    {
        if ( !region )
            return int( ue );
        auto it = edge2idx.find( ue );
        return it != edge2idx.end() ? it->second : -1;
    };

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs;
    rhs.resize( sz );
    const float rh = beta / ( 2 * eps );
    const float k = 2 * beta * eps;
    const auto addEquation = [&]( UndirectedEdgeId ue, int row )
    {
        const EdgeId e = ue; // note that it can be lone edge
        float centralWeight = rh;
        double rhsRow = rh;
        const auto addNeighbor = [&]( EdgeId n, float x )
        {
            centralWeight += x;
            if ( const int c = idxOf( n.undirected() ); c >= 0 )
                mTriplets.emplace_back( row, c, -x );
            else
                rhsRow += double( x ) * v[n.undirected()]; // fixed indicator of an edge outside the region
        };
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
                addNeighbor( mesh.topology.next( e ), k * ( c - mesh.orgPnt( e ) ).length() / lenE );
                addNeighbor( mesh.topology.prev( e.sym() ), k * ( c - mesh.destPnt( e ) ).length() / lenE );
            }
            if ( r )
            {
                const auto c = mesh.triCenter( r );
                addNeighbor( mesh.topology.prev( e ), k * ( c - mesh.orgPnt( e ) ).length() / lenE );
                addNeighbor( mesh.topology.next( e.sym() ), k * ( c - mesh.destPnt( e ) ).length() / lenE );
            }
        }
        mTriplets.emplace_back( row, row, centralWeight );
        rhs[row] = rhsRow;
    };
    if ( region )
    {
        int row = 0;
        for ( auto ue : regionEdges )
            addEquation( ue, row++ );
    }
    else
    {
        for ( auto ue = 0_ue; ue < v.size(); ++ue )
            addEquation( ue, int( ue ) );
    }

    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SparseMatrix A;
    A.resize( sz, sz );
    A.setFromTriplets( mTriplets.begin(), mTriplets.end() );
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute( A );

    Eigen::VectorXd sol = solver.solve( rhs );

    // copy solution back into v
    if ( region )
    {
        BitSetParallelFor( regionEdges, [&]( UndirectedEdgeId ue )
        {
            v[ue] = (float) sol[idxOf( ue )];
        } );
    }
    else
    {
        ParallelFor( v, [&]( UndirectedEdgeId ue )
        {
            v[ue] = (float) sol[ue];
        } );
    }
}

void updateIndicatorFast( const MeshTopology & topology, Vector<float, UndirectedEdgeId> & v, const FaceNormals & normals, float beta, float gamma,
    const FaceBitSet * region )
{
    MR_TIMER;

    assert( v.size() == topology.undirectedEdgeSize() );
    assert( (int)normals.size() >= topology.lastValidFace() );

    const float rh = beta / ( 2 * eps );
    const auto update = [&]( UndirectedEdgeId ue )
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
    };
    if ( region )
        BitSetParallelFor( getIncidentEdges( topology, *region ), update );
    else
        ParallelFor( v, update );
}

/// computes the normals of the faces read during denoising of given region: region faces and their neighbors;
/// the normals of all other faces are left zero
static FaceNormals computeNeededNormals( const MeshTopology & topology, const VertCoords & points, const FaceBitSet * region )
{
    if ( !region )
        return computePerFaceNormals( topology, points );
    MR_TIMER;
    FaceNormals res( topology.faceSize() );
    BitSetParallelFor( getIncidentFaces( topology, getIncidentEdges( topology, *region ) ), [&]( FaceId f )
    {
        res[f] = normal( topology, points, f );
    } );
    return res;
}

void meshDenoiseViaNormals( Mesh & mesh, const DenoiseViaNormalsSettings & settings )
{
    std::ignore = meshDenoiseViaNormals( mesh, settings, {} );
}

bool meshDenoiseViaNormals( Mesh & mesh, const DenoiseViaNormalsSettings & settings, const ProgressCallback & cb )
{
    MR_TIMER;
    assert( settings.normalIters > 0 && settings.pointIters > 0 );
    if ( settings.normalIters <= 0 || settings.pointIters <= 0 )
        return true;

    if ( !reportProgress( cb, 0.0f ) )
        return false;

    auto fnormals = computeNeededNormals( mesh.topology, mesh.points, settings.region );
    Vector<float, UndirectedEdgeId> v( mesh.topology.undirectedEdgeSize(), 1 );

    // denoiseNormals changes only the normals of region faces, so only they are restored before each iteration
    FaceNormals fnormals0; // initial normals of all faces without a region
    std::vector<std::pair<FaceId, Vector3f>> regionNormals0; // initial normals of region faces
    if ( settings.region )
    {
        for ( auto f : *settings.region )
            regionNormals0.emplace_back( f, fnormals[f] );
    }
    else
        fnormals0 = fnormals;

    if ( !reportProgress( cb, 0.05f ) )
        return false;

    auto sp = subprogress( cb, 0.05f, 0.95f );
    for ( int i = 0; i < settings.normalIters; ++i )
    {
        if ( i > 0 )
        {
            if ( settings.region )
                ParallelFor( regionNormals0, [&]( size_t j ) { fnormals[regionNormals0[j].first] = regionNormals0[j].second; } );
            else
                fnormals = fnormals0;
        }
        denoiseNormals( mesh, fnormals, v, settings.gamma, settings.region );
        if ( !reportProgress( sp, float( 2 * i ) / ( 2 * settings.normalIters ) ) )
            return false;

        if ( settings.fastIndicatorComputation )
            updateIndicatorFast( mesh.topology, v, fnormals, settings.beta, settings.gamma, settings.region );
        else
            updateIndicator( { mesh, settings.region }, v, fnormals, settings.beta, settings.gamma );
        if ( !reportProgress( sp, float( 2 * i + 1 ) / ( 2 * settings.normalIters ) ) )
            return false;
    }

    if ( settings.outCreases )
    {
        settings.outCreases->clear();
        settings.outCreases->resize( mesh.topology.undirectedEdgeSize() );
        BitSetParallelForAll( *settings.outCreases, [&]( UndirectedEdgeId ue )
        {
            if ( v[ue] < 0.5f && ( !settings.region
                || ( contains( *settings.region, mesh.topology.left( ue ) ) && contains( *settings.region, mesh.topology.right( ue ) ) ) ) )
                settings.outCreases->set( ue );
        } );
    }

    if ( !reportProgress( cb, 0.95f ) )
        return false;

    VertBitSet innerVerts;
    if ( settings.region )
        innerVerts = getRegionInnerVerts( mesh.topology, *settings.region );

    const auto guide = mesh.points;
    NormalsToPoints n2p;
    n2p.prepare( mesh.topology, settings.guideWeight, settings.region ? &innerVerts : nullptr );
    auto maxInitialDistSq = settings.limitNearInitial ? sqr( settings.maxInitialDist )
        : std::numeric_limits<float>::infinity();
    mesh.invalidateCaches();
    for ( int i = 0; i < settings.pointIters; ++i )
        n2p.run( guide, fnormals, mesh.points, maxInitialDistSq );

    reportProgress( cb, 1.0f );
    return true;
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

bool meshDenoiseWithCreases( Mesh & mesh, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings, const ProgressCallback & cb )
{
    mesh.invalidateCaches();
    return meshDenoiseWithCreases( mesh.topology, mesh.points, creases, settings, cb );
}

bool meshDenoiseWithCreases( const MeshTopology & topology, VertCoords & points, const UndirectedEdgeBitSet & creases, const DenoiseWithCreasesSettings & settings, const ProgressCallback & cb )
{
    MR_TIMER;

    if ( !reportProgress( cb, 0.0f ) )
        return false;

    Vector<float, UndirectedEdgeId> v( topology.undirectedEdgeSize() );
    ParallelFor( v, [&]( UndirectedEdgeId ue )
    {
        v[ue] = creases.test( ue ) ? 0.0f : 1.0f;
    } );

    auto fnormals = computeNeededNormals( topology, points, settings.region );
    denoiseNormals( topology, points, fnormals, v, settings.gamma, settings.region );
    if ( !reportProgress( cb, 0.5f ) )
        return false;

    VertBitSet innerVerts;
    if ( settings.region )
        innerVerts = getRegionInnerVerts( topology, *settings.region );

    const auto guide = points;
    NormalsToPoints n2p;
    n2p.prepare( topology, settings.guideWeight, settings.region ? &innerVerts : nullptr );

    auto sp = subprogress( cb, 0.5f, 1.0f );
    for ( int i = 0; i < settings.pointIters; ++i )
    {
        if ( !reportProgress( sp, float( i ) / settings.pointIters ) )
            return false;
        n2p.run( guide, fnormals, points );
    }

    reportProgress( cb, 1.0f );
    return true;
}

} //namespace MR
