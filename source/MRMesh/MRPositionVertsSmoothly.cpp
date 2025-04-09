#include "MRPositionVertsSmoothly.h"
#include "MRRingIterator.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRTriMath.h"
#include "MRMeshRelax.h"
#include "MRLaplacian.h"
#include "MRTimer.h"
#include <Eigen/SparseCholesky>

namespace MR
{

void positionVertsSmoothly( Mesh& mesh, const VertBitSet& verts,
    EdgeWeights edgeWeights, VertexMass vmass, const VertBitSet * fixedSharpVertices )
{
    mesh.invalidateCaches();
    positionVertsSmoothly( mesh.topology, mesh.points, verts, edgeWeights, vmass, fixedSharpVertices );
}

void positionVertsSmoothly( const MeshTopology& topology, VertCoords& points, const VertBitSet& verts,
    EdgeWeights edgeWeights, VertexMass vmass, const VertBitSet * fixedSharpVertices )
{
    MR_TIMER

    Laplacian laplacian( topology, points );
    laplacian.init( verts, edgeWeights, vmass, Laplacian::RememberShape::No );
    if ( fixedSharpVertices )
        for ( auto v : *fixedSharpVertices )
            laplacian.fixVertex( v, false );
    laplacian.apply();
}

void positionVertsSmoothlySharpBd( Mesh& mesh, const VertBitSet& verts,
    const Vector<Vector3f, VertId>* vertShifts, const VertScalars* vertStabilizers )
{
    mesh.invalidateCaches();
    positionVertsSmoothlySharpBd( mesh.topology, mesh.points, verts, vertShifts, vertStabilizers );
}

void positionVertsSmoothlySharpBd( const MeshTopology& topology, VertCoords& points, const VertBitSet& verts,
    const Vector<Vector3f, VertId>* vertShifts, const VertScalars* vertStabilizers )
{
    MR_TIMER
    assert( vertStabilizers || !MeshComponents::hasFullySelectedComponent( topology, verts ) );

    const auto sz = verts.count();
    if ( sz <= 0 )
        return;

    // vertex id -> position in the matrix
    HashMap<VertId, int> vertToMatPos = makeHashMapWithSeqNums( verts );

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( sz );
    int n = 0;
    for ( auto v : verts )
    {
        double sumW = 0;
        Vector3d sumFixed;
        for ( auto e : orgRing( topology, v ) )
        {
            sumW += 1;
            auto d = topology.dest( e );
            if ( auto it = vertToMatPos.find( d ); it != vertToMatPos.end() )
            {
                // free neighbor
                int di = it->second;
                if ( n > di ) // row > col: fill only lower left part of matrix
                    mTriplets.emplace_back( n, di, -1 );
            }
            else
            {
                // fixed neighbor
                sumFixed += Vector3d( points[d] );
            }
        }
        if ( vertShifts )
            sumFixed += sumW * Vector3d( (*vertShifts)[v] );
        if ( vertStabilizers )
        {
            const auto s = (*vertStabilizers)[v];
            sumW += s;
            sumFixed += Vector3d( s * points[v] );
        }
        mTriplets.emplace_back( n, n, sumW );
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
    ParallelFor( 0, 3, [&]( int i )
    {
        sol[i] = solver.solve( rhs[i] );
    } );

    // copy solution back into mesh points
    n = 0;
    for ( auto v : verts )
    {
        auto & pt = points[v];
        pt.x = (float) sol[0][n];
        pt.y = (float) sol[1][n];
        pt.z = (float) sol[2][n];
        ++n;
    }
}

void positionVertsWithSpacing( Mesh& mesh, const SpacingSettings & settings )
{
    mesh.invalidateCaches();
    positionVertsWithSpacing( mesh.topology, mesh.points, settings );
}

void positionVertsWithSpacing( const MeshTopology& topology, VertCoords& points, const SpacingSettings & settings )
{
    MR_TIMER
    assert( settings.maxSumNegW > 0 );

    const auto & verts = topology.getVertIds( settings.region );
    const auto sz = verts.count();
    if ( sz <= 0 || settings.numIters <= 0 )
        return;

    FaceBitSet myFaces;
    const FaceBitSet * incidentFaces = nullptr;
    if ( settings.isInverted && settings.region )
    {
        myFaces = getIncidentFaces( topology, *settings.region );
        incidentFaces = &myFaces;
    }

    // vertex id -> position in the matrix
    HashMap<VertId, int> vertToMatPos = makeHashMapWithSeqNums( verts );

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( sz );

    VertBitSet shiftedVerts;
    for ( int iter = 0; iter < settings.numIters; ++iter )
    {
        mTriplets.clear();
        int n = 0;
        for ( auto v : verts )
        {
            double sumW = 0;
            float sumNegW = 0;
            Vector3d sumFixed;
            for ( auto e : orgRing( topology, v ) )
            {
                const auto d = topology.dest( e );
                const auto l = ( points[v] - points[d] ).length();
                const auto t = settings.dist( e );
                float w = 0;
                if ( t > l )
                    w = l > 0 ? 1 - t / l : -1;
                else if ( l > t )
                    w = t > 0 ? l / t - 1 : 1;
                sumW += w;
                if ( w < 0 )
                     sumNegW -= w;
                if ( auto it = vertToMatPos.find( d ); it != vertToMatPos.end() )
                {
                    // free neighbor
                    int di = it->second;
                    if ( n > di ) // row > col: fill only lower left part of matrix
                        mTriplets.emplace_back( n, di, -w );
                }
                else
                {
                    // fixed neighbor
                    sumFixed += Vector3d( w * points[d] );
                }
            }
            auto s = settings.stabilizer;
            if ( sumNegW > settings.maxSumNegW )
                s += sumNegW / settings.maxSumNegW;
            sumFixed += Vector3d( s * points[v] );
            mTriplets.emplace_back( n, n, sumW + s );
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
        ParallelFor( 0, 3, [&]( int i )
        {
            sol[i] = solver.solve( rhs[i] );
        } );

        // copy solution back into mesh points
        n = 0;
        for ( auto v : verts )
        {
            auto & pt = points[v];
            pt.x = (float) sol[0][n];
            pt.y = (float) sol[1][n];
            pt.z = (float) sol[2][n];
            ++n;
        }

        if ( settings.isInverted )
        {
            shiftedVerts.clear();
            shiftedVerts.resize( topology.vertSize(), false );
            bool anyInverted = false;
            for ( auto f : topology.getFaceIds( incidentFaces ) )
            {
                if ( !settings.isInverted( f ) )
                    continue;
                anyInverted = true;
                auto vs = topology.getTriVerts( f );
                Triangle3f t0;
                for ( int i = 0; i < 3; ++i )
                    t0[i] = points[ vs[i] ];
                auto t = makeDegenerate( t0 );

                if ( settings.region )
                {
                    // some triangle's vertices can be fixed
                    int numFree = 0;
                    for ( int i = 0; i < 3; ++i )
                        numFree += settings.region->test( vs[i] );
                    assert( numFree >= 1 && numFree <= 3 );
                    if ( numFree == 1 )
                    {
                        // 2 out of 3 vertices are fixed
                        int freeI = -1;
                        for ( int i = 0; i < 3; ++i )
                            if ( settings.region->test( vs[i] ) )
                            {
                                freeI = i;
                                break;
                            }
                        int fixedI0 = ( freeI + 1 ) % 3;
                        int fixedI1 = ( fixedI0 + 1 ) % 3;
                        t = t0;
                        const auto d = ( t[fixedI1] - t[fixedI0] ).normalized();
                        const auto c = 0.5f * ( t[fixedI1] + t[fixedI0] );
                        t[freeI] = c + d * dot( d, t0[freeI] - c );
                    }
                    else if ( numFree == 2 )
                    {
                        // only one vertex is fixed
                        int fixedI = -1;
                        for ( int i = 0; i < 3; ++i )
                            if ( !settings.region->test( vs[i] ) )
                            {
                                fixedI = i;
                                break;
                            }
                        const auto d = t0[fixedI] - t[fixedI];
                        for ( int i = 0; i < 3; ++i )
                            t[i] += d;
                        t[fixedI] = t0[fixedI]; // keep coordinates exactly
                    }
                }

                for ( int i = 0; i < 3; ++i )
                {
                    if ( points[ vs[i] ] != t[i] )
                    {
                        shiftedVerts.set( vs[i] );
                        points[ vs[i] ] = t[i];
                    }
                }
            }
            if ( anyInverted )
            {
                // move each point from degenerated triangle a little toward the center of its neighbor,
                // otherwise they will not be pushed away and the degeneracy remains forever
                MeshRelaxParams relaxParams;
                relaxParams.region = &shiftedVerts;
                relaxParams.force = 0.1f;
                relax( topology, points, relaxParams );
            }
        }
    }
}

void inflate( Mesh& mesh, const VertBitSet& verts, const InflateSettings & settings )
{
    mesh.invalidateCaches();
    inflate( mesh.topology, mesh.points, verts, settings );
}

void inflate( const MeshTopology& topology, VertCoords& points, const VertBitSet& verts, const InflateSettings & settings )
{
    MR_TIMER
    if ( !verts.any() )
        return;
    if ( settings.preSmooth )
        positionVertsSmoothlySharpBd( topology, points, verts );
    if ( settings.iterations <= 0 || settings.pressure == 0 )
        return;

    for ( int i = 0; i < settings.iterations; ++i )
    {
        const auto currPressure = settings.gradualPressureGrowth ?
            ( i + 1 ) * settings.pressure / settings.iterations : settings.pressure;
        inflate1( topology, points, verts, currPressure );
    }
}

void inflate1( const MeshTopology& topology, VertCoords& points, const VertBitSet& verts, float pressure )
{
    if ( pressure == 0 )
        return positionVertsSmoothlySharpBd( topology, points, verts );

    MR_TIMER
    auto vertShifts = dirDblAreas( topology, points, &verts );
    const double sumDblArea = parallel_deterministic_reduce( tbb::blocked_range( 0_v, vertShifts.endId(), 1024 ), 0.0,
    [&] ( const auto & range, double curr )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
            if ( verts.test( v ) )
                curr += vertShifts[v].length();
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
    if ( sumDblArea <= 0 )
        return;
    const float k = float( pressure / sumDblArea );

    BitSetParallelFor( verts, [&]( VertId v )
    {
        vertShifts[v] *= k;
    } );
    // sum( abs( vertShifts[v] ) ) = currPressure
    positionVertsSmoothlySharpBd( topology, points, verts, &vertShifts );
}

} //namespace MR
