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
#include <MRPch/MREigenSparseCore.h>
#include <Eigen/SparseCholesky>
#include <array>

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
    MR_TIMER;

    Laplacian laplacian( topology, points );
    laplacian.init( verts, edgeWeights, vmass, RememberShape::No );
    if ( fixedSharpVertices )
        for ( auto v : *fixedSharpVertices )
            laplacian.fixVertex( v, false );
    laplacian.apply();
}

void positionVertsSmoothlySharpBd( Mesh& mesh, const PositionVertsSmoothlyParams& params )
{
    mesh.invalidateCaches();
    positionVertsSmoothlySharpBd( mesh.topology, mesh.points, params );
}

namespace
{

/// finds N scalar fields on region vertices, where each value is the weighted mean of the values in neighbor vertices;
/// getVal(v) returns the fixed values of not-region vertex or the original values of region vertex (for stabilization),
/// getShift(v) returns additional shifts of the equation of region vertex v, setVal(v, vals) stores the solution
template<size_t N, class GetVal, class GetShift, class SetVal>
void solveLaplaceEquations( const MeshTopology& topology, const VertBitSet& verts,
    float stabilizer, const VertMetric& vertStabilizers, const UndirectedEdgeMetric& edgeWeights,
    GetVal getVal, GetShift getShift, SetVal setVal )
{
    MR_TIMER;
    const auto sz = verts.count();
    if ( sz <= 0 )
        return;

    // vertex id -> position in the matrix
    HashMap<VertId, int> vertToMatPos = makeHashMapWithSeqNums( verts );

    using Vals = std::array<double, N>;
    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[N];
    for ( auto & r : rhs )
        r.resize( sz );
    int n = 0;
    for ( auto v : verts )
    {
        double sumW = 0;
        Vals sumFixed{};
        auto addFixed = [&sumFixed]( double w, const Vals & vals )
        {
            for ( size_t i = 0; i < N; ++i )
                sumFixed[i] += w * vals[i];
        };
        for ( auto e : orgRing( topology, v ) )
        {
            const double edgeW = edgeWeights ? edgeWeights( e ) : 1;
            sumW += edgeW;
            auto d = topology.dest( e );
            if ( auto it = vertToMatPos.find( d ); it != vertToMatPos.end() )
            {
                // free neighbor
                int di = it->second;
                if ( n > di ) // row > col: fill only lower left part of matrix
                    mTriplets.emplace_back( n, di, -edgeW );
            }
            else
            {
                // fixed neighbor
                addFixed( edgeW, getVal( d ) );
            }
        }
        addFixed( sumW, getShift( v ) );
        double s = stabilizer; //for VertexMass::Unit only
        if ( vertStabilizers )
        {
            s = vertStabilizers( v );
            assert( s >= 0 );
        }
        if ( s != 0 )
        {
            sumW += s;
            addFixed( s, getVal( v ) );
        }
        mTriplets.emplace_back( n, n, sumW );
        for ( size_t i = 0; i < N; ++i )
            rhs[i][n] = sumFixed[i];
        ++n;
    }

    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SparseMatrix A;
    A.resize( sz, sz );
    A.setFromTriplets( mTriplets.begin(), mTriplets.end() );
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute( A );

    Eigen::VectorXd sol[N];
    ParallelFor( 0, (int)N, [&]( int i )
    {
        sol[i] = solver.solve( rhs[i] );
    } );

    n = 0;
    for ( auto v : verts )
    {
        Vals vals;
        for ( size_t i = 0; i < N; ++i )
            vals[i] = sol[i][n];
        setVal( v, vals );
        ++n;
    }
}

} // anonymous namespace

void positionVertsSmoothlySharpBd( const MeshTopology& topology, VertCoords& points, const PositionVertsSmoothlyParams& params )
{
    MR_TIMER;
    assert( params.stabilizer > 0 || params.vertStabilizers || ( params.region && !MeshComponents::hasFullySelectedComponent( topology, *params.region ) ) );

    solveLaplaceEquations<3>( topology, topology.getVertIds( params.region ), params.stabilizer, params.vertStabilizers, params.edgeWeights,
        [&points]( VertId v ) { const auto & p = points[v]; return std::array<double, 3>{ p.x, p.y, p.z }; },
        [&params]( VertId v )
        {
            std::array<double, 3> res{};
            if ( params.vertShifts )
            {
                const auto & p = (*params.vertShifts)[v];
                res = { p.x, p.y, p.z };
            }
            return res;
        },
        [&points]( VertId v, const std::array<double, 3> & vals ) { points[v] = Vector3f( (float)vals[0], (float)vals[1], (float)vals[2] ); } );
}

void interpolateScalarsSmoothly( const MeshTopology& topology, VertScalars& field, const InterpolateScalarsParams& params )
{
    MR_TIMER;
    assert( params.stabilizer > 0 || params.vertStabilizers || ( params.region && !MeshComponents::hasFullySelectedComponent( topology, *params.region ) ) );

    solveLaplaceEquations<1>( topology, topology.getVertIds( params.region ), params.stabilizer, params.vertStabilizers, params.edgeWeights,
        [&field]( VertId v ) { return std::array<double, 1>{ field[v] }; },
        []( VertId ) { return std::array<double, 1>{}; },
        [&field]( VertId v, const std::array<double, 1> & vals ) { field[v] = (float)vals[0]; } );
}

void positionVertsWithSpacing( Mesh& mesh, const SpacingSettings & settings )
{
    mesh.invalidateCaches();
    positionVertsWithSpacing( mesh.topology, mesh.points, settings );
}

void positionVertsWithSpacing( const MeshTopology& topology, VertCoords& points, const SpacingSettings & settings )
{
    MR_TIMER;
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

void positionVertsSmoothlySharpBd( Mesh& mesh, const VertBitSet& verts )
{
    positionVertsSmoothlySharpBd( mesh, { .region = &verts } );
}

void inflate( Mesh& mesh, const VertBitSet& verts, const InflateSettings & settings )
{
    mesh.invalidateCaches();
    inflate( mesh.topology, mesh.points, verts, settings );
}

void inflate( const MeshTopology& topology, VertCoords& points, const VertBitSet& verts, const InflateSettings & settings )
{
    MR_TIMER;
    if ( !verts.any() )
        return;
    if ( settings.preSmooth )
        positionVertsSmoothlySharpBd( topology, points, { .region = &verts } );
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
        return positionVertsSmoothlySharpBd( topology, points, { .region = &verts } );

    MR_TIMER;
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
    positionVertsSmoothlySharpBd( topology, points, { .region = &verts, .vertShifts = &vertShifts } );
}

} //namespace MR
