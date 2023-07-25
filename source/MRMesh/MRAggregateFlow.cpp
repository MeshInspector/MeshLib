#include "MRSurfacePath.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

struct FlowOrigin
{
    /// point on the mesh, where this flow starts
    MeshTriPoint point;
    /// amount of flow, e.g. can be proportional to the horizontal area associated with the start point
    float amount = 1;
};

class FlowAggregator
{
public:
    FlowAggregator( const Mesh & mesh, const VertScalars & field );
    VertScalars computeFlow( const std::vector<FlowOrigin> & starts, Polyline3 * outPolyline = nullptr );

private:
    const Mesh & mesh_;
    const VertScalars & field_;
    VertMap downFlowVert_; // for each vertex stores what next vertex is on flow path (invalid vertex for local minima)
    Vector<SurfacePath, VertId> downPath_; // till next vertex
    std::vector<VertId> vertsSortedDesc_; // all vertices sorted in descending field order
};

FlowAggregator::FlowAggregator( const Mesh & mesh, const VertScalars & field ) : mesh_( mesh ), field_( field )
{
    MR_TIMER
    downFlowVert_.resize( mesh.topology.vertSize() );
    downPath_.resize( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        const EdgePoint p0( mesh.topology, v );
        SurfacePath path;
        VertId nextVert;
        computeSteepestDescentPath( mesh, field, p0, {}, &path, &nextVert );
        downFlowVert_[v] = nextVert;
        if ( nextVert )
            downPath_[v] = std::move( path );
    } );

    using MinusHeightVert = std::pair<float, VertId>;
    std::vector<MinusHeightVert> minusHeightVerts;
    minusHeightVerts.reserve( mesh.topology.numValidVerts() );
    for ( auto v : mesh.topology.getValidVerts() )
        minusHeightVerts.push_back( { -field[v], v } );
    tbb::parallel_sort( minusHeightVerts.begin(), minusHeightVerts.end() );

    vertsSortedDesc_.reserve( minusHeightVerts.size() );
    for ( size_t i = 0; i < minusHeightVerts.size(); ++i )
        vertsSortedDesc_.push_back( minusHeightVerts[i].second );
}

VertScalars FlowAggregator::computeFlow( const std::vector<FlowOrigin> & starts, Polyline3 * outPolyline )
{
    MR_TIMER
    VertScalars flowInVert( mesh_.topology.vertSize() );
    std::vector<VertId> start2downVert( starts.size() ); // for each start point stores what next vertex is on flow path (can be invalid)
    std::vector<SurfacePath> start2downPath( starts.size() ); // till next vertex

    ParallelFor( starts, [&]( size_t i )
    {
        VertId nextVert;
        SurfacePath path = computeSteepestDescentPath( mesh_, field_, starts[i].point, {}, &nextVert );
        start2downVert[i] = nextVert;
        if ( nextVert )
            start2downPath[i] = std::move( path );
    } );

    for ( size_t i = 0; i < starts.size(); ++i )
    {
        MeshTriPoint end;
        if ( auto v = start2downVert[i] )
        {
            flowInVert[v] += starts[i].amount;
            end = MeshTriPoint{ mesh_.topology, v };
        }
        if ( outPolyline )
            outPolyline->addFromGeneralSurfacePath( mesh_, starts[i].point, start2downPath[i], end );
    }

    for ( size_t i = 0; i < vertsSortedDesc_.size(); ++i )
    {
        auto vUp = vertsSortedDesc_[i];
        if ( !flowInVert[vUp] )
            continue;
        MeshTriPoint end;
        if ( auto vDn = downFlowVert_[vUp] )
        {
            flowInVert[vDn] += flowInVert[vUp];
            end = MeshTriPoint{ mesh_.topology, vDn };
        }
        if ( outPolyline )
            outPolyline->addFromGeneralSurfacePath( mesh_, { mesh_.topology, vUp }, downPath_[vUp], end );
    }

    return flowInVert;
}

} //namespace MR
