#include "MRSurfacePath.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

class FlowAggregator
{
public:
    FlowAggregator( const Mesh & mesh, const VertScalars & field );

private:
    const Mesh & mesh_;
    const VertScalars & field_;
    VertMap downFlowVert_; // for each vertex stores what next vertex is on flow path (invalid vertex for local minima)
    std::vector<VertId> vertsSortedDesc_; // all vertices sorted in descending field order
};

FlowAggregator::FlowAggregator( const Mesh & mesh, const VertScalars & field ) : mesh_( mesh ), field_( field )
{
    MR_TIMER
    downFlowVert_.resize( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        VertId nextVert;
        computeSteepestDescentPath( mesh, field, MeshTriPoint( mesh.topology, v ), {}, nullptr, &nextVert );
        downFlowVert_[v] = nextVert;
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

} //namespace MR
