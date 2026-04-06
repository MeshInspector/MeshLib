#include "MRSegmentMesh.h"
#include "MRMesh.h"

namespace MR
{

namespace
{

class MeshSegmenter
{
public:
    MeshSegmenter( const MeshTopology& topology, const EdgeMetric& metric );

private:
    const MeshTopology& topology_;
    const EdgeMetric& metric_;
};

MeshSegmenter::MeshSegmenter( const MeshTopology& topology, const EdgeMetric& metric )
    : topology_( topology ), metric_( metric )
{
}

} //anonymous namespace

Expected<GroupOrder> segmentMesh( const MeshTopology& topology, const EdgeMetric& metric )
{
    if ( !metric )
        return unexpected( "no metric given" );

    MeshSegmenter s( topology, metric );

    GroupOrder res;
    return res;
}

} //namespace MR
