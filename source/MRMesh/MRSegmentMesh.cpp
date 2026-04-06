#include "MRSegmentMesh.h"
#include "MRMesh.h"

namespace MR
{

namespace
{

class MeshSegmenter
{
public:
    MeshSegmenter( const Mesh& mesh, const EdgeMetric& metric );

private:
    const Mesh& mesh_;
    const EdgeMetric& metric_;
};

MeshSegmenter::MeshSegmenter( const Mesh& mesh, const EdgeMetric& metric )
    : mesh_( mesh ), metric_( metric )
{
}

} //anonymous namespace

Expected<GroupOrder> segmentMesh( const Mesh& mesh, const EdgeMetric& metric )
{
    if ( !metric )
        return unexpected( "no metric given" );

    MeshSegmenter s( mesh, metric );

    GroupOrder res;
    return res;
}

} //namespace MR
