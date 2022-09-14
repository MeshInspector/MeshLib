#ifndef MRMESH_NO_VOXEL
#include "MRPartialOffset.h"
#include "MRMesh.h"
#include "MRMeshBoolean.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

tl::expected<Mesh, std::string> partialOffsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params /*= {} */ )
{
    auto offsetPart = offsetMesh( mp, offset, params );
    if ( !offsetPart.has_value() )
        return offsetPart;
    auto res = boolean( mp.mesh, *offsetPart, BooleanOperation::Union );
    if ( !res.valid() )
    {
        spdlog::warn( "Partial offset failed: {}", res.errorString );
        return {};
    }
    return std::move( res.mesh );
}

} //namespace MR
#endif //!__EMSCRIPTEN__
