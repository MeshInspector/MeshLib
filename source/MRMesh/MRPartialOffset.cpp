#ifndef __EMSCRIPTEN__
#include "MRPartialOffset.h"
#include "MRMesh.h"
#include "MRMeshBoolean.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

Mesh partialOffsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params /*= {} */ )
{
    auto offsetPart = offsetMesh( mp, offset, params );
    auto res = boolean( mp.mesh, offsetPart, BooleanOperation::Union );
    if ( !res.valid() )
    {
        spdlog::warn( "Partial offset failed: {}", res.errorString );
        return {};
    }
    return std::move( res.mesh );
}

} //namespace MR
#endif //!__EMSCRIPTEN__
