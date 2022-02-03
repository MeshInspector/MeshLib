#ifndef __EMSCRIPTEN__
#include "MREPartialOffset.h"
#include "MRMesh/MRMesh.h"
#include "MREMeshBoolean.h"
#include "MRPch/MRSpdlog.h"

namespace MRE
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

}
#endif
