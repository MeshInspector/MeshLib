#ifndef __EMSCRIPTEN__
#include "MREPartialOffset.h"
#include "MRMesh/MRMesh.h"
#include "MREMeshBoolean.h"

#pragma warning(push)
#pragma warning(disable:4275)
#pragma warning(disable:4251)
#pragma warning(disable:4273)
#include <spdlog/spdlog.h>
#pragma warning(pop)

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
