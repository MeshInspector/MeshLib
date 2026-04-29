#ifndef MESHLIB_NO_MCP

#include "MRMcpCommon.h"

#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRPch/MRFmt.h"

namespace MR::Mcp
{

uint64_t idOf( const Object* obj )
{
    return static_cast<uint64_t>( reinterpret_cast<uintptr_t>( obj ) );
}

Expected<std::shared_ptr<Object>> resolveId( uint64_t id )
{
    auto* asPtr = reinterpret_cast<Object*>( static_cast<uintptr_t>( id ) );
    if ( !asPtr )
        return unexpected( "Object id 0 is not valid here." );

    for ( const auto& obj : getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selectable ) )
    {
        if ( obj.get() == asPtr )
            return obj;
    }
    return unexpected( fmt::format( "No object with id {}. Call scene.listObjectTree to enumerate.", id ) );
}

} // namespace MR::Mcp

#endif
