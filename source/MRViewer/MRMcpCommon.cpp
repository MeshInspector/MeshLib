#ifndef MESHLIB_NO_MCP

#include "MRMcpCommon.h"

#include "MRMesh/MRColor.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRViewer.h"

#include <algorithm>

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
    return unexpected( fmt::format( "No object with id {}. Call scene_listObjectTree to enumerate.", id ) );
}

Color parseColor( const nlohmann::json& j, std::string_view field )
{
    if ( !j.is_array() || ( j.size() != 3 && j.size() != 4 ) )
        throw std::runtime_error( fmt::format( "`{}` must be a 3- or 4-element array of floats in [0,1]", field ) );
    const float r = j[0].get<float>();
    const float g = j[1].get<float>();
    const float b = j[2].get<float>();
    const float a = j.size() == 4 ? j[3].get<float>() : 1.0f;
    return Color{ r, g, b, a }; // the float ctor clamps to [0, 1]
}

std::runtime_error busyError( std::string_view what )
{
    return std::runtime_error( fmt::format( "{}. If MeshInspector is still running an action started by an earlier call, "
        "wait with `ui_progressStatus {{waitSeconds: 20}}` until `busy` is false, then retry.", what ) );
}

void runOnMainThread( const std::function<void()>& func )
{
    try
    {
        CommandLoop::runCommandFromGUIThread( func, cMainThreadTimeout );
    }
    catch ( const CommandLoop::StartTimeout& e )
    {
        throw busyError( e.what() );
    }
}

std::string waitFramesAfterInput( const std::function<void()>& after )
{
    const auto deadline = std::chrono::steady_clock::now() + cMainThreadTimeout;
    auto run = [&] ( const std::function<void()>& func )
    {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>( deadline - std::chrono::steady_clock::now() );
        CommandLoop::runCommandFromGUIThread( func, std::max( left, std::chrono::milliseconds( 0 ) ) );
    };
    try
    {
        for ( int i = 0; i < getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
            run( [] {} );
        if ( after )
            run( after );
    }
    catch ( const CommandLoop::StartTimeout& )
    {
        return fmt::format( "The input was delivered, but MeshInspector's main thread is still busy after {} seconds, "
            "probably with the action the input started. Wait with `ui_progressStatus {{waitSeconds: 20}}` until `busy` is false "
            "before the next call.",
            cMainThreadTimeout.count() );
    }
    return {};
}

nlohmann::json inputResult( std::string busy )
{
    if ( busy.empty() )
        return nlohmann::json::object();
    return nlohmann::json::object( { { "busy", std::move( busy ) } } );
}

} // namespace MR::Mcp

#endif
