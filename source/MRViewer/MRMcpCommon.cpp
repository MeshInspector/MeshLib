#ifndef MESHLIB_NO_MCP

#include "MRMcpCommon.h"

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

std::runtime_error busyError( std::string_view what )
{
    return std::runtime_error( fmt::format( "{}. If MeshInspector is still running an action started by an earlier call, "
        "poll `ui_progressStatus` until `busy` is false, then retry.", what ) );
}

void runOnMainThread( const std::function<void()>& func )
{
    bool started = false;
    try
    {
        CommandLoop::runCommandFromGUIThread( [&] { started = true; func(); }, cMainThreadTimeout );
    }
    catch ( const std::runtime_error& e )
    {
        if ( started ) // an error of `func` itself
            throw;
        throw busyError( e.what() );
    }
}

std::string waitFramesAfterInput()
{
    const auto deadline = std::chrono::steady_clock::now() + cMainThreadTimeout;
    for ( int i = 0; i < getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
    {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>( deadline - std::chrono::steady_clock::now() );
        try
        {
            CommandLoop::runCommandFromGUIThread( [] {}, std::max( left, std::chrono::milliseconds( 0 ) ) );
        }
        catch ( const std::runtime_error& )
        {
            return fmt::format( "The input was delivered, but MeshInspector's main thread is still busy after {} seconds, "
                "probably with the action the input started. Poll `ui_progressStatus` until `busy` is false before the next call.",
                cMainThreadTimeout.count() );
        }
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
