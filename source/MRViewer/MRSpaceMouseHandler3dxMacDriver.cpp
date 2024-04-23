#ifdef __APPLE__
#include "MRSpaceMouseHandler3dxMacDriver.h"
#include "MRViewer.h"

#include <MRPch/MRSpdlog.h>

#include <dlfcn.h>

#include <unordered_set>

namespace
{

using namespace MR;

constexpr auto kConnexionClientWildcard = 0x2A2A2A2A; // '****'
constexpr auto kConnexionClientModeTakeOver = 1;
constexpr auto kConnexionMaskAll = 0x3FFF;
constexpr auto kConnexionMaskAllButtons = 0xFFFFFFFF;
constexpr auto kConnexionCmdHandleButtons = 2;
constexpr auto kConnexionCmdHandleAxis = 3;
constexpr auto kConnexionMsgDeviceState = 0x33645352; // '3dSR'

#pragma pack( push, 1 )
struct ConnexionDeviceState
{
    uint16_t version;
    uint16_t client;
    uint16_t command;
    int16_t param;
    int32_t value;
    uint64_t time;
    uint8_t report[8];
    uint16_t buttons8;
    int16_t axis[6];
    uint16_t address;
    uint32_t buttons;
};
#pragma pack( pop )

typedef void (*ConnexionAddedHandlerProc)( unsigned int productID );
typedef void (*ConnexionRemovedHandlerProc)( unsigned int productID );
typedef void (*ConnexionMessageHandlerProc)( unsigned int productID, unsigned int messageType, void* messageArgument );
typedef int16_t (*SetConnexionHandlersFunc)( ConnexionMessageHandlerProc messageHandler, ConnexionAddedHandlerProc addedHandler, ConnexionRemovedHandlerProc removedHandler, bool useSeparateThread );
typedef void (*CleanupConnexionHandlersFunc)();

typedef uint16_t (*RegisterConnexionClientFunc)( uint32_t signature, uint8_t* name, uint16_t mode, uint32_t mask );
typedef void (*SetConnexionClientMaskFunc)( uint16_t clientID, uint32_t mask );
typedef void (*SetConnexionClientButtonMaskFunc)( uint16_t clientID, uint32_t buttonMask );
typedef void (*UnregisterConnexionClientFunc)( uint16_t clientID );

typedef int16_t (*ConnexionClientControlFunc)( uint16_t clientID, uint32_t message, int32_t param, int32_t* result );

// TODO: thread safety?
std::unordered_set<uint16_t> gKnownClientIds;
uint32_t gButtonState{ 0 };

void onSpaceMouseMessage( uint32_t, uint32_t type, void* arg )
{
    auto& viewer = getViewerInstance();
    if ( type == kConnexionMsgDeviceState )
    {
        assert( arg );
        const auto* state = (ConnexionDeviceState*)arg;
        if ( gKnownClientIds.find( state->client ) == gKnownClientIds.end() )
            return;

        switch ( state->command )
        {
            case kConnexionCmdHandleButtons:
                // TODO: thread safety?
                for ( auto btn = 0; btn < 32; ++btn )
                {
                    const auto mask = 1 << btn;
                    if ( ( state->buttons & mask ) != ( gButtonState & mask ) )
                    {
                        if ( state->buttons & mask )
                            viewer.spaceMouseDown( btn );
                        else
                            viewer.spaceMouseUp( btn );
                    }
                }
                gButtonState = state->buttons;
                break;

            case kConnexionCmdHandleAxis:
            {
                // TODO: correct mapping
                const Vector3f translate {
                    (float)state->axis[0],
                    (float)state->axis[1],
                    (float)state->axis[2],
                };
                const Vector3f rotate {
                    (float)state->axis[3],
                    (float)state->axis[4],
                    (float)state->axis[5],
                };
                viewer.spaceMouseMove( translate, rotate );
            }
                break;
        }
    }
}

} // namespace

namespace MR
{

struct SpaceMouseHandler3dxMacDriver::LibHandle
{
    void* handle;

#define DEFINE_SYM( name ) name##Func name;
    DEFINE_SYM( SetConnexionHandlers )
    DEFINE_SYM( CleanupConnexionHandlers )
    DEFINE_SYM( RegisterConnexionClient )
    DEFINE_SYM( SetConnexionClientMask )
    DEFINE_SYM( SetConnexionClientButtonMask )
    DEFINE_SYM( UnregisterConnexionClient )
    DEFINE_SYM( ConnexionClientControl )
#undef DEFINE_SYM

    bool loadSymbols()
    {
#define LOAD_SYM( name ) \
        {                \
            ( name ) = (name##Func)dlsym( handle, #name ); \
            if ( ( name ) == nullptr )                     \
            {            \
                spdlog::warn( "Failed to load symbol \"{}\": {}", #name, dlerror() ); \
                return false;                              \
            }            \
        }
        LOAD_SYM( SetConnexionHandlers )
        LOAD_SYM( CleanupConnexionHandlers )
        LOAD_SYM( RegisterConnexionClient )
        LOAD_SYM( SetConnexionClientMask )
        LOAD_SYM( SetConnexionClientButtonMask )
        LOAD_SYM( UnregisterConnexionClient )
        LOAD_SYM( ConnexionClientControl )
#undef LOAD_SYM
        return true;
    }
};

SpaceMouseHandler3dxMacDriver::SpaceMouseHandler3dxMacDriver()
    : lib_( std::make_unique<LibHandle>() )
{
    setClientName( "MeshLib" );
}

SpaceMouseHandler3dxMacDriver::~SpaceMouseHandler3dxMacDriver()
{
    if ( lib_->handle != nullptr )
    {
        if ( clientId_ )
        {
            lib_->UnregisterConnexionClient( clientId_ );
            // TODO: thread safety?
            gKnownClientIds.erase( clientId_ );
        }
        lib_->CleanupConnexionHandlers();
        dlclose( lib_->handle );
    }
}

void SpaceMouseHandler3dxMacDriver::setClientName( const char* name, size_t len )
{
    if ( len == 0 )
        len = std::strlen( name );
    assert( len <= 254 );
    // the name must be converted to the Pascal string format
    clientName_ = std::make_unique<uint8_t[]>( len + 1 );
    clientName_[0] = (uint8_t)len;
    std::memcpy( clientName_.get() + 1, (const uint8_t *)name, len );
}

bool SpaceMouseHandler3dxMacDriver::initialize()
{
    // TODO: better design (e.g. `auto lib = Handle::tryLoad()`)

    static constexpr const auto* c3DconnexionClientPath = "/Library/Frameworks/3DconnexionClient.framework/3DconnexionClient";
    std::error_code ec;
    if ( !std::filesystem::exists( c3DconnexionClientPath, ec ) )
    {
        spdlog::warn( "3DxWare driver is not installed" );
        return false;
    }

    lib_->handle = dlopen( c3DconnexionClientPath, RTLD_LAZY );
    if ( lib_->handle == nullptr )
    {
        spdlog::warn( "Failed to load client library" );
        return false;
    }

    if ( !lib_->loadSymbols() )
    {
        dlclose( lib_->handle );
        lib_->handle = nullptr;
        return false;
    }

    lib_->SetConnexionHandlers( onSpaceMouseMessage, nullptr, nullptr, false );

    clientId_ = lib_->RegisterConnexionClient( kConnexionClientWildcard, clientName_.get(), kConnexionClientModeTakeOver, kConnexionMaskAll );
    if ( clientId_ == 0 )
    {
        spdlog::warn( "Failed to connect to the driver" );
        return false;
    }
    // TODO: thread safety?
    gKnownClientIds.emplace( clientId_ );

    lib_->SetConnexionClientMask( clientId_, kConnexionMaskAll );
    lib_->SetConnexionClientButtonMask( clientId_, kConnexionMaskAllButtons );

    return true;
}

void SpaceMouseHandler3dxMacDriver::handle()
{
    // nothing to do here
}

void SpaceMouseHandler3dxMacDriver::postFocus_( bool )
{
    // ...
}

} // namespace MR
#endif
