#ifdef __APPLE__
#include "MRSpaceMouseHandler3dxMacDriver.h"
#include "MRViewer.h"

#include <MRPch/MRSpdlog.h>

#include <GLFW/glfw3.h>

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
constexpr auto kConnexionMsgPrefsChanged = 0x33645043; // '3dPC'

#pragma pack( push, 2 )
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

struct ConnexionDevicePrefs
{
    uint16_t type;
    uint16_t version;
    uint16_t deviceID;
    uint16_t reserved1;
    uint32_t appSignature;
    uint32_t reserved2;
    uint8_t appName[64];
    uint8_t mainSpeed;
    uint8_t zoomOnY;
    uint8_t dominant;
    uint8_t reserved3;
    int8_t mapV[6];
    int8_t mapH[6];
    uint8_t enabled[6];
    uint8_t reversed[6];
    uint8_t speed[6];
    uint8_t sensitivity[6];
    int32_t scale[6];
    uint32_t gamma;
    uint32_t intersect;
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

typedef int16_t (*ConnexionGetCurrentDevicePrefsFunc)( uint32_t deviceID, ConnexionDevicePrefs* prefs );

struct LibHandle
{
    void* handle{ nullptr };

#define DEFINE_SYM( name ) name##Func name{ nullptr };
    DEFINE_SYM( SetConnexionHandlers )
    DEFINE_SYM( CleanupConnexionHandlers )
    DEFINE_SYM( RegisterConnexionClient )
    DEFINE_SYM( SetConnexionClientMask )
    DEFINE_SYM( SetConnexionClientButtonMask )
    DEFINE_SYM( UnregisterConnexionClient )
    DEFINE_SYM( ConnexionClientControl )
    DEFINE_SYM( ConnexionGetCurrentDevicePrefs )
#undef DEFINE_SYM

    bool loadSymbols()
    {
#define LOAD_SYM( name ) \
        {                \
            ( name ) = (name##Func)dlsym( handle, #name ); \
            if ( ( name ) == nullptr )                     \
            {            \
                spdlog::error( "Failed to load symbol \"{}\": {}", #name, dlerror() ); \
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
        LOAD_SYM( ConnexionGetCurrentDevicePrefs )
#undef LOAD_SYM
        return true;
    }
};

std::mutex gStateMutex;
LibHandle lib;
std::unordered_set<uint16_t> gKnownClientIds;
std::unordered_map<uint32_t, ConnexionDevicePrefs> gKnownDevices;
uint32_t gButtonState{ 0 };
std::function<void(const std::string&)> gDeviceSignal;
bool gAnyStateMsg{ false };

float normalize( int16_t value )
{
    constexpr auto cAxisRange = 350.f;
    constexpr auto cThreshold = 0.01f;

    auto result = (float)value / cAxisRange;
    if ( std::abs( result ) < cThreshold )
        result = 0.f;

    return result;
}

void updateDevicePrefs( uint32_t deviceId, ConnexionDevicePrefs& prefs )
{
    lib.ConnexionGetCurrentDevicePrefs( deviceId, &prefs );

    // TODO: remove debug logging
#define PRINT_FIELD( NAME ) \
    spdlog::debug( "3DxWare driver: {} = {}", #NAME, prefs. NAME );
#define PRINT_FIELD_ARRAY( NAME ) \
    {                             \
        auto& a = prefs. NAME;    \
        spdlog::debug( "3DxWare driver: {} = [ {}, {}, {}, {}, {}, {} ]", #NAME, a[0], a[1], a[2], a[3], a[4], a[5] ); \
    }
    PRINT_FIELD( type )
    PRINT_FIELD( version )
    PRINT_FIELD( deviceID )
    PRINT_FIELD( mainSpeed )
    PRINT_FIELD( zoomOnY )
    PRINT_FIELD( dominant )
    PRINT_FIELD_ARRAY( mapV )
    PRINT_FIELD_ARRAY( mapH )
    PRINT_FIELD_ARRAY( enabled )
    PRINT_FIELD_ARRAY( reversed )
    PRINT_FIELD_ARRAY( speed )
    PRINT_FIELD_ARRAY( sensitivity )
    PRINT_FIELD_ARRAY( scale )
    PRINT_FIELD( gamma )
    PRINT_FIELD( intersect )
#undef PRINT_FIELD_ARRAY
#undef PRINT_FIELD
}

void onSpaceMouseDeviceAdded( uint32_t deviceId )
{
    std::unique_lock lock( gStateMutex );
    if ( gDeviceSignal )
        gDeviceSignal( fmt::format( "3DxWare SpaceMouseDeviceAdded: {:04x}", deviceId ) );
    updateDevicePrefs( deviceId, gKnownDevices[deviceId] );
}

void onSpaceMouseDeviceRemoved( uint32_t deviceId )
{
    std::unique_lock lock( gStateMutex );
    if ( gDeviceSignal )
        gDeviceSignal( fmt::format( "3DxWare SpaceMouseDeviceRemoved: {:04x}", deviceId ) );
    gKnownDevices.erase( deviceId );
}

void onSpaceMouseMessage( uint32_t deviceId, uint32_t type, void* arg )
{
    auto& viewer = getViewerInstance();
    if ( type == kConnexionMsgDeviceState )
    {
        std::unique_lock lock( gStateMutex );
        if ( gDeviceSignal && !gAnyStateMsg )
        {
            gDeviceSignal( "3DxWare first SpaceMouseMessage" );
            gAnyStateMsg = true;
        }

        assert( arg );
        const auto* state = (ConnexionDeviceState*)arg;
        if ( gKnownClientIds.find( state->client ) == gKnownClientIds.end() )
            return;

        auto it = gKnownDevices.find( deviceId );
        if ( it == gKnownDevices.end() )
            return;
        const auto& [_, prefs] = *it;

        // TODO: use the device preferences
        (void)prefs;

        switch ( state->command )
        {
            case kConnexionCmdHandleButtons:
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
                glfwPostEmptyEvent();
                break;

            case kConnexionCmdHandleAxis:
            {
                const Vector3f translate {
                    +normalize( state->axis[0] ),
                    -normalize( state->axis[1] ),
                    +normalize( state->axis[2] ),
                };
                const Vector3f rotate {
                    +normalize( state->axis[3] ),
                    +normalize( state->axis[4] ),
                    -normalize( state->axis[5] ),
                };
                viewer.spaceMouseMove( translate, rotate );
                glfwPostEmptyEvent();
            }
                break;
        }
    }

    else if ( type == kConnexionMsgPrefsChanged )
    {
        std::unique_lock lock( gStateMutex );
        updateDevicePrefs( deviceId, gKnownDevices[deviceId] );
    }
}

} // namespace

namespace MR
{

SpaceMouseHandler3dxMacDriver::SpaceMouseHandler3dxMacDriver()
{
    setClientName( "MeshLib" );
}

SpaceMouseHandler3dxMacDriver::~SpaceMouseHandler3dxMacDriver()
{
    std::unique_lock lock( gStateMutex );
    if ( lib.handle != nullptr )
    {
        if ( clientId_ )
        {
            lib.UnregisterConnexionClient( clientId_ );
            gKnownClientIds.erase( clientId_ );
        }
        lib.CleanupConnexionHandlers();
        dlclose( lib.handle );
        lib.handle = nullptr;
    }
    gDeviceSignal = {};
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

bool SpaceMouseHandler3dxMacDriver::initialize( std::function<void(const std::string&)> deviceSignal )
{
    // TODO: better design (e.g. `auto lib = Handle::tryLoad()`)
    std::unique_lock lock( gStateMutex );

    static constexpr const auto* c3DconnexionClientPath = "/Library/Frameworks/3DconnexionClient.framework/3DconnexionClient";
    std::error_code ec;
    if ( !std::filesystem::exists( c3DconnexionClientPath, ec ) )
    {
        spdlog::info( "3DxWare driver is not installed" );
        return false;
    }

    lib.handle = dlopen( c3DconnexionClientPath, RTLD_LAZY );
    if ( lib.handle == nullptr )
    {
        spdlog::error( "Failed to load the 3DxWare client library: {}", dlerror() );
        return false;
    }

    if ( !lib.loadSymbols() )
    {
        spdlog::error( "Failed to load the 3DxWare client library symbols" );
        dlclose( lib.handle );
        lib.handle = nullptr;
        return false;
    }

    if ( lib.SetConnexionHandlers == nullptr )
    {
        spdlog::warn( "Incompatible 3DxWare driver version; consider upgrading to version 10.2.2 or later" );
        dlclose( lib.handle );
        lib.handle = nullptr;
        return false;
    }

    gAnyStateMsg = false;
    gDeviceSignal = std::move( deviceSignal );
    lib.SetConnexionHandlers( onSpaceMouseMessage, onSpaceMouseDeviceAdded, onSpaceMouseDeviceRemoved, false );

    clientId_ = lib.RegisterConnexionClient( kConnexionClientWildcard, clientName_.get(), kConnexionClientModeTakeOver, kConnexionMaskAll );
    if ( clientId_ == 0 )
    {
        spdlog::warn( "Failed to connect to the 3DxWare driver" );
        gDeviceSignal = {};
        return false;
    }
    gKnownClientIds.emplace( clientId_ );

    lib.SetConnexionClientMask( clientId_, kConnexionMaskAll );
    lib.SetConnexionClientButtonMask( clientId_, kConnexionMaskAllButtons );

    spdlog::info( "Successfully connected to the 3DxWare driver" );
    if ( gDeviceSignal )
        gDeviceSignal( "3DxWare driver connect" );
    return true;
}

void SpaceMouseHandler3dxMacDriver::handle()
{
    // all events are processed by the 3DxWare driver; nothing to do here
}

} // namespace MR
#endif
