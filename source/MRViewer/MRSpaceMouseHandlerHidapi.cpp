#ifndef __EMSCRIPTEN__
#include "MRSpaceMouseHandlerHidapi.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"
#include "MRMouseController.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{
SpaceMouseHandlerHidapi::SpaceMouseHandlerHidapi()
    : device_( nullptr )
    , buttonsMapPtr_( nullptr )
    , terminateListenerThread_( false )
    , dataPacket_( { 0 } )
    , packetLength_( 0 )
    , active_( true )
    , activeMouseScrollZoom_( true )
{
    connect( &getViewerInstance(), 0, boost::signals2::connect_position::at_back );
}

SpaceMouseHandlerHidapi::~SpaceMouseHandlerHidapi()
{
    terminateListenerThread_ = true;
    cv_.notify_one();

    if ( listenerThread_.joinable() )
        listenerThread_.join();

    if ( device_ != nullptr )
        hid_close( device_ );

    hid_exit();
}

bool SpaceMouseHandlerHidapi::initialize( std::function<void(const std::string&)> deviceSignal )
{
    deviceSignal_ = std::move( deviceSignal );
    if ( hid_init() != 0 )
    {
        spdlog::error( "HID API: init error" );
        return false;
    }

#ifdef __APPLE__
    hid_darwin_set_open_exclusive( 0 );
#endif

    terminateListenerThread_ = false;
    initListenerThread_();

    return true;
}

bool SpaceMouseHandlerHidapi::findAndAttachDevice_( bool verbose )
{
    const static int HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER = 8; //Multi-axis Controller
    const static int HID_USAGE_PAGE_GENERIC = 1; //Generic Desktop Controls
    assert( !device_ );
    for ( const auto& [vendorId, supportedDevicesId] : vendor2device_ )
    {
        // search through supported vendors
        hid_device_info* localDevicesIt = hid_enumerate( vendorId, 0x0 ); // hid_enumerate( 0x0, 0x0 ) to enumerate all devices of all vendors
        while ( localDevicesIt && ( !device_ || verbose ) )
        {
            if ( verbose )
            {
                spdlog::info( "HID API device found: {:04x}:{:04x}, path={}, usage={}, usage_page={}",
                    vendorId, localDevicesIt->product_id, localDevicesIt->path, localDevicesIt->usage, localDevicesIt->usage_page );
                if ( deviceSignal_ && localDevicesIt->usage == HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER && localDevicesIt->usage_page == HID_USAGE_PAGE_GENERIC )
                    deviceSignal_( fmt::format( "HID API device {:04x}:{:04x} found", vendorId, localDevicesIt->product_id ) );
            }
            for ( ProductId deviceId : supportedDevicesId )
            {
                if ( !device_ && deviceId == localDevicesIt->product_id && localDevicesIt->usage == HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER && localDevicesIt->usage_page == HID_USAGE_PAGE_GENERIC )
                {
                    device_ = hid_open_path( localDevicesIt->path );
                    if ( device_ )
                    {
                        anyAction_ = false;
                        spdlog::info( "SpaceMouse connected: {:04x}:{:04x}, path={}", vendorId, deviceId, localDevicesIt->path );
                        if ( deviceSignal_ )
                            deviceSignal_( fmt::format( "HID API device {:04x}:{:04x} opened", vendorId, localDevicesIt->product_id ) );
                        // setup buttons logger
                        buttonsState_ = 0;
                        setButtonsMap_( vendorId, deviceId );
                        activeMouseScrollZoom_ = false;
                        if ( !verbose )
                            break;
                    }
                    else if ( verbose )
                    {
                        spdlog::error( "HID API device ({:04x}:{:04x}, path={}) open error: {}",
                            vendorId, deviceId, localDevicesIt->path, wideToUtf8( hid_error( nullptr ) ) );
                        if ( deviceSignal_ )
                            deviceSignal_( fmt::format( "HID API device {:04x}:{:04x} open failed", vendorId, localDevicesIt->product_id ) );
                    }
                }
            }
            localDevicesIt = localDevicesIt->next;
        }
        hid_free_enumeration( localDevicesIt );
    }
    return (bool)device_;
}


void SpaceMouseHandlerHidapi::setButtonsMap_( VendorId vendorId, ProductId productId )
{
    if ( vendorId == 0x256f )
    {
        if ( productId == 0xc635 || productId == 0xc652 ) // spacemouse compact
            buttonsMapPtr_ = &buttonMapCompact;
        else if ( productId == 0xc631 || productId == 0xc632 || productId == 0xc638 ) //  spacemouse pro
            buttonsMapPtr_ = &buttonMapPro;
        else if ( productId == 0xc633 ) // spacemouse enterprise
            buttonsMapPtr_ = &buttonMapEnterprise;
    }
    else if ( vendorId == 0x046d )
    {
        if ( productId == 0xc62b ) //  spacemouse pro
            buttonsMapPtr_ = &buttonMapPro;
    }
}

void SpaceMouseHandlerHidapi::handle()
{
    // works in pair with SpaceMouseHandlerHidapi::startListenerThread_()
    std::unique_lock<std::mutex> syncThreadLock( syncThreadMutex_, std::defer_lock );
    if ( !syncThreadLock.try_lock() )
        return;

    getViewerInstance().mouseController().setMouseScroll( !device_ || activeMouseScrollZoom_ );

    if ( packetLength_ <= 0 || !device_ )
    {
        cv_.notify_one();
        return;
    }

    // set the device handle to be non-blocking
    hid_set_nonblocking( device_, 1 );

    SpaceMouseAction action;
    updateActionWithInput_( dataPacket_, packetLength_, action );

    int packetLengthTmp = 0;
    do
    {
        DataPacketRaw dataPacketTmp;
        packetLengthTmp = hid_read( device_, dataPacketTmp.data(), dataPacketTmp.size() );
        updateActionWithInput_( dataPacketTmp, packetLengthTmp, action );
    } while ( packetLengthTmp > 0 );

    processAction_( action );

    syncThreadLock.unlock();
    cv_.notify_one();
}

void SpaceMouseHandlerHidapi::initListenerThread_()
{
    // works in pair with SpaceMouseHandlerHidapi::handle()
    // waits for updates on SpaceMouse and notifies main thread
    listenerThread_ = std::thread( [&] ()
    {
        spdlog::info( "SpaceMouse Listener thread started" );
        SetCurrentThreadName( "SpaceMouse listener" );
        MR_FINALLY {
            spdlog::info( "SpaceMouse listener thread finished" );
        };

        do
        {
            std::unique_lock<std::mutex> syncThreadLock( syncThreadMutex_ );
            // stay in loop until SpaceMouse is found
            bool firstSearch = true;
            while ( !device_ )
            {
                if ( terminateListenerThread_ )
                    return;
                if ( findAndAttachDevice_( firstSearch ) )
                    break;
                firstSearch = false; // avoid spam in log
                syncThreadLock.unlock();
                std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );
                syncThreadLock.lock();
            }

            // set the device handle to be blocking
            hid_set_nonblocking( device_, 0 );
            // wait for active state and read all data packets during inactive state
            while ( !active_ )
            {
                do
                {
                    packetLength_ = hid_read_timeout( device_, dataPacket_.data(), dataPacket_.size(), 200 );
                } while ( packetLength_ > 0 && !active_ && !terminateListenerThread_ );
                if ( terminateListenerThread_ )
                    return;
            }

            // hid_read_timeout() waits until there is data to read before returning or 1000ms passed (to help with thread shutdown)
            packetLength_ = hid_read_timeout( device_, dataPacket_.data(), dataPacket_.size(), 1000 );

            // device connection lost
            if ( packetLength_ < 0 )
            {
                hid_close( device_ );
                device_ = nullptr;
                buttonsMapPtr_ = nullptr;
                buttonsState_ = 0;
                activeMouseScrollZoom_ =  true;
                spdlog::error( "HID API: device lost" );
                if ( deviceSignal_ )
                    deviceSignal_( fmt::format( "HID API device lost" ) );
            }
            else if ( packetLength_ > 0 )
            {
                // trigger main rendering loop and wait for main thread to read and process all SpaceMouse packets
                glfwPostEmptyEvent();
                cv_.wait( syncThreadLock );
            }
            // nothing to do with packet_length == 0
        } while ( !terminateListenerThread_ );
    } );
}

void SpaceMouseHandlerHidapi::postFocus_( bool focused )
{
    active_ = focused;
    cv_.notify_one();
}

void SpaceMouseHandlerHidapi::activateMouseScrollZoom( bool activeMouseScrollZoom )
{
    activeMouseScrollZoom_ = activeMouseScrollZoom;
    getViewerInstance().mouseController().setMouseScroll(  activeMouseScrollZoom );
}


float SpaceMouseHandlerHidapi::convertCoord_( int coord_byte_low, int coord_byte_high )
{
    int value = coord_byte_low | ( coord_byte_high << 8 );
    if ( value > SHRT_MAX )
    {
        value = value - 65536;
    }
    float ret = ( float )value / 350.0f;
    return ( std::abs( ret ) > 0.01f ) ? ret : 0.0f;
}

void SpaceMouseHandlerHidapi::updateActionWithInput_( const DataPacketRaw& packet, int packet_length, SpaceMouseAction& action )
{
    // button update package
    if ( packet[0] == 3 && buttonsMapPtr_ != nullptr )
    {
        action.isButtonStateChanged = true;
        // for all bytes in packet
        for ( int column = 1; column < buttonsMapPtr_->size(); ++column )
        {
            for ( int i = 0; i < ( *buttonsMapPtr_ )[column].size(); ++i )
            {
                if ( packet[column] & ( 1 << i ) )
                    action.buttons.set( ( *buttonsMapPtr_ )[column][i] );
            }
        }
        return;
    }

    Vector3f matrix = { 0.0f, 0.0f, 0.0f };
    if ( packet_length >= 7 )
    {
        matrix = { convertCoord_( packet[1], packet[2] ),
                  convertCoord_( packet[3], packet[4] ),
                  convertCoord_( packet[5], packet[6] ) };

        if ( packet[0] == 1 )
            action.translate = matrix;
        else if ( packet[0] == 2 )
            action.rotate = matrix;
    }
    if ( packet_length == 13 )
    {
        action.translate = matrix;
        action.rotate = { convertCoord_( packet[7], packet[8] ),
                         convertCoord_( packet[9], packet[10] ),
                         convertCoord_( packet[11], packet[12] ) };
    }
}

void SpaceMouseHandlerHidapi::processAction_( const SpaceMouseAction& action )
{
    if ( deviceSignal_ && !anyAction_ )
    {
        deviceSignal_( "HID API first action processing" );
        anyAction_ = true;
    }
    auto& viewer = getViewerInstance();
    viewer.spaceMouseMove( action.translate, action.rotate );
    glfwPostEmptyEvent();

    if ( action.isButtonStateChanged  )
     {
         std::bitset<SMB_BUTTON_COUNT> new_pressed = action.buttons & ~buttonsState_;
         std::bitset<SMB_BUTTON_COUNT> new_unpressed = buttonsState_ & ~action.buttons;
         for (int btn = 0; btn < SMB_BUTTON_COUNT; ++btn)
         {
             if ( new_unpressed.test( btn ) )
                 viewer.spaceMouseUp( btn );
             if ( new_pressed.test( btn ) )
                 viewer.spaceMouseDown( btn );
         }
         buttonsState_ = action.buttons;
     }
}

}
#endif
