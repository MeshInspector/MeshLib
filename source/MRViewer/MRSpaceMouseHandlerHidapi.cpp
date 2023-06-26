#ifndef __EMSCRIPTEN__
#include "MRSpaceMouseHandlerHidapi.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"
#include "MRMesh/MRSystem.h"

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

void SpaceMouseHandlerHidapi::initialize()
{
    if ( hid_init() )
    {
        spdlog::error( "HID API: init error" );
        return;
    }
#ifdef __APPLE__
    hid_darwin_set_open_exclusive( 0 );
#endif

#ifndef NDEBUG
    hid_device_info* devs_ = hid_enumerate( 0x0, 0x0 );
    printDevices_( devs_ );
#endif
    terminateListenerThread_ = false;
    initListenerThread_();
}

bool SpaceMouseHandlerHidapi::findAndAttachDevice_()
{
    bool isDeviceFound = false;
    for ( const auto& [vendorId, supportedDevicesId] : vendor2device_ )
    {
        // search through supported vendors
        hid_device_info* localDevicesIt = hid_enumerate( vendorId, 0x0 );
        while ( localDevicesIt && !isDeviceFound )
        {
            for ( ProductId deviceId : supportedDevicesId )
            {
                if ( deviceId == localDevicesIt->product_id && localDevicesIt->usage == 8 && localDevicesIt->usage_page == 1 )
                {
                    device_ = hid_open_path( localDevicesIt->path );
                    if ( device_ )
                    {
                        isDeviceFound = true;
                        spdlog::info( "SpaceMouse Found: type: {} {} path: {} ", vendorId, deviceId, localDevicesIt->path );
                        // setup buttons logger
                        buttonsState_ = 0;
                        setButtonsMap_( vendorId, deviceId );
                        activeMouseScrollZoom_ = false;
                        break;
                    }
                    else
                    {
                        spdlog::error( "HID API: device open error" );
                    }
                }
            }
            localDevicesIt = localDevicesIt->next;
        }
        hid_free_enumeration( localDevicesIt );
    }
    return isDeviceFound;
}


void SpaceMouseHandlerHidapi::setButtonsMap_( VendorId vendorId, ProductId productId )
{
    if ( vendorId == 0x256f )
    {
        if ( productId == 0xc635 || productId == 0xc652 ) // spacemouse compact
            buttonsMapPtr_ = &buttonMapCompact;
        else if ( productId == 0xc631 || productId == 0xc632 ) //  spacemouse pro
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

    getViewerInstance().mouseController.setMouseScroll( !device_ || activeMouseScrollZoom_ );

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
        struct S
        {
            ~S() { spdlog::info( "SpaceMouse listener thread finished" ); }
        } s;

        do
        {
            std::unique_lock<std::mutex> syncThreadLock( syncThreadMutex_ );
            // stay in loop until SpaceMouse is found
            while ( !device_ )
            {
                if ( terminateListenerThread_ )
                    return;
                if ( findAndAttachDevice_() )
                    break;
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
    getViewerInstance().mouseController.setMouseScroll(  activeMouseScrollZoom );
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

void SpaceMouseHandlerHidapi::printDevices_( struct hid_device_info* cur_dev )
{
    while ( cur_dev )
    {
        if ( vendor2device_.find( cur_dev->vendor_id ) != vendor2device_.end() )
        {
            spdlog::debug( "Device Found: type: {} {} path: {} ", cur_dev->vendor_id, cur_dev->product_id, cur_dev->path );
            spdlog::debug( "{} {}", cur_dev->usage, cur_dev->usage_page );
        }
        cur_dev = cur_dev->next;
    }
    hid_free_enumeration( cur_dev );
}

}
#endif
