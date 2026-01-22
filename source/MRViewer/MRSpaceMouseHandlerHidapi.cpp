#ifndef __EMSCRIPTEN__
#include "MRSpaceMouseHandlerHidapi.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"
#include "MRMouseController.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include <bit>

namespace MR::SpaceMouse
{
SpaceMouseHandlerHidapi::SpaceMouseHandlerHidapi()
    : device_( nullptr )
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
    for ( const auto& [vendorId, supportedDevicesId] : cVendor2Device )
    {
        // search through supported vendors
        hid_device_info* localDevicesIt = hid_enumerate( vendorId, 0x0 ); // hid_enumerate( 0x0, 0x0 ) to enumerate all devices of all vendors
        while ( localDevicesIt && ( !device_ || verbose ) )
        {
            if ( verbose )
            {
                spdlog::info( "HID API device found: {:04x}:{:04x}, path={}, usage={}, usage_page={}, name={}:{}",
                    vendorId, localDevicesIt->product_id, localDevicesIt->path, localDevicesIt->usage, localDevicesIt->usage_page,
                    wideToUtf8( localDevicesIt->manufacturer_string ), wideToUtf8( localDevicesIt->product_string ) );
                if ( deviceSignal_ && localDevicesIt->usage == HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER && localDevicesIt->usage_page == HID_USAGE_PAGE_GENERIC )
                    deviceSignal_( fmt::format( "HID API device {:04x}:{:04x} found: {}:{}", vendorId, localDevicesIt->product_id,
                        wideToUtf8( localDevicesIt->manufacturer_string ), wideToUtf8( localDevicesIt->product_string ) ) );
            }
            for ( ProductId deviceId : supportedDevicesId )
            {
                if ( !device_ && deviceId == localDevicesIt->product_id && localDevicesIt->usage == HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER && localDevicesIt->usage_page == HID_USAGE_PAGE_GENERIC )
                {
                    device_ = hid_open_path( localDevicesIt->path );
                    if ( device_ )
                    {
                        numMsg_ = 0;
                        spdlog::info( "SpaceMouse connected: {:04x}:{:04x}, path={}", vendorId, deviceId, localDevicesIt->path );
                        if ( deviceSignal_ )
                            deviceSignal_( fmt::format( "HID API device {:04x}:{:04x} opened", vendorId, localDevicesIt->product_id ) );
                        if ( !smDevice_ )
                            smDevice_ = std::make_unique<SpaceMouseDevice>();
                        // setup buttons logger
                        smDevice_->resetDevice(); // as far as we now only support single device with HID API we reset old one
                        smDevice_->updateDevice( vendorId, deviceId );
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
                if ( smDevice_ )
                    smDevice_->resetDevice();
                smDevice_.reset();
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

void SpaceMouseHandlerHidapi::updateActionWithInput_( const DataPacketRaw& packet, int packet_length, SpaceMouseAction& action )
{
    if ( !smDevice_ )
        return;
    smDevice_->parseRawEvent( packet, packet_length, action );
}

void SpaceMouseHandlerHidapi::processAction_( const SpaceMouseAction& action )
{
    ++numMsg_;
    if ( deviceSignal_ )
    {
        if ( numMsg_ == 1 )
            deviceSignal_( "HID API first action processing" );
        if ( std::popcount( numMsg_ ) == 1 ) // report every power of 2
            deviceSignal_( "SpaceMouse next log messages" );
    }
    if ( smDevice_ )
        smDevice_->processAction( action );
    glfwPostEmptyEvent();
}

}
#endif
