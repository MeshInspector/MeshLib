#ifndef __EMSCRIPTEN__
#include "MRSpaceMouseHandlerHidapi.h"

#include "MRViewer/MRViewerFwd.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"

namespace MR
{
SpaceMouseHandlerHidapi::SpaceMouseHandlerHidapi()
        : device_(nullptr)
        , terminateListenerThread_(false)
        , dataPacket_({0})
        , packetLength_(0)
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
    if ( hid_init() ) {
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

bool SpaceMouseHandlerHidapi::findAndAttachDevice_() {
    bool isDeviceFound = false;
    for ( const auto& [vendorId, supportedDevicesId] : vendor2device_ )
    {
        // search through supported vendors
        hid_device_info* localDevicesIt = hid_enumerate( vendorId, 0x0 );
        while( localDevicesIt && !isDeviceFound )
        {
            for ( ProductId deviceId: supportedDevicesId )
            {
                if (  deviceId == localDevicesIt->product_id && localDevicesIt->usage == 8 && localDevicesIt->usage_page == 1 )
                {
                    device_ = hid_open_path( localDevicesIt->path );
                    if ( device_ )
                    {
                        isDeviceFound = true;
                        spdlog::info( "SpaceMouse Found: type: {} {} path: {} ", vendorId, deviceId, localDevicesIt->path );
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

void SpaceMouseHandlerHidapi::handle()
{
    // works in pair with SpaceMouseHandlerHidapi::startListenerThread_()
    std::unique_lock<std::mutex> syncThreadLock( syncThreadMutex_, std::defer_lock );
    if ( !syncThreadLock.try_lock() )
        return;

    if ( packetLength_ <= 0 || !device_ )
    {
        cv_.notify_one();
        return;
    }

    // set the device handle to be non-blocking
    Vector3f translate, rotate;
    convertInput_( dataPacket_, packetLength_, translate, rotate );

    hid_set_nonblocking( device_, 1 );
    int packetLengthTmp = 0;
    do {
        DataPacketRaw dataPacketTmp;
        packetLengthTmp = hid_read( device_, dataPacketTmp.data(), dataPacketTmp.size() );
        convertInput_( dataPacketTmp, packetLengthTmp, translate, rotate );
    } while ( packetLengthTmp > 0 );

    auto& viewer = MR::getViewerInstance();
    viewer.spaceMouseMove( translate, rotate );

    syncThreadLock.unlock();
    cv_.notify_one();
}

void SpaceMouseHandlerHidapi::initListenerThread_()
{
    // works in pair with SpaceMouseHandlerHidapi::handle()
    // waits for updates on SpaceMouse and notifies main thread
    listenerThread_ = std::thread( [&]() {
        do {
            std::unique_lock<std::mutex> syncThreadLock( syncThreadMutex_ );
            // stay in loop until SpaceMouse is found
            while ( !device_ )
            {
                if ( terminateListenerThread_ )
                    return;
                if ( findAndAttachDevice_() )
                    break;
                syncThreadLock.unlock();
                std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
                syncThreadLock.lock();
            }

            // to read all data packets during inactive state
            while ( !active_ )
            {
                packetLength_ = hid_read_timeout( device_, dataPacket_.data(), dataPacket_.size(), 500 );
            }

            // set the device handle to be blocking
            hid_set_nonblocking( device_, 0 );
            // hid_read_timeout() waits until there is data to read before returning or 1000ms passed (to help with thread shutdown)
            packetLength_ = hid_read_timeout( device_, dataPacket_.data(), dataPacket_.size(), 1000 );

            // device connection lost
            if ( packetLength_ < 0)
            {
                hid_close( device_ );
                device_ = nullptr;
                spdlog::error( "HID API: device lost" );
            }
            else if ( packetLength_ > 0)
            {
                // trigger main rendering loop and wait for main thread to read and process all SpaceMouse packets
                glfwPostEmptyEvent();
                cv_.wait( syncThreadLock );
            }
            // nothing to do with packet_length == 0
        } while ( !terminateListenerThread_ );
    });
}

void SpaceMouseHandlerHidapi::postFocusSignal_( bool focused )
{
    active_ = focused;
    cv_.notify_one();
}

float SpaceMouseHandlerHidapi::convertCoord_( int coord_byte_low, int coord_byte_high )
{
    int value = coord_byte_low | (coord_byte_high << 8);
    if ( value > SHRT_MAX ) {
        value = value - 65536;
    }
    float ret = (float)value / 350.0f;
    return (std::abs(ret) > 0.01f) ? ret : 0.0f;
}

void SpaceMouseHandlerHidapi::convertInput_( const DataPacketRaw& packet, int packet_length, Vector3f& translate, Vector3f& rotate )
{
    if ( !packet_length )
        return;

    Vector3f matrix1 = {0.0f, 0.0f, 0.0f};
    Vector3f matrix2 = {0.0f, 0.0f, 0.0f};
    if ( packet_length == 7 ) {
        matrix1 = {convertCoord_( packet[1], packet[2] ),
                   convertCoord_( packet[3], packet[4] ),
                   convertCoord_( packet[5], packet[6] )};
        if ( packet[0] == 1 )
            translate = matrix1;
        else if ( packet[0] == 2 )
            rotate = matrix1;
    }
    if ( packet_length == 13 ) {
        matrix2 = {convertCoord_( packet[7], packet[8] ),
                   convertCoord_( packet[9], packet[10] ),
                   convertCoord_( packet[11], packet[12] )};

        translate = matrix1;
        rotate = matrix2;
    }
}

void SpaceMouseHandlerHidapi::printDevices_( struct hid_device_info *cur_dev ) {
    while ( cur_dev )
    {
        if ( vendor2device_.find( cur_dev->vendor_id) != vendor2device_.end() )
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
