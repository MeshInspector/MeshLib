#ifndef _WIN32
#ifndef __EMSCRIPTEN__
#include "MRSpaceMouseHandlerHidapi.h"

#include "MRViewer/MRViewerFwd.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"

namespace MR
{
SpaceMouseHandlerHidapi::SpaceMouseHandlerHidapi()
        : device_(nullptr)
        , updateThreadActive_(false)
        , hasMousePackets_(false)
        , translate_(0.0f, 0.0f, 0.0f)
        , rotate_(0.0f, 0.0f, 0.0f)
{}

SpaceMouseHandlerHidapi::~SpaceMouseHandlerHidapi()
{
    updateThreadActive_ = false;

    std::unique_lock<std::mutex> dataLock( dataMutex_ );
    hasMousePackets_ = false;
    dataLock.unlock();
    cv_.notify_one();

    if ( updateThread_.joinable() )
        updateThread_.join();

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
    updateThreadActive_ = true;
    startUpdateThread_();
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
                if (  deviceId == localDevicesIt->product_id )
                {
                    device_ = hid_open( vendorId, deviceId, NULL );
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

    std::unique_lock<std::mutex> dataLock( dataMutex_, std::defer_lock );
    if ( !dataLock.try_lock() )
        return;

    if ( !device_ )
        return;

    if ( !hasMousePackets_ )
        return;

    auto &viewer = getViewerInstance();
    // process saved data
    viewer.spaceMouseMove( translate_, rotate_ );
    int packet_length = 0;
    // set the device handle to be non-blocking
    hid_set_nonblocking( device_, 1 );
    do {
        DataPacketRaw packet = {0};
        packet_length = hid_read( device_, packet.data(), packet.size() );

        if ( packet_length == 13 || packet_length == 7 ) {
            Vector3f translate, rotate;
            convertInput_( packet, packet_length, translate, rotate );
            viewer.spaceMouseMove( translate, rotate );
        }
    } while ( packet_length > 0 );

    hasMousePackets_ = false;
    dataLock.unlock();
    cv_.notify_one();
}

void SpaceMouseHandlerHidapi::startUpdateThread_()
{
    updateThread_ = std::thread( [&]() {
        do {
            std::unique_lock<std::mutex> dataLock( dataMutex_ );
            while ( !device_ )
            {
                if ( !updateThreadActive_ )
                    return;
                if ( findAndAttachDevice_() )
                    break;
                dataLock.unlock();
                std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
                dataLock.lock();
            }

            int packet_length = 0;
            DataPacketRaw packet = {0};

            // set the device handle to be blocking
            hid_set_nonblocking( device_, 0 );
            // hid_read_timeout() waits until there is data to read before returning or 1000ms passed (to help with thread shutdown)
            packet_length = hid_read_timeout( device_, packet.data(), packet.size(), 1000 );
            // device connection lost
            if ( packet_length < 0)
            {
                hasMousePackets_ = false;
                hid_close( device_ );
                device_ = nullptr;
                spdlog::error( "HID API: device lost" );
            }
            else if ( packet_length > 0)
            {
                hasMousePackets_ = true;
                // save data and trigger main rendering loop
                convertInput_( packet, packet_length, translate_, rotate_ );
                glfwPostEmptyEvent();
                // wait for main thread to read and process all SpaceMouse packets
                cv_.wait( dataLock, [&]{return !hasMousePackets_;} );
            }
            // nothing to do with packet_length == 0
        } while ( updateThreadActive_ );
    });
}

float SpaceMouseHandlerHidapi::convertCoord_( int coord_byte_low, int coord_byte_high )
{
    int value = coord_byte_low | (coord_byte_high << 8);
    if ( value > SHRT_MAX ) {
        value = value - 65536;
    }
    float ret = (float)value / 350.0;
    return (std::abs(ret) > 0.01) ? ret : 0.0;
}

void SpaceMouseHandlerHidapi::convertInput_( const DataPacketRaw& packet, int packet_length, Vector3f& translate, Vector3f& rotate )
{
    translate = {0.0f, 0.0f, 0.0f};
    rotate = {0.0f, 0.0f, 0.0f};
    if ( packet_length >= 7 ) {
        translate = {convertCoord_( packet[1], packet[2] ),
                     convertCoord_( packet[3], packet[4] ),
                     convertCoord_( packet[5], packet[6] )};
    }
    if ( packet_length >= 13 ) {
        rotate = {convertCoord_( packet[7], packet[8] ),
                  convertCoord_( packet[9], packet[10] ),
                  convertCoord_( packet[11], packet[12] )};
    }
}

void SpaceMouseHandlerHidapi::printDevices_( struct hid_device_info *cur_dev ) {
    while ( cur_dev )
    {
        if ( vendor2device_.find( cur_dev->vendor_id) != vendor2device_.end() )
        {
            spdlog::info( "Device Found: type: {} {} path: {} ", cur_dev->vendor_id, cur_dev->product_id, cur_dev->path );
        }
        cur_dev = cur_dev->next;
    }
    hid_free_enumeration( cur_dev );
}

}
#endif
#endif