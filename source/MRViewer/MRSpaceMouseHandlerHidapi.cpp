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
    if ( device_ != nullptr )
        hid_set_nonblocking( device_, 1 );

    std::unique_lock<std::mutex> lock( mtx_ );
    hasMousePackets_ = false;
    lock.unlock();
    cv_.notify_one();
    if ( updateThread_.joinable() )
        updateThread_.join();

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

    hid_device_info* devs_ = hid_enumerate( 0x0, 0x0 );
#ifndef NDEBUG
    printDevices( devs_ );
#endif
    HidDevice attachDevice;
    if ( !findDevice( devs_, attachDevice ) )
        return;

    device_ = hid_open( attachDevice.vendor_id_, attachDevice.product_id_, NULL );
    if ( !device_ ) {
        spdlog::error( "HID API: device open error" );
        return;
    }
    updateThreadActive_ = true;
    startUpdateThread_();
}

void SpaceMouseHandlerHidapi::handle()
{
    if ( !device_ )
        return;

    if ( !hasMousePackets_ )
        return;

    std::unique_lock<std::mutex> lock( mtx_, std::defer_lock );
    if ( !lock.try_lock() )
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
            convertInput( packet, packet_length, translate, rotate );
            viewer.spaceMouseMove( translate, rotate );
        }
    } while ( packet_length > 0 );

    hasMousePackets_ = false;
    lock.unlock();
    cv_.notify_one();
}

void SpaceMouseHandlerHidapi::startUpdateThread_()
{
    if ( !device_ ) {
        updateThreadActive_ = false;
        return;
    }

    updateThread_ = std::thread( [&]() {
        do {
            int packet_length = 0;
            DataPacketRaw packet = {0};

            std::unique_lock<std::mutex> lock( mtx_ );

            // set the device handle to be blocking
            hid_set_nonblocking( device_, 0 );
            do {
                // hid_read_timeout() waits until there is data to read before returning or 1000ms passed (to help with thread shutdown)
                packet_length = hid_read_timeout( device_, packet.data(), packet.size(), 1000 );
            } while ( packet_length <= 0 && updateThreadActive_  ); // wait for the first valid package
            // save data and trigger main rendering loop
            convertInput( packet, packet_length, translate_, rotate_ );
            hasMousePackets_ = true;
            glfwPostEmptyEvent();

            // wait for main thread to read and process all SpaceMouse packets
            cv_.wait( lock, [&]{return !hasMousePackets_;} );
        } while ( updateThreadActive_ );
    });
}

float SpaceMouseHandlerHidapi::convertCoord( int coord_byte_low, int coord_byte_high )
{
    int value = coord_byte_low | (coord_byte_high << 8);
    if ( value > SHRT_MAX ) {
        value = value - 65536;
    }
    float ret = (float)value / 350.0;
    return (std::abs(ret) > 0.01) ? ret : 0.0;
}

void SpaceMouseHandlerHidapi::convertInput( const DataPacketRaw& packet, int packet_length, Vector3f& translate, Vector3f& rotate )
{
    translate = {0.0f, 0.0f, 0.0f};
    rotate = {0.0f, 0.0f, 0.0f};
    if ( packet_length >= 7 ) {
        translate = {convertCoord( packet[1], packet[2] ),
                     convertCoord( packet[3], packet[4] ),
                     convertCoord( packet[5], packet[6] )};
    }
    if ( packet_length >= 13 ) {
        rotate = {convertCoord( packet[7], packet[8] ),
                  convertCoord( packet[9], packet[10] ),
                  convertCoord( packet[11], packet[12] )};
    }
}

bool SpaceMouseHandlerHidapi::findDevice( struct hid_device_info *cur_dev, HidDevice& attachDevice )
{
    bool isDeviceFound = false;
    while( cur_dev )
    {
        for ( const HidDevice& device: supportedDevices_ )
        {
            if ( device.vendor_id_ == cur_dev->vendor_id && device.product_id_ == cur_dev->product_id )
            {
                attachDevice.vendor_id_ = cur_dev->vendor_id;
                attachDevice.product_id_ = cur_dev->product_id;
                isDeviceFound = true;
                break;
            }
        }
        cur_dev = cur_dev->next;
    }
    hid_free_enumeration( cur_dev );
    return isDeviceFound;
}

void SpaceMouseHandlerHidapi::printDevices( struct hid_device_info *cur_dev ) {
    while ( cur_dev )
    {
        if ( cur_dev->vendor_id == 0x256F )
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