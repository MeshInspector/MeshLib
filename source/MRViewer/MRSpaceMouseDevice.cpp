#include "MRSpaceMouseDevice.h"
#include "MRViewer.h"

namespace MR::SpaceMouse
{

void SpaceMouseDevice::updateDevice( VendorId vendorId, ProductId productId )
{
    if ( vId_ == vendorId && pId_ == productId )
        return;
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
    vId_ = vendorId;
    pId_ = productId;
}

bool SpaceMouseDevice::valid() const
{
    return bool( buttonsMapPtr_ );
}

void SpaceMouseDevice::resetDevice()
{
    buttonsMapPtr_ = nullptr;
    buttonsState_ = 0;
}

void SpaceMouseDevice::processAction( const SpaceMouseAction& action )
{
    if ( !valid() )
        return;
    auto& viewer = getViewerInstance();
    viewer.spaceMouseMove( action.translate, action.rotate );
    if ( action.btnStateChanged )
    {
        std::bitset<SMB_BUTTON_COUNT> new_pressed = action.buttons & ~buttonsState_;
        std::bitset<SMB_BUTTON_COUNT> new_unpressed = buttonsState_ & ~action.buttons;
        for ( int btn = 0; btn < SMB_BUTTON_COUNT; ++btn )
        {
            if ( new_unpressed.test( btn ) )
                viewer.spaceMouseUp( btn );
            if ( new_pressed.test( btn ) )
                viewer.spaceMouseDown( btn );
        }
        buttonsState_ = action.buttons;
    }
}

void SpaceMouseDevice::parseRawEvent( const DataPacketRaw& raw, int numBytes, SpaceMouseAction& action ) const
{
    if ( !valid() )
        return;
    // button update package
    if ( raw[0] == 3 && buttonsMapPtr_ != nullptr )
    {
        action.btnStateChanged = true;
        // for all bytes in packet
        for ( int column = 1; column < buttonsMapPtr_->size(); ++column )
        {
            for ( int i = 0; i < ( *buttonsMapPtr_ )[column].size(); ++i )
            {
                if ( raw[column] & ( 1 << i ) )
                    action.buttons.set( ( *buttonsMapPtr_ )[column][i] );
            }
        }
        return;
    }

    auto convertCoord = [] ( int coord_byte_low, int coord_byte_high )
    {
        int value = coord_byte_low | ( coord_byte_high << 8 );
        if ( value > SHRT_MAX )
        {
            value = value - 65536;
        }
        float ret = ( float )value / 350.0f;
        return ( std::abs( ret ) > 0.01f ) ? ret : 0.0f;
    };

    Vector3f matrix = { 0.0f, 0.0f, 0.0f };
    if ( numBytes >= 7 )
    {
        matrix = { convertCoord( raw[1], raw[2] ),
                  convertCoord( raw[3], raw[4] ),
                  convertCoord( raw[5], raw[6] ) };

        if ( raw[0] == 1 )
            action.translate = matrix;
        else if ( raw[0] == 2 )
            action.rotate = matrix;
    }
    if ( numBytes == 13 )
    {
        action.translate = matrix;
        action.rotate = { convertCoord( raw[7], raw[8] ),
                         convertCoord( raw[9], raw[10] ),
                         convertCoord( raw[11], raw[12] ) };
    }
}

}
