#include "MRPointsLoad.h"
#include "MRPointsLoadSettings.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRColor.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointsLoad.h"

using namespace MR;

REGISTER_AUTO_CAST( PointCloud )
REGISTER_AUTO_CAST( PointsLoadSettings )
REGISTER_AUTO_CAST2( std::string, MRString )

MRPointCloud* mrPointsLoadFromAnySupportedFormat( const char* filename, const MRPointsLoadSettings* settings_, MRString** errorString )
{
    PointsLoadSettings settings;
    if ( settings_ )
    {
        if ( settings_->colors )
        {
            vector_wrapper<Color>* wrapper = reinterpret_cast<vector_wrapper<Color>* >( settings_->colors );
            if ( wrapper )
                settings.colors = reinterpret_cast<VertColors* >( &reinterpret_cast<std::vector<Color>& >( *wrapper ) );
        }
        settings.outXf = ( AffineXf3f* )settings_->outXf;
        settings.callback = settings_->callback;
    }

    auto res = PointsLoad::fromAnySupportedFormat( filename, settings );

    if ( res )
    {
        if ( settings.colors )
            mrVertColorsInvalidate( settings_->colors );

        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return NULL;
    }
}
