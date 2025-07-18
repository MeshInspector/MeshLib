#include "MRPointsSave.h"
#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRSaveSettings.h"

using namespace MR;

REGISTER_AUTO_CAST( PointCloud )
REGISTER_AUTO_CAST( VertColors )
REGISTER_AUTO_CAST2( std::string, MRString )

void mrPointsSaveToAnySupportedFormat( const MRPointCloud* pc_, const char* file, const MRSaveSettings* settings_, MRString** errorString )
{
    ARG( pc );
    SaveSettings settings;
    if ( settings_ )
    {
        settings.onlyValidPoints = settings_->onlyValidPoints;
        settings.packPrimitives = settings_->packPrimitives;
        settings.progress = settings_->progress;
        vector_wrapper<Color>* wrapper = (vector_wrapper<Color>*)( settings_->colors );
        if ( wrapper )
            settings.colors = reinterpret_cast<const VertColors*>( &(std::vector<Color>&)( *wrapper ) );
    }

    auto res = PointsSave::toAnySupportedFormat( pc, file, settings );
    if ( !res && errorString != nullptr )
        *errorString = auto_cast( new_from( std::move( res.error() ) ) );
}
