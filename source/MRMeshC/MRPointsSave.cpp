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
    settings.saveValidOnly = settings_->saveValidOnly;
    settings.rearrangeTriangles = settings_->rearrangeTriangles;
    settings.progress = settings_->progress;
    settings.colors = auto_cast( settings_->colors );

    auto res = PointsSave::toAnySupportedFormat( pc, file );
    if ( !res && errorString != nullptr )
        *errorString = auto_cast( new_from( std::move( res.error() ) ) );
}
