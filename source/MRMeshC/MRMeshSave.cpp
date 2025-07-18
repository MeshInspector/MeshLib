#include "MRMeshSave.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( VertColors )
REGISTER_AUTO_CAST2( std::string, MRString )

void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh_, const char* file, const MRSaveSettings* settings_, MRString** errorStr )
{
    ARG( mesh );
    SaveSettings settings;
    if ( settings_ )
    {
        settings.onlyValidPoints = settings_->onlyValidPoints;
        settings.packPrimitives = settings_->packPrimitives;
        settings.progress = settings_->progress;
        vector_wrapper<Color>* wrapper = (vector_wrapper<Color>*)( settings_->colors );
        if ( wrapper )
            settings.colors = reinterpret_cast< const VertColors* >( &(const std::vector<Color>&) ( *wrapper ) );
    }

    auto res = MeshSave::toAnySupportedFormat( mesh, file );
    if ( !res && errorStr )
    {
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );
    }
}
