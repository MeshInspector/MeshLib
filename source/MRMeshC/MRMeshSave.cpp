#include "MRMeshSave.h"

#include "detail/TypeCast.h"

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
    settings.saveValidOnly = settings_->saveValidOnly;
    settings.rearrangeTriangles = settings_->rearrangeTriangles;
    settings.progress = settings_->progress;
    settings.colors = auto_cast( settings_->colors );

    auto res = MeshSave::toAnySupportedFormat( mesh, file );
    if ( !res && errorStr )
    {
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );
    }
}
