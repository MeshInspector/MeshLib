#include "MRMeshSave.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"

using namespace MR;

void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh, const char* file, MRString** errorStr )
{
    auto res = MeshSave::toAnySupportedFormat( *reinterpret_cast<const Mesh*>( mesh ), file );
    if ( !res && errorStr )
    {
        auto* str = new std::string( res.error() );
        *errorStr = reinterpret_cast<MRString*>( str );
    }
}
