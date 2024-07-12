#include "MRMeshLoad.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshLoad.h"

using namespace MR;

MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr )
{
    auto res = MeshLoad::fromAnySupportedFormat( file );
    if ( res )
    {
        auto* mesh = new Mesh( std::move( *res ) );
        return reinterpret_cast<MRMesh*>( mesh );
    }
    if ( errorStr )
    {
        auto* str = new std::string( std::move( res.error() ) );
        *errorStr = reinterpret_cast<MRString*>( str );
    }
    return nullptr;
}
