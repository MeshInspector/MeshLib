#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>
#include <MRSymbolMesh/MRSymbolMesh.h>

#include <iostream>

int main( int argc, char* argv[] )
{
    if ( argc < 3 )
    {
        std::cerr << "Usage: MeshFromText fontpath text" << std::endl;
        return EXIT_FAILURE;
    }
    const auto* fontPath = argv[1];
    const auto* text = argv[2];

    // set text-to-mesh parameters
    MR::SymbolMeshParams params {
        // input text
        .text = text,
        // font file name
        .pathToFontFile = fontPath,
    };
    auto convRes = MR::createSymbolsMesh( params );
    if ( !convRes )
    {
        std::cerr << "Failed to convert text to mesh: " << convRes.error() << std::endl;
        return EXIT_FAILURE;
    }

    auto saveRes = MR::MeshSave::toAnySupportedFormat( *convRes, "mesh.ply" );
    if ( !saveRes )
    {
        std::cerr << "Failed to save result: " << saveRes.error() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}