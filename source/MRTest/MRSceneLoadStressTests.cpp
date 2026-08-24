#include "MRMesh/MRCube.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRPch/MRSpdlog.h"
#include <gtest/gtest.h>

#include <filesystem>

namespace MR
{

// Reloads one and the same scene archive many times, hunting a rare stall seen
// twice in a wasm64 browser build: loadObjectFromFile never returned while the
// page itself stayed responsive. Three meshes, sized after the archive that
// stalled: ~3.4 MB, ~10 KB, ~250 B of .ply.
TEST( MRMesh, SceneLoadStress )
{
    auto addMesh = [] ( Object& root, std::string name, Mesh mesh )
    {
        auto obj = std::make_shared<ObjectMesh>();
        obj->setName( std::move( name ) );
        obj->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
        root.addChild( std::move( obj ) );
    };

    Object root;
    root.setName( "Root" );
    addMesh( root, "big", makeUVSphere( 1.0f, 300, 300 ) );
    addMesh( root, "small", makeUVSphere( 1.0f, 16, 16 ) );
    addMesh( root, "tiny", makeCube() );

    UniqueTemporaryFolder folder;
    ASSERT_TRUE( bool( folder ) );
    const auto mruPath = folder / "stress.mru";
    const auto saved = serializeObjectTree( root, mruPath );
    ASSERT_TRUE( saved.has_value() ) << saved.error();

    std::error_code ec;
    spdlog::info( "stress.mru: {} bytes", std::filesystem::file_size( mruPath, ec ) );

    constexpr int iterations = 200;
    for ( int i = 0; i < iterations; ++i )
    {
        Timer t( "t" );
        auto loaded = loadObjectFromFile( mruPath );
        ASSERT_TRUE( loaded.has_value() ) << "iteration " << i << ": " << loaded.error();
        ASSERT_FALSE( loaded->objs.empty() ) << "iteration " << i;
        spdlog::info( "iteration {}/{}: {} sec", i + 1, iterations, t.secondsPassed() );
    }
}

} // namespace MR
