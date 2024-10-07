#include <gtest/gtest.h>
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRGTest.h"
#include "MRMesh/MRQuadraticForm.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRSystem.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRGetSystemInfoJson.h"
#include "MRViewer/MRCommandLoop.h"

#ifndef __EMSCRIPTEN__
#include "MRPython/MRPython.h"
#include "MRPython/MREmbeddedPython.h"
#include "mrmeshpy/MRLoadModule.h"
#endif

namespace MR
{

TEST(MRMesh, QuadraticForm)
{
    QuadraticForm3f q0, q1;
    q0.addDistToOrigin( 1 );
    q1.addDistToOrigin( 1 );
    auto r = sum( q0, Vector3f{0,0,0}, q1, Vector3f{2,0,0} );

    EXPECT_EQ( r.second, (Vector3f{1,0,0}) );
}

} //namespace MR

int main( int argc, char** argv )
{
    //! If `flag` exists in `argv`, returns true and removes it from there.
    [[maybe_unused]] auto consumeFlag = [&]( std::string_view flag ) -> bool
    {
        if ( argc < 1 )
            return false;
        char **end = argv + argc;
        auto it = std::find( argv + 1, end, flag );
        if ( it == end )
            return false;
        *std::rotate( it, it + 1, end ) = nullptr;
        argc--;
        return true;
    };

    MR::loadMeshDll();
    MR::loadMRViewerDll();

#ifndef __EMSCRIPTEN__
    MR::loadMRMeshPyModule();
#endif

    MR::setupLoggerByDefault();

    // print compiler info
#ifdef __clang__
    spdlog::info( "{}", __VERSION__ );
#elif defined __GNUC__
    spdlog::info( "GCC {}", __VERSION__ );
#else
    spdlog::info( "MSVC {}", _MSC_FULL_VER );
#endif
    spdlog::info( "System info:\n{}", MR::GetSystemInfoJson().toStyledString() );
#ifndef __EMSCRIPTEN__
    if ( !consumeFlag( "--no-python-tests" ) )
    {
        //Test python mrmeshpy
        {
            MR::EmbeddedPython::init( {} );
            auto str = "import mrmeshpy\n"
                "print( \"List of python module functions available in mrmeshpy:\\n\" )\n"
                "funcs = dir( mrmeshpy )\n"
                "for f in funcs :\n"
                " if not f.startswith( '_' ) :\n"
                "  print( \"mrmeshpy.\" + f )\n"
                "print( \"\\n\" )";

            if ( !MR::EmbeddedPython::runString( str ) )
                return 1;
        }

        if ( StderrPyRedirector::getNumWritten() > 0 )
        {
            spdlog::error( "Some errors reported from python" );
            return 1;
        }
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);
    MR::CommandLoop::removeCommands( false ); // that are added there by plugin constructors
    return RUN_ALL_TESTS();
}
