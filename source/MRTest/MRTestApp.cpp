#include <gtest/gtest.h>
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRQuadraticForm.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRSystemPath.h"
#include "MRMesh/MRStringConvert.h"
#include "MRViewer/MRGetSystemInfoJson.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRSpdlog.h"

#include <chrono>
#include <thread>

#ifndef MESHLIB_NO_PYTHON
#include "MRPython/MRPython.h"
#include "MREmbeddedPython/MREmbeddedPython.h"
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

    MR::setupLoggerByDefault();

    // print compiler info
#ifdef __clang__
    spdlog::info( "{}", __VERSION__ );
#elif defined __GNUC__
    spdlog::info( "GCC {}", __VERSION__ );
#else
    spdlog::info( "MSVC {}", _MSC_FULL_VER );
#endif

    // print standard library info
#ifdef _MSVC_STL_UPDATE
    // https://github.com/microsoft/STL/wiki/Macro-_MSVC_STL_UPDATE
    spdlog::info( "Microsoft's STL version {}", _MSVC_STL_UPDATE );
#endif
#ifdef __GLIBCXX__
    spdlog::info( "GNU libstdc++ version {}", __GLIBCXX__ );
#endif
#ifdef _LIBCPP_VERSION
    spdlog::info( "Clang's libc++ version {}", _LIBCPP_VERSION );
#endif

    spdlog::info( "System info:\n{}", MR::GetSystemInfoJson().toStyledString() );
#ifndef MESHLIB_NO_PYTHON
    if ( consumeFlag( "--with-python-tests" ) )
    {
        //Test python mrmeshpy
        {
            // Directly from the package directory: the `__init__.py` of a wheel build would redirect the resources of this process.
            auto str = "import sys\n"
                "sys.path.insert( 0, r'" + MR::utf8string( MR::SystemPath::getPythonModulesDirectory() / "meshlib" ) + "' )\n"
                "import mrmeshpy\n"
                "print( \"List of python module functions available in mrmeshpy:\\n\" )\n"
                "funcs = dir( mrmeshpy )\n"
                "for f in funcs :\n"
                " if not f.startswith( '_' ) :\n"
                "  print( \"mrmeshpy.\" + f )\n"
                "print()"; // one empty line

            spdlog::info( "Running embedded python" );
            bool ok = MR::EmbeddedPython::runString( str );
            if ( ok )
                spdlog::info( "Embedded python run passed" );
            else
                spdlog::error( "Embedded python run failed" );

            // A thread started by one script must run while the interpreter waits for the next one, i.e. the interpreter
            // thread must not hold the GIL while idle. The thread binds `sys` and `time` itself: the globals are cleared after a script.
            if ( ok )
            {
                spdlog::info( "Checking that embedded python releases the GIL between scripts" );
                ok = MR::EmbeddedPython::runString(
                    "import sys, threading, time\n"
                    "def mark( sys=sys, time=time ):\n"
                    "    time.sleep( 0.02 )\n"
                    "    sys.mrRanWhileIdleAt = time.monotonic()\n"
                    "threading.Thread( target=mark, daemon=True ).start()\n" );
                std::this_thread::sleep_for( std::chrono::milliseconds( 500 ) );
                ok = ok && MR::EmbeddedPython::runString(
                    "import sys, time\n"
                    "startedAt = time.monotonic()\n"
                    "ranAt = getattr( sys, 'mrRanWhileIdleAt', None )\n"
                    "assert ranAt is not None and ranAt < startedAt, 'a Python thread did not run between the scripts: the GIL is held while idle'\n" );
                if ( ok )
                    spdlog::info( "Embedded python releases the GIL between scripts" );
                else
                    spdlog::error( "Embedded python holds the GIL between scripts" );
            }
            MR::EmbeddedPython::shutdown();
            spdlog::info( "Embedded python shut down" );

            if ( !ok )
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
