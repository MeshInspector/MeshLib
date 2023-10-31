#include "MRMesh/MRPython.h"
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRVoxelsSave.h"
#include "MRMesh/MRVoxelsLoad.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRLinesLoad.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRExpected.h"

using namespace MR;

namespace MR
{

class PythonIstreamBuf : public std::streambuf
{
public:
    PythonIstreamBuf( pybind11::object inFileHandle ) :
        pyseek_( inFileHandle.attr( "seek" ) ),
        pytell_( inFileHandle.attr( "tell" ) ),
        pyread_( inFileHandle.attr( "read" ) )
    {
        size_ = pyseek_( 0, 2 ).cast<std::streamsize>();
        pyseek_( 0 );
    };

    virtual std::streamsize showmanyc() override
    {
        std::streamsize currentPos = pytell_().cast<std::streamsize>();
        return size_ - currentPos;
    }

    virtual std::streamsize xsgetn( char* elem, std::streamsize count ) override
    {
        count = std::min( showmanyc(), count );
        if ( count == 0 )
            return 0;
        std::string readBytes = pyread_( count ).cast<std::string>();
        std::copy( readBytes.c_str(), readBytes.c_str() + count, elem );
        return count;
    }

    virtual int_type underflow() override
    {
        std::streamsize currentPos = pytell_().cast<std::streamsize>();
        auto res = uflow();
        if ( res == std::streambuf::traits_type::eof() )
            return std::streambuf::traits_type::eof();
        pyseek_( currentPos );
        return res;
    }

    virtual int_type uflow() override
    {
        char c;
        auto numRead = xsgetn( &c, 1 );
        if ( numRead == 0 )
            return std::streambuf::traits_type::eof();
        return std::streambuf::traits_type::to_int_type( c );
    }

    virtual pos_type seekoff( off_type off, std::ios_base::seekdir way, std::ios_base::openmode ) override
    {
        std::streamsize currentPos = pytell_().cast<std::streamsize>();
        std::streamsize reqPos = currentPos + off;
        if ( way == std::ios_base::beg )
            reqPos = off;
        else if ( way == std::ios_base::end )
            reqPos = size_ + off;
        pyseek_( reqPos );
        return reqPos;
    }

    virtual pos_type seekpos( pos_type pos, std::ios_base::ios_base::openmode ) override
    {
        pyseek_( std::streamsize( pos ) );
        return pytell_().cast<std::streamsize>();
    }

    virtual int_type pbackfail( int_type c ) override
    {
        std::streamsize currentPos = pytell_().cast<std::streamsize>();
        if ( currentPos == 0 )
            return std::streambuf::traits_type::eof();
        pyseek_( currentPos - 1 );
        return c;
    }

private:
    pybind11::object pyseek_;
    pybind11::object pytell_;
    pybind11::object pyread_;
    std::streamsize size_;
};

// Buffer that writes in Python instead of C++
class PythonOstreamBuf : public std::stringbuf
{
public:
    PythonOstreamBuf( pybind11::object outFileHandle ) :
        pywrite_( outFileHandle.attr( "write" ) ),
        pyflush_( outFileHandle.attr( "flush" ) )
    {
    }
    ~PythonOstreamBuf()
    {
        sync_();
    }
    int sync() override
    {
        sync_();
        return 0;
    }

private:
    pybind11::object pywrite_;
    pybind11::object pyflush_;
    void sync_()
    {
        pybind11::bytes bytes = pybind11::bytes( this->str() );
        pywrite_( bytes );
        pyflush_();
    }
};

}

Expected<MR::Mesh, std::string> pythonLoadMeshFromAnyFormat( pybind11::object fileHandle, const std::string& extension )
{
    if ( !( pybind11::hasattr( fileHandle, "read" ) && pybind11::hasattr( fileHandle, "seek" ) && pybind11::hasattr( fileHandle, "tell" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonIstreamBuf streambuf( fileHandle );
    std::istream ifs( &streambuf );
    return MR::MeshLoad::fromAnySupportedFormat( ifs, extension );
}

VoidOrErrStr pythonSaveMeshToAnyFormat( const Mesh& mesh, const std::string& extension, pybind11::object fileHandle )
{
    if ( !( pybind11::hasattr( fileHandle, "write" ) && pybind11::hasattr( fileHandle, "flush" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonOstreamBuf pybuf( fileHandle );
    std::ostream outfs( &pybuf );
    return MR::MeshSave::toAnySupportedFormat( mesh, outfs, extension );
}

VoidOrErrStr pythonSaveLinesToAnyFormat( const MR::Polyline3& lines, const std::string& extension, pybind11::object fileHandle )
{
    if ( !( pybind11::hasattr( fileHandle, "write" ) && pybind11::hasattr( fileHandle, "flush" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonOstreamBuf pybuf( fileHandle );
    std::ostream outfs( &pybuf );
    return MR::LinesSave::toAnySupportedFormat( lines, outfs, extension );
}

Expected<Polyline3, std::string> pythonLoadLinesFromAnyFormat( pybind11::object fileHandle, const std::string& extension )
{
    if ( !( pybind11::hasattr( fileHandle, "read" ) && pybind11::hasattr( fileHandle, "seek" ) && pybind11::hasattr( fileHandle, "tell" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonIstreamBuf streambuf( fileHandle );
    std::istream ifs( &streambuf );
    return MR::LinesLoad::fromAnySupportedFormat( ifs, extension );
}

VoidOrErrStr pythonSavePointCloudToAnyFormat( const PointCloud& points, const std::string& extension, pybind11::object fileHandle )
{
    if ( !( pybind11::hasattr( fileHandle, "write" ) && pybind11::hasattr( fileHandle, "flush" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonOstreamBuf pybuf( fileHandle );
    std::ostream outfs( &pybuf );
    return MR::PointsSave::toAnySupportedFormat( points, outfs, extension );
}

Expected<PointCloud, std::string> pythonLoadPointCloudFromAnyFormat( pybind11::object fileHandle, const std::string& extension )
{
    if ( !( pybind11::hasattr( fileHandle, "read" ) && pybind11::hasattr( fileHandle, "seek" ) && pybind11::hasattr( fileHandle, "tell" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonIstreamBuf streambuf( fileHandle );
    std::istream ifs( &streambuf );
    return MR::PointsLoad::fromAnySupportedFormat( ifs, extension );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SaveMesh, [] ( pybind11::module_& m )
{
    m.def( "saveMesh",
        MR::decorateExpected( ( VoidOrErrStr( * )( const MR::Mesh&, const std::filesystem::path&, const VertColors*, ProgressCallback ) )& MR::MeshSave::toAnySupportedFormat ),
        pybind11::arg( "mesh" ), pybind11::arg( "path" ), pybind11::arg( "colors" ) = nullptr, pybind11::arg( "callback" ) = ProgressCallback{}, 
        "detects the format from file extension and save mesh to it" );
    m.def( "saveMesh",
        MR::decorateExpected( ( VoidOrErrStr( * )( const MR::Mesh&, const std::string&, pybind11::object ) )& pythonSaveMeshToAnyFormat ),
        pybind11::arg( "mesh" ), pybind11::arg( "extension" ), pybind11::arg( "fileHandle" ), "saves mesh in python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadMesh, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshLoadSettings>( m, "MeshLoadSettings", "mesh load settings" ).
        def( pybind11::init<>() ).
        def_readwrite( "colors", &MR::MeshLoadSettings::colors ).
        def_readwrite( "skippedFaceCount", &MR::MeshLoadSettings::skippedFaceCount ).
        def_readwrite( "duplicatedVertexCount", &MR::MeshLoadSettings::duplicatedVertexCount ).
        def_readwrite( "xf", &MR::MeshLoadSettings::xf ).
        def_readwrite( "callback", &MR::MeshLoadSettings::callback );

    m.def( "loadMesh",
        MR::decorateExpected( ( Expected<MR::Mesh, std::string>( * )( const std::filesystem::path&, const MeshLoadSettings& ) )& MR::MeshLoad::fromAnySupportedFormat),
        pybind11::arg( "path" ), pybind11::arg( "settings" ) = MeshLoadSettings(),
        "detects the format from file extension and loads mesh from it" );
    m.def( "loadMesh",
        MR::decorateExpected( ( Expected<MR::Mesh, std::string>( * )( pybind11::object, const std::string& ) )& pythonLoadMeshFromAnyFormat ),
        pybind11::arg( "fileHandle" ), pybind11::arg( "extension" ), "load mesh from python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SaveLines, [] ( pybind11::module_& m )
{
    m.def( "saveLines",
        MR::decorateExpected( ( VoidOrErrStr( * )( const MR::Polyline3&, const std::filesystem::path&, ProgressCallback ) )& MR::LinesSave::toAnySupportedFormat ),
        pybind11::arg( "polyline" ), pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{}, 
        "detects the format from file extension and saves polyline in it" );
    m.def( "saveLines",
        MR::decorateExpected( ( VoidOrErrStr( * )( const MR::Polyline3&, const std::string&, pybind11::object ) )& pythonSaveLinesToAnyFormat ),
        pybind11::arg( "polyline" ), pybind11::arg( "extension" ), pybind11::arg( "fileHandle" ), "saves lines in python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadLines, [] ( pybind11::module_& m )
{
    m.def( "loadLines",
        MR::decorateExpected( ( Expected<Polyline3, std::string>( * )( const std::filesystem::path&, ProgressCallback ) )& MR::LinesLoad::fromAnySupportedFormat ),
        pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{}, 
        "detects the format from file extension and loads polyline from it" );
    m.def( "loadLines",
        MR::decorateExpected( ( Expected<Polyline3, std::string>( * )( pybind11::object, const std::string& ) )& pythonLoadLinesFromAnyFormat) ,
        pybind11::arg( "fileHandle" ), pybind11::arg( "extension" ), "load lines from python file handler, second arg: extension (`*.ext` format)" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SavePoints, [] ( pybind11::module_& m )
{
    m.def( "savePoints",
        MR::decorateExpected( []( const PointCloud& cloud, const std::filesystem::path& file, const VertColors* colors, ProgressCallback callback )
            { return MR::PointsSave::toAnySupportedFormat( cloud, file, { .colors = colors, .callback = callback } ); } ),
        pybind11::arg( "pointCloud" ), pybind11::arg( "path" ), pybind11::arg( "colors" ) = nullptr, pybind11::arg( "callback" ) = ProgressCallback{},
        "detects the format from file extension and save points to it" );
    m.def( "savePoints",
        MR::decorateExpected( ( VoidOrErrStr( * )( const MR::PointCloud&, const std::string&, pybind11::object ) )& pythonSavePointCloudToAnyFormat ),
        pybind11::arg( "pointCloud" ), pybind11::arg( "extension" ), pybind11::arg( "fileHandle" ), "saves point cloud in python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadPoints, [] ( pybind11::module_& m )
{
    m.def( "loadPoints",
        MR::decorateExpected( ( Expected<PointCloud, std::string>( * )( const std::filesystem::path&, VertColors*, AffineXf3f*, ProgressCallback ) )& MR::PointsLoad::fromAnySupportedFormat ),
        pybind11::arg( "path" ), pybind11::arg( "colors" ) = nullptr, pybind11::arg( "outXf" ) = nullptr, pybind11::arg( "callback" ) = ProgressCallback{},
        "detects the format from file extension and loads points from it" );
    m.def( "loadPoints",
        MR::decorateExpected( ( Expected<PointCloud, std::string>( * )( pybind11::object, const std::string& ) )& pythonLoadPointCloudFromAnyFormat ),
        pybind11::arg( "fileHandle" ), pybind11::arg( "extension" ), "load point cloud from python file handler, second arg: extension (`*.ext` format)" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SaveVoxels, [] ( pybind11::module_& m )
{
    m.def( "saveVoxels",
        MR::decorateExpected( &MR::VoxelsSave::toRawAutoname ),
        pybind11::arg( "path" ), pybind11::arg( "VdbVoxels" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Save raw voxels file, writing parameters in name." );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadDCMResult, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::VoxelsLoad::LoadDCMResult>( m, "LoadDCMResult" ).
        def_readwrite( "vdbVolume", &MR::VoxelsLoad::LoadDCMResult::vdbVolume ).
        def_readwrite( "name", &MR::VoxelsLoad::LoadDCMResult::name ).
        def_readwrite( "xf", &MR::VoxelsLoad::LoadDCMResult::xf );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, LoadDCMResults, MR::VoxelsLoad::LoadDCMResult )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadVoxels, [] ( pybind11::module_& m )
{
    m.def( "loadVoxels",
        MR::decorateExpected( ( Expected<VdbVolume, std::string>( * )( const std::filesystem::path&, const ProgressCallback& ) )& MR::VoxelsLoad::fromRaw ),
        pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Load raw voxels file, parsing parameters from name." );

    m.def( "loadDCMFolder", MR::decorateExpected( &MR::VoxelsLoad::loadDCMFolder ),
        pybind11::arg( "path" ), pybind11::arg( "maxNumThreads" ) = 4, pybind11::arg( "callback" ) = ProgressCallback{},
        "Loads first volumetric data from DICOM file(s)" );

    m.def( "loadDCMsFolder", 
        [] ( const std::filesystem::path& p, unsigned maxNumThreads, const ProgressCallback& cb)
    {
        auto res = MR::VoxelsLoad::loadDCMsFolder( p, maxNumThreads, cb );
        std::vector<MR::VoxelsLoad::LoadDCMResult> resVec;
        std::string accumError;
        for ( auto& r : res )
        {
            if ( r.has_value() )
                resVec.push_back( std::move( *r ) );
            else
                accumError += ( r.error() + "\n" );
        }
        if ( resVec.empty() )
            throwExceptionFromExpected( accumError );
        return resVec;
    },
        pybind11::arg( "path" ), pybind11::arg( "maxNumThreads" ) = 4, pybind11::arg( "callback" ) = ProgressCallback{},
        "Loads all volumetric data from DICOM file(s)" );
} )
