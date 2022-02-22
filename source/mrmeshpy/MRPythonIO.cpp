#include "MRMesh/MRPython.h"
#include <pybind11/iostream.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRVoxelsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRLinesLoad.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRLog.h"

using namespace MR;

bool pythonSaveMeshToAnyFormat( const Mesh& mesh, const std::string& path )
{
    auto res = MR::MeshSave::toAnySupportedFormat( mesh, path );
    return res.has_value();
}

Mesh pythonLoadMeshFromAnyFormat( const std::string& path )
{
    auto res = MR::MeshLoad::fromAnySupportedFormat( path );
    if ( res.has_value() )
        return std::move( *res );
    return {};
}

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

}

Mesh pythonLoadMeshFromAnyFormat( pybind11::object fileHandle, const std::string& extension )
{
    if ( !( pybind11::hasattr( fileHandle, "read" ) && pybind11::hasattr( fileHandle, "seek" ) && pybind11::hasattr( fileHandle, "tell" ) ) )
    {
        spdlog::error( "Argument is not file handle" );
        return {};
    }
    PythonIstreamBuf streambuf( fileHandle );
    std::istream ifs( &streambuf );
    auto res = MR::MeshLoad::fromAnySupportedFormat( ifs, extension );
    if ( res.has_value() )
        return std::move( *res );
    std::cout << res.error() << '\n';
    return {};
}

bool pythonSaveLinesToAnyFormat( const MR::Polyline& lines, const std::string& path )
{
    auto res = MR::LinesSave::toAnySupportedFormat( lines, path );
    return res.has_value();
}

MR::Polyline pythonLoadLinesFromAnyFormat( const std::string& path )
{
    auto res = MR::LinesLoad::fromAnySupportedFormat( path );
    if ( res.has_value() )
        return std::move( *res );
    return {};
}

bool pythonSavePointCloudToAnyFormat( const PointCloud& points, const std::string& path )
{
    auto res = MR::PointsSave::toAnySupportedFormat( points, path );
    return res.has_value();
}

PointCloud pythonLoadPointCloudFromAnyFormat( const std::string& path )
{
    auto res = MR::PointsLoad::fromAnySupportedFormat( path );
    if ( res.has_value() )
        return std::move( *res );
    return {};
}

MR_ADD_PYTHON_FUNCTION( mrmeshpy, save_mesh, pythonSaveMeshToAnyFormat, "saves mesh in file of known format/extension" )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadMesh, [] ( pybind11::module_& m )
{
    m.def( "load_mesh", ( MR::Mesh( * )( const std::string& ) )& pythonLoadMeshFromAnyFormat, "load mesh of known format" );
    m.def( "load_mesh", ( MR::Mesh( * )( pybind11::object, const std::string& ) )& pythonLoadMeshFromAnyFormat, "load mesh from python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, save_lines, pythonSaveLinesToAnyFormat, "saves lines in file of known format/extension" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, load_lines, pythonLoadLinesFromAnyFormat, "load lines of known format" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, save_points, pythonSavePointCloudToAnyFormat, "saves point cloud in file of known format/extension" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, load_points, pythonLoadPointCloudFromAnyFormat, "load point cloud of known format" )
