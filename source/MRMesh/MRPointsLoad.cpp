#include "MRPointsLoad.h"
#include "MRTimer.h"
#include "miniply.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "OpenCTM/openctm.h"
#include "MRStreamOperators.h"
#include <fstream>

namespace MR
{

namespace PointsLoad
{

const IOFilters Filters =
{
    {"All (*.*)",         "*.*"},
    {"CTM (.ctm)",        "*.ctm"},
    {"OBJ (.obj)",        "*.obj"},
    {"PLY (.ply)",        "*.ply"},
    {"PTS (.pts)",        "*.pts"}
};

tl::expected<MR::PointCloud, std::string> fromCtm( const std::filesystem::path& file, std::vector<Color>* colors /*= nullptr */ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromCtm( in, colors );
}

tl::expected<MR::PointCloud, std::string> fromCtm( std::istream& in, std::vector<Color>* colors /*= nullptr */ )
{
    MR_TIMER;

    class ScopedCtmConext
    {
        CTMcontext context_ = ctmNewContext( CTM_IMPORT );
    public:
        ~ScopedCtmConext()
        {
            ctmFreeContext( context_ );
        }
        operator CTMcontext()
        {
            return context_;
        }
    } context;

    ctmLoadCustom( context, []( void* buf, CTMuint size, void* data )
    {
        std::istream& s = *reinterpret_cast<std::istream*>( data );
        auto pos = s.tellg();
        s.read( (char*) buf, size );
        return (CTMuint) ( s.tellg() - pos );
    }, &in );

    auto vertCount = ctmGetInteger( context, CTM_VERTEX_COUNT );
    auto vertices = ctmGetFloatArray( context, CTM_VERTICES );
    if ( ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( "Error reading CTM format" );

    if ( colors )
    {
        auto colorAttrib = ctmGetNamedAttribMap( context, "Color" );
        if ( colorAttrib != CTM_NONE )
        {
            auto colorArray = ctmGetFloatArray( context, colorAttrib );
            colors->resize( vertCount );
            for ( CTMuint i = 0; i < vertCount; ++i )
            {
                auto j = 4 * i;
                ( *colors )[i] = Color( colorArray[j], colorArray[j + 1], colorArray[j + 2], colorArray[j + 3] );
            }
        }
    }

    PointCloud points;
    points.points.resize( vertCount );
    points.validPoints.resize( vertCount, true );
    for ( VertId i{0}; i < (int) vertCount; ++i )
        points.points[i] = Vector3f( vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2] );

    if ( ctmGetInteger( context, CTM_HAS_NORMALS ) == CTM_TRUE )
    {
        auto normals = ctmGetFloatArray( context, CTM_NORMALS );
        points.normals.resize( vertCount );
        for ( VertId i{0}; i < (int) vertCount; ++i )
            points.normals[i] = Vector3f( normals[3 * i], normals[3 * i + 1], normals[3 * i + 2] );
    }

    return points;
}

tl::expected<MR::PointCloud, std::string> fromPly( const std::filesystem::path& file, std::vector<Color>* colors /*= nullptr */ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromPly( in, colors );
}

tl::expected<MR::PointCloud, std::string> fromPly( std::istream& in, std::vector<Color>* colors /*= nullptr */ )
{
    MR_TIMER;

    miniply::PLYReader reader( in );
    if ( !reader.valid() )
        return tl::make_unexpected( std::string( "PLY file open error" ) );

    uint32_t indecies[3];
    bool gotVerts = false;

    std::vector<unsigned char> colorsBuffer;
    PointCloud res;
    for ( ; reader.has_element() && !gotVerts; reader.next_element() )
    {
        if ( reader.element_is( miniply::kPLYVertexElement ) && reader.load_element() )
        {
            auto numVerts = reader.num_rows();
            if ( reader.find_pos( indecies ) )
            {
                res.points.resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, res.points.data() );
                gotVerts = true;
            }
            if ( colors && reader.find_color( indecies ) )
            {
                colorsBuffer.resize( 3 * numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
            }
            continue;
        }
    }

    if ( !reader.valid() )
        return tl::make_unexpected( std::string( "PLY file read or parse error" ) );

    if ( !gotVerts )
        return tl::make_unexpected( std::string( "PLY file does not contain vertices" ) );

    res.validPoints.resize( res.points.size(), true );
    if ( colors && !colorsBuffer.empty() )
    {
        colors->resize( res.points.size() );
        for ( int i = 0; i < res.points.size(); ++i )
        {
            int ind = 3 * i;
            ( *colors )[i] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
        }
    }

    return std::move( res );
}

tl::expected<MR::PointCloud, std::string> fromPts( const std::filesystem::path& file )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromPts( in );
}

tl::expected<MR::PointCloud, std::string> fromPts( std::istream& in )
{
    std::string line;
    int i = 0;
    PointCloud cloud;
    while ( std::getline( in, line ) )
    {
        line.erase( std::find_if( line.rbegin(), line.rend(), [] ( unsigned char ch ) { return !std::isspace( ch ); } ).base(), line.end() ); 
        if ( i == 0 )
        {
            if ( line != "BEGIN_Polyline" )
                return tl::make_unexpected( "Not valid .pts format" );
            else
            {
                ++i;
                continue;
            }
        }
        if ( line == "END_Polyline" )
            break;
        std::istringstream iss( line );
        Vector3f point;
        if ( !( iss >> point ) )
            return tl::make_unexpected( "Not valid .pts format" );
        cloud.points.push_back( point );
        ++i;
    }
    cloud.validPoints.resize( i - 1, true );
    return std::move( cloud );
}

tl::expected<MR::PointCloud, std::string> fromObj( const std::filesystem::path& file )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromObj( in );
}

tl::expected<MR::PointCloud, std::string> fromObj( std::istream& in )
{
    PointCloud cloud;

    for ( ;; )
    {
        if ( !in )
            return tl::make_unexpected( std::string( "OBJ-format read error" ) );
        char ch = 0;
        in >> ch;
        if ( in.eof() )
            break;
        if ( ch == 'v' )
        {
            float x, y, z;
            in >> x >> y >> z;
            cloud.points.emplace_back( x, y, z );
        }
        else
        {
            // skip unknown line
            std::string str;
            std::getline( in, str );
        }
    }

    cloud.validPoints.resize( cloud.points.size(), true );
    return std::move( cloud );
}

tl::expected<MR::PointCloud, std::string> fromAnySupportedFormat( const std::filesystem::path& file, std::vector<Color>* colors /*= nullptr */ )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<MR::PointCloud, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".ply" )
        res = MR::PointsLoad::fromPly( file, colors );
    else if ( ext == u8".ctm" )
        res = MR::PointsLoad::fromCtm( file, colors );
    else if ( ext == u8".pts" )
        res = MR::PointsLoad::fromPts( file );
    else if ( ext == u8".obj" )
        res = MR::PointsLoad::fromObj( file );
    return res;
}

tl::expected<MR::PointCloud, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, std::vector<Color>* colors /*= nullptr */ )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<MR::PointCloud, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".ply" )
        res = MR::PointsLoad::fromPly( in, colors );
    else if ( ext == ".ctm" )
        res = MR::PointsLoad::fromCtm( in, colors );
    else if ( ext == ".pts" )
        res = MR::PointsLoad::fromPts( in );
    else if ( ext == ".obj" )
        res = MR::PointsLoad::fromObj( in );
    return res;
}

}
}
