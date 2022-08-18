#include "MRPointsLoad.h"
#include "MRTimer.h"
#include "miniply.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "OpenCTM/openctm.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
#include "MRPointCloud.h"
#include <fstream>

namespace MR
{

namespace PointsLoad
{

const IOFilters Filters =
{
    {"All (*.*)",         "*.*"},
    {"ASC (.asc)",        "*.asc"},
    {"CTM (.ctm)",        "*.ctm"},
    {"OBJ (.obj)",        "*.obj"},
    {"PLY (.ply)",        "*.ply"},
    {"PTS (.pts)",        "*.pts"}
};

tl::expected<MR::PointCloud, std::string> fromCtm( const std::filesystem::path& file, Vector<Color, VertId>* colors /*= nullptr */, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromCtm( in, colors, callback );
}

tl::expected<MR::PointCloud, std::string> fromCtm( std::istream& in, Vector<Color, VertId>* colors /*= nullptr */, ProgressCallback callback )
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

    struct LoadData
    {
        std::function<bool( float )> callbackFn{};
        std::istream* stream;
        bool wasCanceled{ false };
    } loadData;
    loadData.stream = &in;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );

    if ( callback )
    {
        loadData.callbackFn = [callback, posStart, sizeAll = float( posEnd - posStart ), &in]( float )
        {
            float progress = float( in.tellg() - posStart ) / sizeAll;
            return callback( progress );
        };
    }

    ctmLoadCustom( context, []( void* buf, CTMuint size, void* data )
    {
        LoadData& loadData = *reinterpret_cast< LoadData* >( data );
        auto& stream = *loadData.stream;
        auto pos = stream.tellg();
        loadData.wasCanceled |= !readByBlocks( stream, (char*)buf, size, loadData.callbackFn, 1u << 12 );
        if ( loadData.wasCanceled )
            return 0u;
        return ( CTMuint )( stream.tellg() - pos );
    }, & loadData );

    auto vertCount = ctmGetInteger( context, CTM_VERTEX_COUNT );
    auto vertices = ctmGetFloatArray( context, CTM_VERTICES );
    if ( loadData.wasCanceled )
        return tl::make_unexpected( "Loading canceled" );
    if ( ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( "Error reading CTM format" );

    if ( colors )
    {
        auto colorAttrib = ctmGetNamedAttribMap( context, "Color" );
        if ( colorAttrib != CTM_NONE )
        {
            auto colorArray = ctmGetFloatArray( context, colorAttrib );
            colors->resize( vertCount );
            for ( VertId i{ 0 }; CTMuint( i ) < vertCount; ++i )
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

tl::expected<MR::PointCloud, std::string> fromPly( const std::filesystem::path& file, Vector<Color, VertId>* colors /*= nullptr */, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromPly( in, colors, callback );
}

tl::expected<MR::PointCloud, std::string> fromPly( std::istream& in, Vector<Color, VertId>* colors /*= nullptr */, ProgressCallback callback )
{
    MR_TIMER;

    miniply::PLYReader reader( in );
    if ( !reader.valid() )
        return tl::make_unexpected( std::string( "PLY file open error" ) );

    uint32_t indecies[3];
    bool gotVerts = false;

    std::vector<unsigned char> colorsBuffer;
    PointCloud res;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    for ( int i = 0; reader.has_element() && !gotVerts; reader.next_element(), ++i )
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
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( callback && !callback( progress ) )
                return tl::make_unexpected( std::string( "Loading canceled" ) );
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
        for ( VertId i{ 0 }; i < res.points.size(); ++i )
        {
            int ind = 3 * i;
            ( *colors )[i] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
        }
    }

    return std::move( res );
}

tl::expected<MR::PointCloud, std::string> fromPts( const std::filesystem::path& file, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromPts( in, callback );
}

tl::expected<MR::PointCloud, std::string> fromPts( std::istream& in, ProgressCallback callback )
{
    std::string line;
    int pointCount = 0;
    PointCloud cloud;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    bool isPolylineBlock{ false };

    while ( std::getline( in, line ) )
    {
        line.erase( std::find_if( line.rbegin(), line.rend(), [] ( unsigned char ch )
        {
            return !std::isspace( ch );
        } ).base(), line.end() );
        if ( !isPolylineBlock )
        {
            if ( line != "BEGIN_Polyline" )
                return tl::make_unexpected( "Not valid .pts format" );
            else
            {
                isPolylineBlock = true;
                continue;
            }
        }
        else if ( line == "END_Polyline" )
        {
            isPolylineBlock = false;
            continue;
        }

        std::istringstream iss( line );
        Vector3f point;
        if ( !( iss >> point ) )
            return tl::make_unexpected( "Not valid .pts format" );
        cloud.points.push_back( point );
        ++pointCount;

        if ( callback && !( pointCount & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !callback( progress ) )
                return tl::make_unexpected( std::string( "Loading canceled" ) );
        }
    }
    if ( isPolylineBlock )
        return tl::make_unexpected( "Not valid .pts format" );

    cloud.validPoints.resize( pointCount, true );
    return std::move( cloud );
}

tl::expected<MR::PointCloud, std::string> fromObj( const std::filesystem::path& file, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromObj( in, callback );
}

tl::expected<MR::PointCloud, std::string> fromObj( std::istream& in, ProgressCallback callback )
{
    PointCloud cloud;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    for ( int i = 0;; ++i )
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

        if ( callback && !( i & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !callback( progress ) )
                return tl::make_unexpected( std::string( "Loading canceled" ) );
        }
    }

    cloud.validPoints.resize( cloud.points.size(), true );
    return std::move( cloud );
}

tl::expected<MR::PointCloud, std::string> fromAsc( const std::filesystem::path& file, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return fromAsc( in, callback );
}

tl::expected<MR::PointCloud, std::string> fromAsc( std::istream& in, ProgressCallback callback )
{
    PointCloud cloud;
    bool allNormalsValid = true;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    for( int i = 0; in; ++i )
    {
        std::string str;
        std::getline( in, str );
        if ( str.empty() && in.eof() )
            break;
        if ( !in )
            return tl::make_unexpected( std::string( "ASC-stream read error" ) );
        if ( str.empty() )
            continue;
        if ( str[0] == '#' )
            continue; //comment line

        std::istringstream is( str );
        float x, y, z;
        is >> x >> y >> z;
        if ( !is )
            return tl::make_unexpected( std::string( "ASC-format parse error" ) );
        cloud.points.emplace_back( x, y, z );

        if ( allNormalsValid )
        {
            is >> x >> y >> z;
            if ( is )
                cloud.normals.emplace_back( x, y, z );
            else
            {
                cloud.normals = {};
                allNormalsValid = false;
            }
        }

        if ( callback && !( i & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !callback( progress ) )
                return tl::make_unexpected( std::string( "Loading canceled" ) );
        }
    }

    cloud.validPoints.resize( cloud.points.size(), true );
    return std::move( cloud );
}

tl::expected<MR::PointCloud, std::string> fromAnySupportedFormat( const std::filesystem::path& file, Vector<Color, VertId>* colors /*= nullptr */,
                                                                  ProgressCallback callback )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<MR::PointCloud, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".ply" )
        res = MR::PointsLoad::fromPly( file, colors, callback );
    else if ( ext == u8".ctm" )
        res = MR::PointsLoad::fromCtm( file, colors, callback );
    else if ( ext == u8".pts" )
        res = MR::PointsLoad::fromPts( file, callback );
    else if ( ext == u8".obj" )
        res = MR::PointsLoad::fromObj( file, callback );
    else if ( ext == u8".asc" )
        res = MR::PointsLoad::fromAsc( file, callback );
    return res;
}

tl::expected<MR::PointCloud, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, Vector<Color, VertId>* colors /*= nullptr */,
                                                                  ProgressCallback callback )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<MR::PointCloud, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".ply" )
        res = MR::PointsLoad::fromPly( in, colors, callback );
    else if ( ext == ".ctm" )
        res = MR::PointsLoad::fromCtm( in, colors, callback );
    else if ( ext == ".pts" )
        res = MR::PointsLoad::fromPts( in, callback );
    else if ( ext == ".obj" )
        res = MR::PointsLoad::fromObj( in, callback );
    else if ( ext == ".asc" )
        res = MR::PointsLoad::fromAsc( in, callback );
    return res;
}

}
}
