#include "MRPointsLoad.h"
#include "MRTimer.h"
#include "miniply.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRStreamOperators.h"
#include "MRProgressReadWrite.h"
#include "MRPointCloud.h"
#include "MRIOParsing.h"
#include "MRParallelFor.h"
#include "MRComputeBoundingBox.h"
#include <fstream>

#ifndef MRMESH_NO_OPENCTM
#include "OpenCTM/openctm.h"
#endif

namespace MR
{

namespace PointsLoad
{

const IOFilters Filters =
{
    {"All (*.*)",         "*.*"},
    {"ASC (.asc)",        "*.asc"},
    {"CSV (.csv)",        "*.csv"},
    {"XYZ (.xyz)",        "*.xyz"},
    {"OBJ (.obj)",        "*.obj"},
    {"PLY (.ply)",        "*.ply"},
    {"PTS (.pts)",        "*.pts"},
    {"DXF (.dxf)",        "*.dxf"},
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
    {"E57 (.e57)",        "*.e57"},
#endif
#ifndef MRMESH_NO_LAS
    {"LAS (.las)",        "*.las"},
    {"LASzip (.laz)",     "*.laz"},
#endif
#ifndef MRMESH_NO_OPENCTM
    {"CTM (.ctm)",        "*.ctm"},
#endif
};

Expected<MR::PointCloud, std::string> fromText( const std::filesystem::path& file, AffineXf3f* outXf, ProgressCallback callback /*= {} */ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromText( in, outXf, callback ), file );
}

Expected<MR::PointCloud, std::string> fromText( std::istream& in, AffineXf3f* outXf, ProgressCallback callback /*= {} */ )
{
    // read all to buffer
    MR_TIMER;
    auto dataExp = readCharBuffer( in );
    if ( !dataExp.has_value() )
        return unexpected( dataExp.error() );

    if ( callback && !callback( 0.25f ) )
        return unexpected( "Loading canceled" );

    const auto& data = *dataExp;
    auto lineOffsets = splitByLines( data.data(), data.size() );

    int firstLine = 0;
    Vector3d firstLineCoord;
    std::string_view headerLine( data.data() + lineOffsets[firstLine], lineOffsets[firstLine + 1] - lineOffsets[firstLine] );
    if ( !parseTextCoordinate( headerLine, firstLineCoord ).has_value() )
    {
        firstLine = 1;
        std::string_view secodLine( data.data() + lineOffsets[firstLine], lineOffsets[firstLine + 1] - lineOffsets[firstLine] );
        [[maybe_unused]] auto shiftRes = parseTextCoordinate( secodLine, firstLineCoord );
        assert( shiftRes.has_value() );
    }

    if ( outXf )
        *outXf = AffineXf3f::translation( Vector3f( firstLineCoord ) );


    PointCloud pc;
    pc.points.resize( lineOffsets.size() - firstLine - 1 );

    std::string parseError;
    tbb::task_group_context ctx;
    auto keepGoing = ParallelFor( pc.points, [&] ( size_t i )
    {
        std::string_view line( data.data() + lineOffsets[firstLine + i], lineOffsets[firstLine + i + 1] - lineOffsets[firstLine + i] );
        Vector3d tempDoubleCoord;
        auto parseRes = parseTextCoordinate( line, tempDoubleCoord );
        if ( !parseRes.has_value() && ctx.cancel_group_execution() )
            parseError = std::move( parseRes.error() );

        pc.points[VertId( i )] = Vector3f( tempDoubleCoord - firstLineCoord );
    }, subprogress( callback, 0.25f, 1.0f ) );

    if ( !keepGoing )
        return unexpected( "Loading canceled" );

    if ( !parseError.empty() )
        return unexpected( parseError );

    pc.validPoints.resize( pc.points.size(), true );
    return pc;
}

Expected<MR::PointCloud, std::string> fromPts( const std::filesystem::path& file, VertColors* colors /*= nullptr*/, 
    AffineXf3f* outXf /*= nullptr*/, ProgressCallback callback /*= {} */ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPts( in, colors, outXf, callback ), file );
}

Expected<MR::PointCloud, std::string> fromPts( std::istream& in, VertColors* colors /*= nullptr*/, 
    AffineXf3f* outXf /*= nullptr*/, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;
    std::string numPointsLine;
    if ( !std::getline( in, numPointsLine ) )
        return unexpected( "Cannot read header line" );
    auto numPoints = std::atoll( numPointsLine.c_str() );
    if ( numPoints == 0 )
        return unexpected( "Empty pts file" );

    auto dataExp = readCharBuffer( in );
    if ( !dataExp.has_value() )
        return unexpected( dataExp.error() );

    if ( callback && !callback( 0.25f ) )
        return unexpected( "Loading canceled" );

    const auto& data = *dataExp;
    auto lineOffsets = splitByLines( data.data(), data.size() );

    int firstLine = 1;
    Vector3d firstLineCoord;
    Color firstLineColor;
    std::string_view shitLine( data.data() + lineOffsets[firstLine], lineOffsets[firstLine + 1] - lineOffsets[firstLine] );
    auto shiftLineRes = parsePtsCoordinate( shitLine, firstLineCoord, firstLineColor );
    if ( !shiftLineRes.has_value() )
        return unexpected( shiftLineRes.error() );

    if ( outXf )
        *outXf = AffineXf3f::translation( Vector3f( firstLineCoord ) );

    if ( colors )
        colors->resize( lineOffsets.size() - firstLine - 1 );

    PointCloud pc;
    pc.points.resize( lineOffsets.size() - firstLine - 1 );

    std::string parseError;
    tbb::task_group_context ctx;
    auto keepGoing = ParallelFor( pc.points, [&] ( size_t i )
    {
        std::string_view line( data.data() + lineOffsets[firstLine + i], lineOffsets[firstLine + i + 1] - lineOffsets[firstLine + i] );
        Vector3d tempDoubleCoord;
        Color tempColor;
        auto parseRes = parsePtsCoordinate( line, tempDoubleCoord, tempColor );
        if ( !parseRes.has_value() && ctx.cancel_group_execution() )
            parseError = std::move( parseRes.error() );

        pc.points[VertId( i )] = Vector3f( tempDoubleCoord - firstLineCoord );
        if ( colors )
            ( *colors )[VertId( i )] = tempColor;
    }, subprogress( callback, 0.25f, 1.0f ) );

    if ( !keepGoing )
        return unexpected( "Loading canceled" );

    if ( !parseError.empty() )
        return unexpected( parseError );

    pc.validPoints.resize( pc.points.size(), true );
    return pc;
}

#ifndef MRMESH_NO_OPENCTM

Expected<MR::PointCloud, std::string> fromCtm( const std::filesystem::path& file, VertColors* colors /*= nullptr */, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromCtm( in, colors, callback ), file );
}

Expected<MR::PointCloud, std::string> fromCtm( std::istream& in, VertColors* colors /*= nullptr */, ProgressCallback callback )
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
        return unexpected( "Loading canceled" );
    if ( ctmGetError( context ) != CTM_NONE )
        return unexpected( "Error reading CTM format" );

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
#endif

Expected<MR::PointCloud, std::string> fromPly( const std::filesystem::path& file, VertColors* colors /*= nullptr */, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPly( in, colors, callback ), file );
}

Expected<MR::PointCloud, std::string> fromPly( std::istream& in, VertColors* colors /*= nullptr */, ProgressCallback callback )
{
    MR_TIMER;

    miniply::PLYReader reader( in );
    if ( !reader.valid() )
        return unexpected( std::string( "PLY file open error" ) );

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
            if ( reader.find_normal( indecies ) )
            {
                Timer t( "extractNormals" );
                res.normals.resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, res.normals.data() );
            }
            if ( colors && reader.find_color( indecies ) )
            {
                colorsBuffer.resize( 3 * numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
            }
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( callback && !callback( progress ) )
                return unexpected( std::string( "Loading canceled" ) );
            continue;
        }
    }

    if ( !reader.valid() )
        return unexpected( std::string( "PLY file read or parse error" ) );

    if ( !gotVerts )
        return unexpected( std::string( "PLY file does not contain vertices" ) );

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

Expected<MR::PointCloud, std::string> fromObj( const std::filesystem::path& file, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromObj( in, callback ), file );
}

Expected<MR::PointCloud, std::string> fromObj( std::istream& in, ProgressCallback callback )
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
            return unexpected( std::string( "OBJ-format read error" ) );
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
                return unexpected( std::string( "Loading canceled" ) );
        }
    }

    cloud.validPoints.resize( cloud.points.size(), true );
    return std::move( cloud );
}

Expected<MR::PointCloud, std::string> fromAsc( const std::filesystem::path& file, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromAsc( in, callback ), file );
}

Expected<MR::PointCloud, std::string> fromAsc( std::istream& in, ProgressCallback callback )
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
            return unexpected( std::string( "ASC-stream read error" ) );
        if ( str.empty() )
            continue;
        if ( str[0] == '#' )
            continue; //comment line

        std::istringstream is( str );
        float x, y, z;
        is >> x >> y >> z;
        if ( !is )
            return unexpected( std::string( "ASC-format parse error" ) );
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
                return unexpected( std::string( "Loading canceled" ) );
        }
    }

    cloud.validPoints.resize( cloud.points.size(), true );
    return std::move( cloud );
}

Expected<MR::PointCloud, std::string> fromDxf( const std::filesystem::path& file, ProgressCallback callback )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromDxf( in, callback ), file );
}

Expected<MR::PointCloud, std::string> fromDxf( std::istream& in, ProgressCallback cb )
{
    PointCloud cloud;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    std::string str;
    std::getline( in, str );

    int code{};
    if ( !parseSingleNumber<int>( str, code ) )
        return unexpected( "File is corrupted" );

    bool isPointFound = false;

    for ( int i = 0; !in.eof(); ++i )
    {
        if ( i % 1024 == 0 && !reportProgress( cb, float( in.tellg() ) / streamSize ) )
            return unexpectedOperationCanceled();

        std::getline( in, str );

        if ( str == "POINT" )
        {
            cloud.points.emplace_back();
            isPointFound = true;
        }

        if ( isPointFound )
        {
            const int vIdx = code % 10;
            const int cIdx = code / 10 - 1;
            if ( vIdx == 0 && cIdx >= 0 && cIdx < 3 )
            {
                if ( !parseSingleNumber<float>( str, cloud.points.back()[cIdx] ) )
                    return unexpected( "File is corrupted" );
            }
        }

        std::getline( in, str );
        if ( str.empty() )
            continue;
        
        if ( !parseSingleNumber<int>( str, code ) )
            return unexpected( "File is corrupted" );

        if ( code == 0 )
            isPointFound = false;
    }

    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( cloud.points.empty() )
        return unexpected( "No points are found " );

    cloud.validPoints.resize( cloud.points.size(), true );
    return cloud;
}

Expected<PointCloud, std::string> fromAnySupportedFormat( const std::filesystem::path& file, VertColors* colors,
                                                          AffineXf3f* outXf, ProgressCallback callback )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    Expected<MR::PointCloud, std::string> res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".ply" )
        res = MR::PointsLoad::fromPly( file, colors, callback );
    else if ( ext == ".pts" )
        res = MR::PointsLoad::fromPts( file, colors, outXf, callback );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
        res = MR::PointsLoad::fromCtm( file, colors, callback );
#endif
    else if ( ext == ".obj" )
        res = MR::PointsLoad::fromObj( file, callback );
    else if ( ext == ".asc" )
        res = MR::PointsLoad::fromAsc( file, callback );
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
    else if ( ext == ".e57" )
        res = MR::PointsLoad::fromE57( file, colors, outXf, callback );
#endif
#if !defined( MRMESH_NO_LAS )
    else if ( ext == ".las" || ext == ".laz" )
        res = MR::PointsLoad::fromLas( file, colors, outXf, callback );
#endif
    else if ( ext == ".csv" || ext == ".xyz" )
        res = MR::PointsLoad::fromText( file, outXf, callback );
    else if ( ext == ".dxf" )
        res = MR::PointsLoad::fromDxf( file, callback );
    return res;
}

Expected<PointCloud, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension,
                                                          VertColors* colors, AffineXf3f* outXf,
                                                          ProgressCallback callback )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
    assert( ext != ".e57" ); // no support for reading e57 from arbitrary stream yet
#endif

    Expected<MR::PointCloud, std::string> res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".ply" )
        res = MR::PointsLoad::fromPly( in, colors, callback );
    else if ( ext == ".pts" )
        res = MR::PointsLoad::fromPts( in, colors, outXf, callback );
#ifndef MRMESH_NO_OPENCTM
    else if ( ext == ".ctm" )
        res = MR::PointsLoad::fromCtm( in, colors, callback );
#endif
    else if ( ext == ".obj" )
        res = MR::PointsLoad::fromObj( in, callback );
    else if ( ext == ".asc" )
        res = MR::PointsLoad::fromAsc( in, callback );
#if !defined( MRMESH_NO_LAS )
    else if ( ext == ".las" || ext == ".laz" )
        res = MR::PointsLoad::fromLas( in, colors, outXf, callback );
#endif
    else if ( ext == ".csv" || ext == ".xyz" )
        res = MR::PointsLoad::fromText( in, outXf, callback );
    else if ( ext == ".dxf" )
        res = MR::PointsLoad::fromDxf( in, callback );
    return res;
}

}
}
