#include "MRMeshLoadObj.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRMeshBuilder.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRphmap.h"
#include "MRPch/MRTBB.h"

#include <boost/algorithm/string/trim.hpp>
#include <boost/spirit/home/x3.hpp>

#include <map>

namespace
{
    using namespace MR;

    std::vector<size_t> splitByLines( const char* data, size_t size )
    {
        constexpr size_t blockSize = 4096;
        const auto blockCount = (size_t)std::ceil( (float)size / blockSize );

        constexpr size_t maxGroupCount = 256;
        const auto blocksPerGroup = (size_t)std::ceil( (float)blockCount / maxGroupCount );
        const auto groupSize = blockSize * blocksPerGroup;
        const auto groupCount = (size_t)std::ceil( (float)size / groupSize );
        assert( groupCount <= maxGroupCount );
        assert( groupSize * groupCount >= size );
        assert( groupSize * ( groupCount - 1 ) < size );

        std::vector<std::vector<size_t>> groups( groupCount );
        tbb::task_group taskGroup;
        for ( size_t gi = 0; gi < groupCount; gi++ )
        {
            taskGroup.run( [&, i = gi]
            {
                std::vector<size_t> group;
                const auto begin = i * groupSize;
                const auto end = std::min( ( i + 1 ) * groupSize, size );
                for ( auto ci = begin; ci < end; ci++ )
                    if ( data[ci] == '\n' )
                        group.emplace_back( ci + 1 );
                groups[i] = std::move( group );
            } );
        }
        taskGroup.wait();

        std::vector<size_t> newlines{ 0 };
        auto sum = newlines.size();
        std::vector<size_t> groupOffsets;
        for ( const auto& group : groups )
        {
            groupOffsets.emplace_back( sum );
            sum += group.size();
        }
        newlines.resize( sum );

        for ( size_t gi = 0; gi < groupCount; gi++ )
        {
            taskGroup.run( [&, i = gi]
            {
                const auto& group = groups[i];
                const auto offset = groupOffsets[i];
                for ( auto li = 0; li < group.size(); li++ )
                    newlines[offset + li] = group[li];
            } );
        }
        taskGroup.wait();

        // add finish line
        if ( newlines.back() != size )
            newlines.emplace_back( size );

        return newlines;
    }

    enum class ObjElement
    {
        Unknown,
        Vertex,
        Face,
        Object,
        TextureVertex,
        MaterialLibrary,
        MaterialName,
    };

    enum class MtlElement
    {
        Unknown,
        MaterialName,
        AmbientColor,
        DiffuseColor,
        SpecularColor,
        SpecularExponent,
        Dissolve,
        IlluminationModel,
        AmbientTexture,
        DiffuseTexture,
        SpecularTexture,
        SpecularExponentTexture,
        DissolveTexture,
    };

    template <typename T>
    T parseToken( std::string_view line );

    template <>
    ObjElement parseToken<ObjElement>( std::string_view line )
    {
        while ( !line.empty() && line[0] == ' ' )
            line.remove_prefix( 1 );

        assert( !line.empty() );
        switch ( line[0] )
        {
        case 'v':
            assert( line.size() >= 2 );
            switch ( line[1] )
            {
            case ' ':
                return ObjElement::Vertex;
            case 't':
                return ObjElement::TextureVertex;
            default:
                return ObjElement::Unknown;
            }
        case 'f':
            return ObjElement::Face;
        case 'o':
            return ObjElement::Object;
        case 'u':
            if ( line.starts_with( "usemtl" ) )
                return ObjElement::MaterialName;
            return ObjElement::Unknown;
        case 'm':
            if ( line.starts_with( "mtllib" ) )
                return ObjElement::MaterialLibrary;
            return ObjElement::Unknown;
        default:
            return ObjElement::Unknown;
        }
    }

    template <>
    MtlElement parseToken<MtlElement>( std::string_view line )
    {
        while ( !line.empty() && line[0] == ' ' )
            line.remove_prefix( 1 );

        assert( !line.empty() );
        switch ( line[0] )
        {
        case 'n':
            if ( line.starts_with( "newmtl" ) )
                return MtlElement::MaterialName;
            else
                return MtlElement::Unknown;
        case 'K':
            assert( line.size() >= 2 );
            switch ( line[1] )
            {
            case 'a':
                return MtlElement::AmbientColor;
            case 'd':
                return MtlElement::DiffuseColor;
            case 's':
                return MtlElement::SpecularColor;
            default:
                return MtlElement::Unknown;
            }
        case 'm':
            if ( line.starts_with( "map_" ) )
            {
                assert( line.size() >= 5 );
                switch ( line[4] )
                {
                case 'K':
                    assert( line.size() >= 6 );
                    switch ( line[5] )
                    {
                    case 'a':
                        return MtlElement::AmbientTexture;
                    case 'd':
                        return MtlElement::DiffuseTexture;
                    case 's':
                        return MtlElement::SpecularTexture;
                    default:
                        return MtlElement::Unknown;
                    }
                default:
                    return MtlElement::Unknown;
                }
            }
            else
            {
                return MtlElement::Unknown;
            }
        default:
            return MtlElement::Unknown;
        }
    }

    template <typename Element>
    struct ElementGroup
    {
        Element element{ Element() };
        size_t begin{ 0 };
        size_t end{ 0 };
    };

    template <typename Element>
    std::vector<ElementGroup<Element>> groupLines( const char* data, size_t, const std::vector<size_t>& newlines )
    {
        const auto lineCount = newlines.size() - 1;

        std::vector<ElementGroup<Element>> groups{ { Element(), 0, 0 } }; // emplace stub initial group
        for ( size_t li = 0; li < lineCount; li++ )
        {
            std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
            const auto element = parseToken<Element>( line );
            if ( element != groups.back().element )
            {
                groups.back().end = li;
                groups.push_back( { element, li, 0 } );
            }
        }
        groups.back().end = lineCount;
        return groups;
    }

    VoidOrErrStr parseObjVertex( const std::string_view& str, Vector3f& v )
    {
        using namespace boost::spirit::x3;

        int i = 0;
        auto coord = [&] ( auto& ctx ) { v[i++] = _attr( ctx ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            ( 'v' >> float_[coord] >> float_[coord] >> float_[coord] ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse vertex in OBJ-file" );

        return {};
    }

    VoidOrErrStr parseObjTextureVertex( const std::string_view& str, UVCoord& vt )
    {
        using namespace boost::spirit::x3;

        std::array<float, 3> coords{ 0.f, 0.f, 0.f };
        int i = 0;
        auto coord = [&] ( auto& ctx ) { coords[i++] = _attr( ctx ); };

        bool r = phrase_parse(
                str.begin(),
                str.end(),
                ( lit( "vt" ) >> float_[coord] >> -( float_[coord] >> -( float_[coord] ) ) ),
                ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse vertex in OBJ-file" );

        vt = { coords[0], coords[1] };
        return {};
    }

    struct ObjFace
    {
        std::vector<int> vertices;
        std::vector<int> textures;
        std::vector<int> normals;
    };

    VoidOrErrStr parseObjFace( const std::string_view& str, ObjFace& f )
    {
        using namespace boost::spirit::x3;

        auto v = [&] ( auto& ctx ) { f.vertices.emplace_back( _attr( ctx ) ); };
        auto vt = [&] ( auto& ctx ) { f.textures.emplace_back( _attr( ctx ) ); };
        auto vn = [&] ( auto& ctx ) { f.normals.emplace_back( _attr( ctx ) ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            // NOTE: actions are not being reverted after backtracking
            // https://github.com/boostorg/spirit/issues/378
            ( 'f' >> *( int_[v] >> -( '/' >> ( ( int_[vt] >> -( '/' >> int_[vn] ) ) | ( '/' >> int_[vn] ) ) ) ) ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse face in OBJ-file" );

        if ( f.vertices.empty() )
            return tl::make_unexpected( "Invalid face vertex count in OBJ-file" );
        if ( !f.textures.empty() && f.textures.size() != f.vertices.size() )
            return tl::make_unexpected( "Invalid face texture count in OBJ-file" );
        if ( !f.normals.empty() && f.normals.size() != f.vertices.size() )
            return tl::make_unexpected( "Invalid face normal count in OBJ-file" );
        return {};
    }

    VoidOrErrStr parseMtlColor( const std::string_view& str, Vector3f& color )
    {
        using namespace boost::spirit::x3;

        auto r = [&] ( auto& ctx ) { color.x = _attr( ctx ); };
        auto g = [&] ( auto& ctx ) { color.y = _attr( ctx ); };
        auto b = [&] ( auto& ctx ) { color.z = _attr( ctx ); };

        bool res = phrase_parse(
            str.begin(),
            str.end(),
            (
                char_( 'K' ) >> ( char_( 'a' ) | char_( 'd' ) | char_( 's' ) )
                >> (
                        ( float_[r] >> float_[g] >> float_[b] ) |
                        ( lit( "spectral" ) >> no_skip[+char_] ) |
                        ( lit( "xyz" ) >> float_ >> float_ >> float_ )
                    )
            ),
            space
        );
        if ( !res )
            return tl::make_unexpected( "Failed to parse color in MTL-file" );

        return {};
    }

    VoidOrErrStr parseMtlTexture( const std::string_view& str, std::string& textureFile )
    {
        using namespace boost::spirit::x3;

        auto file = [&] ( auto& ctx ) { textureFile = _attr( ctx ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            (
                lit( "map_") >> ( lit( "Ka" ) | lit( "Kd" ) | lit( "Ks" ) | lit( "Ns" ) | lit( "d" ) )
                >> *(
                        ( lit( "-blendu" ) >> ( lit( "on" ) | lit( "off" ) ) ) |
                        ( lit( "-blendv" ) >> ( lit( "on" ) | lit( "off" ) ) ) |
                        ( lit( "-cc" ) >> ( lit( "on" ) | lit( "off" ) ) ) |
                        ( lit( "-clamp" ) >> ( lit( "on" ) | lit( "off" ) ) ) |
                        ( lit( "-imfchan" ) >> ( char_( 'r' ) | char_( 'b' ) | char_( 'g' ) | char_( 'm' ) | char_( 'l' ) | char_( 'z' ) ) ) |
                        ( lit( "-mm" ) >> float_ >> float_ ) |
                        ( lit( "-bm" ) >> float_ ) |
                        ( lit( "-o" ) >> float_ >> float_ >> float_ ) |
                        ( lit( "-s" ) >> float_ >> float_ >> float_ ) |
                        ( lit( "-t" ) >> float_ >> float_ >> float_ ) |
                        ( lit( "-texres" ) >> float_ )
                    )
                >> no_skip[+char_][file]
            ),
            space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse texture in MTL-file" );

        boost::trim( textureFile );
        return {};
    }

    struct MtlMaterial
    {
        Vector3f diffuseColor = Vector3f::diagonal( -1.0f );
        std::string diffuseTextureFile;
    };

    using MtlLibrary = HashMap<std::string, MtlMaterial>;

    tl::expected<MtlLibrary, std::string> loadMtlLibrary( const std::filesystem::path& path )
    {
        std::ifstream mtlIn( path, std::ios::binary );
        if ( !mtlIn.is_open() )
            return tl::make_unexpected( "unable to open MTL file" );

        const auto posStart = mtlIn.tellg();
        mtlIn.seekg( 0, std::ios_base::end );
        const auto posEnd = mtlIn.tellg();
        mtlIn.seekg( posStart );
        const auto mtlSize = posEnd - posStart;
        std::string mtlContent( mtlSize, '\0' );
        const auto data = mtlContent.data();
        mtlIn.read( data, mtlSize );

        const auto newlines = splitByLines( data, mtlSize );

        const auto groups = groupLines<MtlElement>( data, mtlSize, newlines );

        MtlLibrary result;
        std::string currentMaterialName;
        MtlMaterial currentMaterial;

        for ( const auto& group : groups )
        {
            std::string parseError;
            switch ( group.element )
            {
            case MtlElement::Unknown:
                break;
            case MtlElement::MaterialName:
            {
                const auto li = group.end - 1;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );

                if ( !currentMaterialName.empty() )
                {
                    result.emplace( std::move( currentMaterialName ), std::move( currentMaterial ) );
                    currentMaterial = {};
                }
                currentMaterialName = line.substr( 6, std::string_view::npos );
                boost::trim( currentMaterialName );
            }
            break;
            case MtlElement::DiffuseColor:
            {
                const auto li = group.end - 1;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );

                auto res = parseMtlColor( line, currentMaterial.diffuseColor );
                if ( !res.has_value() )
                    parseError = std::move( res.error() );
            }
            break;
            case MtlElement::DiffuseTexture:
            {
                const auto li = group.end - 1;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );

                auto res = parseMtlTexture( line, currentMaterial.diffuseTextureFile );
                if ( !res.has_value() )
                    parseError = std::move( res.error() );
            }
            break;
            default:
                break;
            }
            if ( !parseError.empty() )
                return tl::make_unexpected( parseError );
        }

        if ( !currentMaterialName.empty() )
            result.emplace( std::move( currentMaterialName ), std::move( currentMaterial ) );
        return result;
    }
}

namespace MR
{

namespace MeshLoad
{

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromSceneObjFile( in, combineAllObjects, file.parent_path(), callback ), file );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects, const std::filesystem::path& dir,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const auto streamSize = posEnd - posStart;

    Buffer<char> data( streamSize );
    // important on Windows: in stream must be open in binary mode, otherwise next will fail
    in.read( data.data(), (ptrdiff_t)data.size() );
    if ( !in )
        return tl::make_unexpected( std::string( "OBJ-format read error" ) );

    if ( callback && !callback( 0.25f ) )
        return tl::make_unexpected( "Loading canceled" );
    // TODO: redefine callback

    return fromSceneObjFile( data.data(), data.size(), combineAllObjects, dir, callback );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects, const std::filesystem::path& dir,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    std::vector<NamedMesh> res;
    std::string currentObjName;
    std::vector<Vector3f> points;
    std::vector<UVCoord> textureVertices;
    std::vector<int> texCoords( points.size(), -1 );
    Triangulation t;
    Vector<UVCoord, VertId> uvCoords;
    tl::expected<MtlLibrary, std::string> mtl;
    std::string currentMaterialName;

    std::map<int, int> additions;
    additions[0] = 0;
    int originalPointCount = 0;

    auto finishObject = [&]() 
    {
        MR_NAMED_TIMER( "finish object" )
        if ( !t.empty() )
        {
            auto& result = res.emplace_back();
            result.name = std::move( currentObjName );

            // copy only minimal span of vertices for this object
            VertId minV(INT_MAX), maxV(-1);
            for ( const auto & vs : t )
            {
                minV = std::min( { minV, vs[0], vs[1], vs[2] } );
                maxV = std::max( { maxV, vs[0], vs[1], vs[2] } );
            }
            for ( auto & vs : t )
            {
                for ( int i = 0; i < 3; ++i )
                    vs[i] -= minV;
            }

            std::vector<MeshBuilder::VertDuplication> dups;
            result.mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices(
                VertCoords( points.begin() + minV, points.begin() + maxV + 1 ), t, &dups );
            t.clear();


            VertHashMap dst2Src;
            dst2Src.reserve( dups.size() );
            for ( const auto& dup : dups )
                dst2Src.emplace( dup.dupVert, dup.srcVert );

            if ( mtl.has_value() && ! mtl->empty() )
            {
                auto materialIt = mtl->find( currentMaterialName );

                if ( materialIt == mtl->end() )
                {
                    materialIt = mtl->find( "default" );
                }
                if ( materialIt == mtl->end() )
                {
                    materialIt = mtl->begin();
                }
               
                if ( !materialIt->second.diffuseTextureFile.empty() )
                {
                    result.pathToTexture = dir / materialIt->second.diffuseTextureFile;
                }

                if ( materialIt->second.diffuseColor != Vector3f::diagonal(-1) )
                    result.diffuseColor = Color( materialIt->second.diffuseColor );

                if ( !texCoords.empty() )
                {
                    uvCoords.resize( points.size() );
                    tbb::parallel_for( tbb::blocked_range<VertId>( VertId( 0 ), VertId( points.size() ) ), [&] ( const tbb::blocked_range<VertId>& range )
                    {
                        for ( VertId i = range.begin(); i < range.end(); ++i )
                        {
                            if ( texCoords[i] < uvCoords.size() )
                                uvCoords[i] = textureVertices[texCoords[i]];
                        }
                    } );

                    assert( uvCoords.size() >= maxV );
                    result.uvCoords = { begin( uvCoords ) + minV, begin( uvCoords ) + maxV + 1 };
                    result.uvCoords.resize( result.mesh.points.size() );
                    for ( const auto& dup : dups )
                        result.uvCoords[dup.dupVert] = result.uvCoords[dup.srcVert];
                }
            }
        }
        currentObjName.clear();
    };

    Timer timer( "split by lines" );
    const auto newlines = splitByLines( data, size );
    const auto lineCount = newlines.size() - 1;

    if ( callback && !callback( 0.40f ) )
        return tl::make_unexpected( "Loading canceled" );

    timer.restart( "group element lines" );
    const auto groups = groupLines<ObjElement>( data, size, newlines );

    if ( callback && !callback( 0.50f ) )
        return tl::make_unexpected( "Loading canceled" );

    auto parseVertices = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        const auto offset = points.size();
        originalPointCount += int( end - begin );
        const size_t newSize = points.size() + ( end - begin );
        texCoords.resize( newSize );
        std::fill( texCoords.begin() + points.size(), texCoords.end(), -1 );

        points.resize( newSize );        
        uvCoords.resize( newSize );

        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            Vector3f v;
            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjVertex( line, v );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
                points[offset + ( li - begin )] = v;
            }
        }, ctx );
    };

    auto parseTextureVertices = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        const auto offset = textureVertices.size();
        textureVertices.resize( textureVertices.size() + ( end - begin ) );

        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            UVCoord vt;
            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjTextureVertex( line, vt );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
                textureVertices[offset + ( li - begin )] = vt;
            }
        }, ctx );
    };

    auto parseFaces = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        tbb::task_group_context ctx;
        tbb::enumerable_thread_specific<Triangulation> trisPerThread;

        std::mutex mutex;

        int newPoints = 0;
        int lastAddition = 0;
        if ( !additions.empty() )
            lastAddition = additions.rbegin()->second;

        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            auto& tris = trisPerThread.local();

            ObjFace f;
            // usually a face has 3 or 4 vertices
            for ( auto* elements : { &f.vertices, &f.textures, &f.normals } )
               elements->reserve( 4 );

            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                for ( auto* elements : { &f.vertices, &f.textures, &f.normals } )
                    elements->clear();

                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjFace( line, f );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }

                auto& vs = f.vertices;
                for ( auto& v : vs )
                {
                    if ( --v < 0 )
                    {
                        v += originalPointCount + 1;

                        if ( v < 0 )
                        {
                            if ( ctx.cancel_group_execution() )
                                parseError = "Too negative vertex ID in OBJ-file";
                            return;
                        }
                    }
                   
                    auto it = additions.upper_bound( v );
                    int toAdd = 0;
                    if ( it == additions.end() )
                        toAdd = additions.rbegin()->second;
                    else if ( it != additions.begin() )
                        toAdd = (--it)->second;

                    if ( toAdd > 0 )
                        v += toAdd;
                   
                }
                if ( vs.size() < 3 )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = "Face with less than 3 vertices in OBJ-file";
                    return;
                }
                
                auto& vts = f.textures;
                if ( !vts.empty() )
                {
                    for ( auto& vt : vts )
                    {
                        if ( vt < 0 )
                            vt += int( textureVertices.size() ) + 1;

                        if ( --vt < 0 )
                        {
                            if ( ctx.cancel_group_execution() )
                                parseError = "Too negative texture vertex ID in OBJ-file";
                            return;
                        }
                    }

                    assert( vts.size() == vs.size() );
                    const std::lock_guard<std::mutex> lock( mutex );
                    for ( int j = 0; j < vs.size(); ++j )
                    {
                        if ( texCoords[vs[j]] == vts[j] )
                            continue;

                        if ( texCoords[vs[j]] < 0 )
                        {
                            texCoords[vs[j]] = vts[j];
                            continue;
                        }
                        points.push_back( points[vs[j]] );
                        texCoords.push_back( vts[j] );
                        vs[j] = int( points.size() ) - 1;
                        ++newPoints;
                    }
                }

                // TODO: make smarter triangulation based on point coordinates
                for ( int j = 1; j + 1 < vs.size(); ++j )
                    tris.push_back( { VertId( vs[0] ), VertId( vs[j] ), VertId( vs[j+1] ) } );
            }
        }, ctx );

        additions.insert_or_assign( originalPointCount, lastAddition + newPoints );

        if ( !parseError.empty() )
            return;

        size_t trisSize = 0;
        for ( auto& tris : trisPerThread )
            trisSize += tris.size();
        t.reserve( t.size() + trisSize );
        for ( auto& tris : trisPerThread )
            t.vec_.insert( t.vec_.end(), tris.vec_.begin(), tris.vec_.end() );
    };

    auto parseObject = [&] ( size_t, size_t end, std::string& )
    {
        if ( combineAllObjects )
            return;

        // finish previous object
        finishObject();

        const auto li = end - 1;
        std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
        currentObjName = line.substr( strlen( "o" ), std::string_view::npos );
        boost::trim( currentObjName );
    };

    auto parseMaterialLibrary = [&] ( size_t, size_t end, std::string& )
    {
        const auto li = end - 1;
        std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
        // TODO: support multiple files
        std::string filename ( line.substr( strlen( "mtllib" ), std::string_view::npos ) );
        boost::trim( filename );
        mtl = loadMtlLibrary( dir / filename );
    };

    auto parseMaterialName = [&] ( size_t, size_t end, std::string& )
    {
        const auto li = end - 1;
        std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
        currentMaterialName = line.substr( strlen( "usemtl" ), std::string_view::npos );
        boost::trim( currentMaterialName );
    };

    timer.restart( "parse groups" );
    for ( const auto& group : groups )
    {
        std::string parseError;
        switch ( group.element )
        {
        case ObjElement::Unknown:
            break;
        case ObjElement::Vertex:
            parseVertices( group.begin, group.end, parseError );
            break;
        case ObjElement::Face:
            parseFaces( group.begin, group.end, parseError );
            break;
        case ObjElement::Object:
            parseObject( group.begin, group.end, parseError );
            break;
        case ObjElement::TextureVertex:
            parseTextureVertices( group.begin, group.end, parseError );
            break;
        case ObjElement::MaterialLibrary:
            parseMaterialLibrary( group.begin, group.end, parseError );
            break;
        case ObjElement::MaterialName:
            parseMaterialName( group.begin, group.end, parseError );
            break;
        }
        if ( !parseError.empty() )
            return tl::make_unexpected( parseError );

        if ( callback && !callback( 0.50f + 0.50f * ( (float)group.end / (float)lineCount ) ) )
            return tl::make_unexpected( "Loading canceled" );
    }

    finishObject();
    return res;
}

} //namespace MeshLoad

} //namespace MR
