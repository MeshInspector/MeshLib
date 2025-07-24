#include "MRMeshLoadObj.h"
#include "MRAffineXf3.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRComputeBoundingBox.h"
#include "MRIOFormatsRegistry.h"
#include "MRIOParsing.h"
#include "MRImageLoad.h"
#include "MRMeshBuilder.h"
#include "MRObjectMesh.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRphmap.h"
#include "MRString.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRTBB.h"
#include "MRParallelFor.h"

#include <boost/algorithm/string/trim.hpp>
#include <boost/spirit/home/x3.hpp>

#include <map>

namespace
{
using namespace MR;

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

Expected<void> parseObjTextureVertex( const std::string_view& str, UVCoord& vt )
{
    using namespace boost::spirit::x3;

    std::array<float, 3> coords{ 0.f, 0.f, 0.f };
    int i = 0;
    auto coord = [&] ( auto& ctx )
    {
        coords[i++] = _attr( ctx );
    };

    bool r = phrase_parse(
            str.begin(),
            str.end(),
            ( lit( "vt" ) >> float_[coord] >> -( float_[coord] >> -( float_[coord] ) ) ),
            ascii::space
    );

    static constexpr int MaxErrorStringLen = 80;
    if ( !r )
        return unexpected( "Failed to parse vertex in OBJ-file: " + std::string( trimRight( str.substr( 0, MaxErrorStringLen ) ) ) );

    vt = { coords[0], coords[1] };
    return {};
}

struct ObjFace
{
    std::vector<int> vertices;
    std::vector<int> textures;
    std::vector<int> normals;
};

Expected<void> parseObjFace( const std::string_view& str, ObjFace& f )
{
    using namespace boost::spirit::x3;

    auto v = [&] ( auto& ctx )
    {
        f.vertices.emplace_back( _attr( ctx ) );
    };
    auto vt = [&] ( auto& ctx )
    {
        f.textures.emplace_back( _attr( ctx ) );
    };
    auto vn = [&] ( auto& ctx )
    {
        f.normals.emplace_back( _attr( ctx ) );
    };

    bool r = phrase_parse(
        str.begin(),
        str.end(),
        // NOTE: actions are not being reverted after backtracking
        // https://github.com/boostorg/spirit/issues/378
        ( 'f' >> *( int_[v] >> -( '/' >> ( ( int_[vt] >> -( '/' >> int_[vn] ) ) | ( '/' >> int_[vn] ) ) ) ) ),
        ascii::space
    );
    if ( !r )
        return unexpected( "Failed to parse face in OBJ-file" );

    if ( f.vertices.empty() )
        return unexpected( "Invalid face vertex count in OBJ-file" );
    if ( !f.textures.empty() && f.textures.size() != f.vertices.size() )
        return unexpected( "Invalid face texture count in OBJ-file" );
    if ( !f.normals.empty() && f.normals.size() != f.vertices.size() )
        return unexpected( "Invalid face normal count in OBJ-file" );
    return {};
}

Expected<void> parseMtlColor( const std::string_view& str, Vector3f& color )
{
    using namespace boost::spirit::x3;

    auto r = [&] ( auto& ctx )
    {
        color.x = _attr( ctx );
    };
    auto g = [&] ( auto& ctx )
    {
        color.y = _attr( ctx );
    };
    auto b = [&] ( auto& ctx )
    {
        color.z = _attr( ctx );
    };

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
        return unexpected( "Failed to parse color in MTL-file" );

    return {};
}

Expected<void> parseMtlTexture( const std::string_view& str, std::string& textureFile )
{
    using namespace boost::spirit::x3;

    auto file = [&] ( auto& ctx )
    {
        textureFile = _attr( ctx );
    };

    bool r = phrase_parse(
        str.begin(),
        str.end(),
        (
            lit( "map_" ) >> ( lit( "Ka" ) | lit( "Kd" ) | lit( "Ks" ) | lit( "Ns" ) | lit( "d" ) )
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
        return unexpected( "Failed to parse texture in MTL-file" );

    boost::trim( textureFile );
    return {};
}

struct MtlMaterial
{
    Vector3f diffuseColor = Vector3f::diagonal( -1.0f );
    std::string diffuseTextureFile;
};

using MtlLibrary = HashMap<std::string, MtlMaterial>;

Expected<MtlLibrary> loadMtlLibrary( const std::filesystem::path& path )
{
    std::ifstream mtlIn( path, std::ios::binary );
    if ( !mtlIn.is_open() )
        return unexpected( "unable to open MTL file" );

    const auto mtlContent = readCharBuffer( mtlIn );
    if ( !mtlContent )
        return unexpected( "Unable to open MTL file: " + mtlContent.error() );
    const auto* data = mtlContent->data();
    const auto mtlSize = mtlContent->size();

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
            return unexpected( parseError );
    }

    if ( !currentMaterialName.empty() )
        result.emplace( std::move( currentMaterialName ), std::move( currentMaterial ) );
    return result;
}

struct MaterialScope
{
    size_t fId{ 0 }; // material begins with this face
    std::string mtName;
};
struct ObjectScope
{
    size_t fId{ 0 }; // object begins with this face
    std::string objName;
};


struct VertexRepr
{
    int vId{ 0 };
    int vtId{ 0 };
    // int vnId{0}; // not used yet
    bool operator==( const VertexRepr& o ) const = default;
    bool operator<( const VertexRepr& o ) const
    {
        return std::tuple( vId, vtId ) < std::tuple( o.vId, o.vtId );
    }
};

struct VertexReprHasher
{
    size_t operator()( VertexRepr const& vr ) const noexcept
    {
        std::uint64_t vvt;
        std::memcpy( &vvt, &vr.vId, sizeof( std::uint64_t ) );
        return size_t( vvt );
    }
};

Expected<MeshLoad::NamedMesh> loadSingleModelFromObj(
    const std::filesystem::path& dir,
    const Vector<Vector3d, VertId>& points,  // all points from file
    const std::vector<Color>& colors,     // all colors from file
    const std::vector<UVCoord>& uvCoords, // all uvs from file
    const std::vector<ObjFace>& faces,    // all faces from file
    const std::vector<MaterialScope>& materialScope, // all material scopes from file
    size_t minFace, size_t maxFace,       // this model faces span in `faces`, max face excluding
    const MeshLoad::ObjLoadSettings& settings,
    const MtlLibrary* mtl ) // optional materials, if nullptr `materialScope` will be ignored
{
    MR_TIMER;

    bool haveColors = false;
    bool haveUVs = false;
    int firstVert = -1;
    if ( minFace < faces.size() ) // do not crash if minFace = 0 and faces are empty
    {
        haveUVs = !faces[minFace].textures.empty();
        firstVert = faces[minFace].vertices.front();
    }
    if ( firstVert < 0 )
        firstVert = int( points.size() ) - firstVert;
    if ( firstVert < 0 || firstVert >= points.size() )
        return unexpected( "Out of bounds Vertex ID in OBJ-file" );
    haveColors = firstVert < colors.size();


    Timer timer( "prepare unique vertices" );

    std::string error;
    tbb::task_group_context ctx;
    tbb::enumerable_thread_specific<std::vector<VertexRepr>> tlOrderedPoints;
    tbb::parallel_for( tbb::blocked_range<size_t>( minFace, maxFace ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto& local = tlOrderedPoints.local();
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( faces[i].vertices.size() < 3 )
            {
                if ( ctx.cancel_group_execution() )
                    error = "Face with less than 3 vertices in OBJ-file";
                return;
            }
            for ( int v = 0; v < faces[i].vertices.size(); ++v )
            {
                VertexRepr repr;
                repr.vId = faces[i].vertices[v];
                if ( repr.vId < 0 )
                    repr.vId = int( points.size() ) - repr.vId;
                else
                    --repr.vId;
                if ( repr.vId < 0 || repr.vId >= points.size() )
                {
                    if ( ctx.cancel_group_execution() )
                        error = "Out of bounds Vertex ID in OBJ-file";
                    return;
                }
                if ( v < faces[i].textures.size() )
                {
                    repr.vtId = faces[i].textures[v];
                    if ( repr.vtId < 0 )
                        repr.vtId = int( uvCoords.size() ) - repr.vtId;
                    else
                        --repr.vtId;
                    if ( repr.vtId < 0 || repr.vtId >= uvCoords.size() )
                    {
                        if ( ctx.cancel_group_execution() )
                            error = "Out of bounds Texture Vertex ID in OBJ-file";
                        return;
                    }
                }

                local.emplace_back( repr );
            }
        }
    }, ctx );

    if ( !reportProgress( settings.callback, 0.2f ) )
        return unexpectedOperationCanceled();

    if ( !error.empty() )
        return unexpected( error );

    size_t sumOVsSize = 0;
    for ( const auto& localPoints : tlOrderedPoints )
        sumOVsSize += localPoints.size();
    std::vector<VertexRepr> orderedPoints;
    orderedPoints.reserve( sumOVsSize );
    for ( auto& localPoints : tlOrderedPoints )
        orderedPoints.insert( orderedPoints.end(), std::make_move_iterator( localPoints.begin() ), std::make_move_iterator( localPoints.end() ) );
    tlOrderedPoints = {}; // reduce peak memory
    tbb::parallel_sort( orderedPoints.begin(), orderedPoints.end() );

    orderedPoints.erase( std::unique( orderedPoints.begin(), orderedPoints.end() ), orderedPoints.end() );

    using ParallelIndicesMap = ParallelHashMap<VertexRepr, VertId, VertexReprHasher>;
    ParallelIndicesMap map;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, map.subcnt(), 1 ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t mId = range.begin(); mId < range.end(); ++mId )
        {
            for ( size_t i = 0; i < orderedPoints.size(); ++i )
            {
                auto hash = map.hash( orderedPoints[i] );
                if ( mId != map.subidx( hash ) )
                    continue;
                map.insert( { orderedPoints[i],VertId( i ) } );
            }
        }
    } );

    size_t numCoords = orderedPoints.size();
    if ( numCoords == 0 )
        return unexpected( "No vertex found in OBJ-file" );

    MinMax<int> minmaxV;
    minmaxV.min = orderedPoints.front().vId;
    minmaxV.max = orderedPoints.back().vId;
    orderedPoints = {}; // reduce peak memory

    if ( !reportProgress( settings.callback, 0.3f ) )
        return unexpectedOperationCanceled();

    MeshLoad::NamedMesh res;
    VertCoords coords( numCoords );
    res.colors.resize( haveColors ? coords.size() : 0 );
    res.uvCoords.resize( haveUVs ? coords.size() : 0 );

    // set vertices and attributes first
    Vector3d pointOffset;
    if ( settings.customXf )
    {
        auto box = computeBoundingBox( points, VertId( minmaxV.min ), VertId( minmaxV.max ) );
        // shift the reference frame to the center of bounding box for best relative precision of mesh point coordinates (they are stored as 32-bit floats),
        // with the exception when boundary box already contains the origin point (0,0,0)
        pointOffset = box.contains( Vector3d{} ) ? Vector3d{} : box.center();
        res.xf = AffineXf3f::translation( Vector3f( pointOffset ) );
    }

    ParallelFor( size_t( 0 ), map.subcnt(), [&] ( size_t id )
    {
        map.with_submap( id, [&] ( const ParallelIndicesMap::EmbeddedSet& subset )
        {
            for ( const auto& [repr, vId] : subset )
            {
                coords[vId] = Vector3f( points[VertId( repr.vId )] - pointOffset );
                if ( haveColors && repr.vId < colors.size() )
                    res.colors[vId] = colors[repr.vId];
                if ( haveUVs && repr.vtId < uvCoords.size() )
                    res.uvCoords[vId] = uvCoords[repr.vtId];
            }
        } );
    } );

    if ( !reportProgress( settings.callback, 0.4f ) )
        return unexpectedOperationCanceled();

    auto getReprVertId = [&] ( size_t fId, int ind )->VertId
    {
        VertexRepr repr;
        repr.vId = faces[fId].vertices[ind];
        if ( repr.vId < 0 )
            repr.vId = int( points.size() ) - repr.vId;
        else
            --repr.vId;
        if ( ind < faces[fId].textures.size() )
        {
            repr.vtId = faces[fId].textures[ind];
            if ( repr.vtId < 0 )
                repr.vtId = int( uvCoords.size() ) - repr.vtId;
            else
                --repr.vtId;
        }
        auto it = map.find( repr );
        assert( it != map.end() );
        return it->second;
    };


    timer.restart( "prepare model triangulation" );

    // triangulation
    struct OrderedMaterial
    {
        int mScopeId{ 0 };
        size_t fId{ 0 };
        // for ordering
        size_t orderedTriangulationStartF = 0;
        size_t orderedTriangulationOffset = 0;
    };
    std::vector<OrderedMaterial> materialFaces;
    if ( mtl )
    {
        MinMax<int> minmaxMtl;
        for ( int i = 0; i < materialScope.size(); ++i )
        {
            if ( materialScope[i].fId >= maxFace )
                break;
            if ( materialScope[i].fId <= minFace )
                minmaxMtl.max = minmaxMtl.min = i;
            else if ( materialScope[i].fId < maxFace )
                minmaxMtl.max = i;
        }
        if ( minmaxMtl.valid() )
        {
            materialFaces.resize( minmaxMtl.max - minmaxMtl.min + 1 );
            for ( int i = minmaxMtl.min; i <= minmaxMtl.max; ++i )
                materialFaces[i - minmaxMtl.min] = { i,materialScope[i].fId };
        }
    }

    struct OrderedTriangulation
    {
        size_t startF{ 0 };
        Triangulation t;
    };
    tbb::enumerable_thread_specific<std::vector<OrderedTriangulation>> tls;
    tbb::parallel_for( tbb::blocked_range<size_t>( minFace, maxFace ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto& local = tls.local();
        auto& thisOT = local.emplace_back();
        thisOT.startF = range.begin();
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( !materialFaces.empty() && i <= materialFaces.back().fId )
            {
                for ( int mi = 0; mi < materialFaces.size(); ++mi )
                {
                    if ( i == materialFaces[mi].fId )
                    {
                        // should be thread safe
                        materialFaces[mi].orderedTriangulationStartF = thisOT.startF;
                        materialFaces[mi].orderedTriangulationOffset = thisOT.t.size();
                    }
                }
            }
            for ( int j = 1; j + 1 < faces[i].vertices.size(); ++j )
                thisOT.t.push_back( { getReprVertId( i, 0 ), getReprVertId( i, j ), getReprVertId( i, j + 1 ) } );
        }
    } );

    if ( !reportProgress( settings.callback, 0.5f ) )
        return unexpectedOperationCanceled();

    size_t sumOTsSize = 0;
    for ( const auto& local : tls )
        sumOTsSize += local.size();

    std::vector<OrderedTriangulation> mergedOT;
    mergedOT.reserve( sumOTsSize );
    for ( auto& local : tls )
        mergedOT.insert( mergedOT.end(), std::make_move_iterator( local.begin() ), std::make_move_iterator( local.end() ) );

    tls = {}; // reduce peak memory

    tbb::parallel_sort( mergedOT.begin(), mergedOT.end(), [] ( const auto& l, const auto& r ) { return l.startF < r.startF; } );
    size_t sumFaceSize = 0;
    for ( const auto& ot : mergedOT )
    {
        if ( !materialFaces.empty() && ot.startF <= materialFaces.back().orderedTriangulationStartF )
        {
            for ( int mi = 0; mi < materialFaces.size(); ++mi )
            {
                if ( ot.startF == materialFaces[mi].orderedTriangulationStartF )
                    materialFaces[mi].orderedTriangulationOffset += sumFaceSize;
            }
        }
        sumFaceSize += ot.t.size();
    }
    Triangulation t;
    t.reserve( sumFaceSize );
    for ( auto& ot : mergedOT )
        t.vec_.insert( t.vec_.end(), std::make_move_iterator( ot.t.vec_.begin() ), std::make_move_iterator( ot.t.vec_.end() ) );

    mergedOT = {}; // reduce peak memory

    for ( const auto& mf : materialFaces )
    {
        auto mIt = mtl->find( materialScope[mf.mScopeId].mtName );
        if ( mIt == mtl->end() )
            break;
        if ( mIt->second.diffuseColor == Vector3f::diagonal( -1.0f ) )
            break;
        if ( !res.diffuseColor )
        {
            res.diffuseColor = Color( mIt->second.diffuseColor );
        }
        else if ( *res.diffuseColor != Color( mIt->second.diffuseColor ) )
        {
            res.diffuseColor = std::nullopt;
            break;
        }
    }

    if ( !reportProgress( settings.callback, 0.6f ) )
        return unexpectedOperationCanceled();

    timer.restart( "triangulate" );
    std::vector<MeshBuilder::VertDuplication> dups;
    res.mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( coords ), t, &dups,
        { .skippedFaceCount = settings.countSkippedFaces ? &res.skippedFaceCount : nullptr } );
    res.duplicatedVertexCount = int( dups.size() );
    if ( !dups.empty() )
    {
        if ( !res.colors.empty() )
        {
            res.colors.resize( res.mesh.points.size() );
            for ( const auto& [src, dup] : dups )
                res.colors[dup] = res.colors[src];
        }
        if ( !res.uvCoords.empty() )
        {
            res.uvCoords.resize( res.mesh.points.size() );
            for ( const auto& [src, dup] : dups )
                res.uvCoords[dup] = res.uvCoords[src];
        }
    }

    if ( mtl )
    {
        if ( !reportProgress( settings.callback, 0.9f ) )
            return unexpectedOperationCanceled();

        timer.restart( "update textures" );
        HashMap<std::string, TextureId>  texMap;
        for ( const auto& mf : materialFaces )
        {
            auto mIt = mtl->find( materialScope[mf.mScopeId].mtName );
            if ( mIt == mtl->end() )
                break;
            if ( mIt->second.diffuseTextureFile.empty() )
            {
                texMap.clear();
                break;
            }
            texMap[mIt->second.diffuseTextureFile] = TextureId();
        }
        if ( !texMap.empty() )
        {
            for ( auto& [tName, id] : texMap )
            {
                id = res.textureFiles.endId();
                res.textureFiles.push_back( dir / tName );
            }
        }
        if ( texMap.size() > 1 )
        {
            res.texturePerFace.reserve( res.mesh.topology.lastValidFace() + 1 );
            for ( int i = 0; i < materialFaces.size(); ++i )
            {
                auto mIt = mtl->find( materialScope[materialFaces[i].mScopeId].mtName );
                auto textId = texMap[mIt->second.diffuseTextureFile];
                size_t endFaceId = i + 1 < materialFaces.size() ? materialFaces[i + 1].orderedTriangulationOffset : size_t( res.mesh.topology.lastValidFace() + 1 );
                auto numFaces = endFaceId - materialFaces[i].orderedTriangulationOffset;
                res.texturePerFace.vec_.insert( res.texturePerFace.vec_.end(), numFaces, textId );
            }
        }
    }

    return res;
}

Expected<std::vector<MeshLoad::NamedMesh>> loadModelsFromObj(
    const std::filesystem::path& dir,
    bool mergeAllObjects,
    const char* data,
    const std::vector<size_t>& newlines,
    const std::vector<ElementGroup<ObjElement>>& groups,
    const MeshLoad::ObjLoadSettings& settings )
{
    MR_TIMER;
    Vector<Vector3d,VertId> points; // flat list of all points in this object scope
    std::vector<Color> colors; // flat list of all colors in this object scope
    std::vector<UVCoord> uvCoords; // flat list of all uv coords in this object scope
    std::vector<ObjFace> faces; // flat list of face elements in this list

    size_t numPoints = 0;
    size_t numUVs = 0;
    size_t numFaces = 0;
    bool colorChecked = false;
    bool hasColors = false;

    Expected<MtlLibrary> mtl; // all materials

    std::string parseError;

    Timer timer( "prepare groups" ); // calculate sizes and read material

    auto sb = subprogress( settings.callback, 0.0f, 0.2f );
    int groupId = 0;
    for ( const auto& g : groups )
    {
        switch ( g.element )
        {
        case ObjElement::Vertex:
            numPoints += ( g.end - g.begin );
            if ( !colorChecked )
            {
                colorChecked = true;
                constexpr Vector3d cInvalidColor = { -1., -1., -1. };
                Vector3d v;
                Vector3d c{ cInvalidColor };
                std::string_view line( data + newlines[g.begin], newlines[g.begin + 1] - newlines[g.begin] );
                auto res = parseObjCoordinate( line, v, &c );
                if ( !res.has_value() )
                    return unexpected( res.error() );
                hasColors = c != cInvalidColor;
            }
            break;
        case ObjElement::TextureVertex:
            numUVs += ( g.end - g.begin );
            break;
        case ObjElement::Face:
            numFaces += ( g.end - g.begin );
            break;
        case ObjElement::MaterialLibrary:
        {
            std::string_view line( data + newlines[g.begin], newlines[g.end] - newlines[g.begin] );
            // TODO: support multiple files
            std::string filename( line.substr( strlen( "mtllib" ), std::string_view::npos ) );
            boost::trim( filename );
            mtl = loadMtlLibrary( dir / filename );
            break;
        }
        default:
            break;
        }
        if ( !reportProgress( sb, ( ++groupId ) / float( groups.size() ) ) )
            return unexpectedOperationCanceled();
    }

    timer.restart( "alloc flat arrays" );

    points.reserve( numPoints );
    if ( hasColors )
        colors.reserve( numPoints );
    uvCoords.reserve( numUVs );
    faces.reserve( numFaces );

    if ( !reportProgress( settings.callback, 0.3f ) )
        return unexpectedOperationCanceled();

    timer.restart( "fill flat arrays" ); // read arrays data and map objects and materials

    std::vector<MaterialScope> mScopes;
    std::vector<ObjectScope> oScopes;

    // simply read all points and colors into vectors
    auto fillPointsAndColors = [&] ( size_t begin, size_t end )
    {
        size_t pointsSize = points.size();
        points.resize( pointsSize + end - begin );
        if ( hasColors )
            colors.resize( points.size() );
        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            Vector3d c{ noInit };
            for ( size_t li = range.begin(); li < range.end(); ++li )
            {
                auto id = li - begin + pointsSize;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjCoordinate( line, points[VertId( id )], hasColors ? &c : nullptr );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
                if ( hasColors )
                    colors[id] = Color( c );
            }
        }, ctx );
    };

    // simply read all uv coords in vector
    auto fillUVVertices = [&] ( size_t begin, size_t end )
    {
        size_t uvSize = uvCoords.size();
        uvCoords.resize( uvSize + end - begin );
        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t li = range.begin(); li < range.end(); ++li )
            {
                auto id = li - begin + uvSize;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjTextureVertex( line, uvCoords[id] );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
            }
        }, ctx );
    };

    // simply read all faces in vector
    auto fillFaces = [&] ( size_t begin, size_t end )
    {
        size_t facesSize = faces.size();
        faces.resize( facesSize + end - begin );
        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t li = range.begin(); li < range.end(); ++li )
            {
                auto id = li - begin + facesSize;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjFace( line, faces[id] );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
            }
        }, ctx );
    };

    sb = subprogress( settings.callback, 0.3f, 0.5f );
    groupId = 0;
    for ( const auto& g : groups )
    {
        switch ( g.element )
        {
        case ObjElement::Vertex:
            fillPointsAndColors( g.begin, g.end );
            break;
        case ObjElement::Face:
            fillFaces( g.begin, g.end );
            break;
        case ObjElement::Object:
        {
            std::string_view line( data + newlines[g.begin], newlines[g.end] - newlines[g.begin] );
            auto& objData = oScopes.emplace_back();
            objData.objName = line.substr( strlen( "o" ), std::string_view::npos );
            objData.fId = faces.size();
            boost::trim( objData.objName );
            break;
        }
        case ObjElement::TextureVertex:
            fillUVVertices( g.begin, g.end );
            break;
        case ObjElement::MaterialName:
        {
            std::string_view line( data + newlines[g.begin], newlines[g.end] - newlines[g.begin] );
            auto& mtlData = mScopes.emplace_back();
            mtlData.mtName = line.substr( strlen( "usemtl" ), std::string_view::npos );
            mtlData.fId = faces.size();
            boost::trim( mtlData.mtName );
            break;
        }
        default:
            break;
        }
        if ( !reportProgress( sb, ( ++groupId ) / float( groups.size() ) ) )
            return unexpectedOperationCanceled();
        if ( !parseError.empty() )
            return unexpected( parseError );
    }

    timer.restart( "triangulate models" );

    auto newSettings = settings;
    std::vector<MeshLoad::NamedMesh> res;
    if ( mergeAllObjects || oScopes.size() <= 1 )
    {
        newSettings.callback = subprogress( settings.callback, 0.5f, 1.0f );
        auto meshObj = loadSingleModelFromObj( dir, points, colors, uvCoords, faces, mScopes, 0, faces.size(), newSettings, mtl.has_value() ? &*mtl : nullptr );
        if ( !meshObj.has_value() )
            return unexpected( std::move( meshObj.error() ) );
        res.emplace_back( std::move( *meshObj ) );
        if ( oScopes.size() == 1 )
            res.back().name = std::move( oScopes.front().objName );
        return res;
    }

    assert( oScopes.size() > 1 );
    res.resize( oScopes.size() );
    for ( int i = 0; i < res.size(); ++i )
    {
        newSettings.callback = subprogress( settings.callback, 0.5f + i / float( res.size() ) * 0.5f, 0.5f + ( i + 1 ) / float( res.size() ) * 0.5f );

        size_t minFace = oScopes[i].fId;
        size_t maxFace = i + 1 < res.size() ? oScopes[i + 1].fId : faces.size();

        auto meshObj = loadSingleModelFromObj( dir, points, colors, uvCoords, faces, mScopes, minFace, maxFace, newSettings, mtl.has_value() ? &*mtl : nullptr );
        if ( !meshObj.has_value() )
            return unexpected( std::move( meshObj.error() ) );
        res[i] = std::move( *meshObj );
        res[i].name = std::move( oScopes[i].objName );
    }
    return res;
}

}

namespace MR
{

namespace MeshLoad
{

Expected<std::vector<NamedMesh>> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                const ObjLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromSceneObjFile( in, combineAllObjects, file.parent_path(), settings ), file );
}

Expected<std::vector<NamedMesh>> fromSceneObjFile( std::istream& in, bool combineAllObjects, const std::filesystem::path& dir,
                                                                const ObjLoadSettings& settings /*= {}*/ )
{
    MR_TIMER;

    auto data = readCharBuffer( in );
    if ( !data.has_value() )
        return unexpected( data.error() );

    if ( !reportProgress(settings.callback, 0.1f) )
        return unexpectedOperationCanceled();
    // TODO: redefine callback

    ObjLoadSettings newSettings = settings;
    newSettings.callback = subprogress( settings.callback, 0.1f, 1.f );

    return fromSceneObjFile( data->data(), data->size(), combineAllObjects, dir, newSettings );
}

Expected<std::vector<NamedMesh>> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects, const std::filesystem::path& dir,
                                                                const ObjLoadSettings& settings /*= {}*/ )
{
    MR_TIMER;

    Timer timer( "split by lines" );
    const auto newlines = splitByLines( data, size );

    if ( !reportProgress( settings.callback, 0.15f ) )
        return unexpectedOperationCanceled();

    timer.restart( "group element lines" );
    const auto groups = groupLines<ObjElement>( data, size, newlines );

    if ( !reportProgress( settings.callback, 0.3f ) )
        return unexpectedOperationCanceled();

    auto newSettings = settings;
    newSettings.callback = subprogress( settings.callback, 0.3f, 1.0f );

    return loadModelsFromObj( dir, combineAllObjects, data, newlines, groups, newSettings );
}

Expected<LoadedObjects> loadObjectFromObj( const std::filesystem::path& file, const ProgressCallback& cb )
{
    return fromSceneObjFile( file, false, { .customXf = true, .countSkippedFaces = true, .callback = cb } )
    .transform( [&] ( std::vector<NamedMesh>&& results )
    {
        int totalSkippedFaceCount = 0;
        int totalDuplicatedVertexCount = 0;
        int holesCount = 0;
        LoadedObjects res;
        res.objs.resize( results.size() );
        for ( int i = 0; i < res.objs.size(); ++i )
        {
            auto& result = results[i];

            std::shared_ptr<ObjectMesh> objectMesh = std::make_shared<ObjectMesh>();
            if ( result.name.empty() )
                objectMesh->setName( utf8string( file.stem() ) );
            else
                objectMesh->setName( std::move( result.name ) );
            objectMesh->select( true );
            objectMesh->setMesh( std::make_shared<Mesh>( std::move( result.mesh ) ) );
            if ( result.diffuseColor )
                objectMesh->setFrontColor( *result.diffuseColor, false );

            objectMesh->setUVCoords( std::move( result.uvCoords ) );

            int numEmptyTexture = 0;
            for ( const auto& p : result.textureFiles )
            {
                if ( p.empty() )
                    numEmptyTexture++;
            }

            if ( numEmptyTexture != 0 && numEmptyTexture != result.textureFiles.size() )
            {
                res.warnings += "object has material with and without texture\n";
            }
            else if ( numEmptyTexture == 0 && result.textureFiles.size() != 0 )
            {
                bool crashTextureLoad = false;
                for ( const auto& p : result.textureFiles )
                {
                    auto image = ImageLoad::fromAnySupportedFormat( p );
                    if ( image.has_value() )
                    {
                        MeshTexture meshTexture;
                        meshTexture.resolution = std::move( image.value().resolution );
                        meshTexture.pixels = std::move( image.value().pixels );
                        meshTexture.filter = FilterType::Linear;
                        meshTexture.wrap = WrapType::Clamp;
                        objectMesh->addTexture( std::move( meshTexture ) );
                    }
                    else
                    {
                        crashTextureLoad = true;
                        objectMesh->setTextures( {} );
                        res.warnings += image.error() + '\n';
                        break;
                    }
                }
                if ( !crashTextureLoad )
                {
                    objectMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
                    objectMesh->setTexturePerFace( std::move( result.texturePerFace ) );
                }
            }

            if ( !result.colors.empty() )
            {
                objectMesh->setVertsColorMap( std::move( result.colors ) );
                objectMesh->setColoringType( ColoringType::VertsColorMap );
            }

            objectMesh->setXf( result.xf );

            res.objs[i] = std::dynamic_pointer_cast< Object >( objectMesh );

            holesCount += int( objectMesh->numHoles() );

            totalSkippedFaceCount += result.skippedFaceCount;
            totalDuplicatedVertexCount += result.duplicatedVertexCount;
        }

        if ( totalSkippedFaceCount )
            res.warnings += fmt::format( "{} triangles were skipped as inconsistent with others.\n", totalSkippedFaceCount );
        if ( totalDuplicatedVertexCount )
            res.warnings += fmt::format( "{} vertices were duplicated to make them manifold.\n", totalDuplicatedVertexCount );
        if ( holesCount )
            res.warnings += fmt::format( "The objects contain {} holes. Please consider using Fill Holes tool.\n", holesCount );
        return res;
    } );
}

MR_ADD_OBJECT_LOADER( IOFilter( "3D model object (.obj)", "*.obj" ), loadObjectFromObj )

} //namespace MeshLoad

} //namespace MR
