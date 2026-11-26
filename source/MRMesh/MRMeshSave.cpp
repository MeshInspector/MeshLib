#include "MRMeshSave.h"
#include "MRMeshOrTriMesh.h"
#include "MRIOFormatsRegistry.h"
#include "MRMesh.h"
#include "MRTriMesh.h"
#include "MRTimer.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "MRProgressReadWrite.h"
#include "MRBitSetParallelFor.h"
#include "MRPch/MRFmt.h"
#include "MRMeshTexture.h"
#include "MRImageSave.h"
#include "MRSerializer.h"
#include "MRTelemetry.h"
#include <cstring>
#include <fstream>
#include <numeric>

namespace MR
{

namespace MeshSave
{

using namespace MR::detail;

Expected<void> toMrmesh( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toMrmesh( mesh, out, settings );
}

Expected<void> toMrmesh( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;
    mesh.topology.write( out );

    // write points
    auto numPoints = (std::uint32_t)( mesh.topology.lastValidVert() + 1 );
    out.write( (const char*)&numPoints, 4 );

    VertCoords buf;
    const auto & xfVerts = transformPoints( mesh.points, mesh.topology.getValidVerts(), settings.xf, buf );
    if ( !writeByBlocks( out, ( const char* )xfVerts.data(), numPoints * sizeof( Vector3f ), settings.progress ) )
        return unexpectedOperationCanceled();

    if ( !out )
        return unexpected( std::string( "Error saving in Mrmesh-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

template<typename MeshType>
static Expected<void> toOffImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings );

template<typename MeshType>
static Expected<void> toOffImpl( const MeshType & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    // although .off is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toOffImpl( mesh, out, settings );
}

template<typename MeshType>
static Expected<void> toOffImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;

    const auto vertRenumber = getVertRenumber( mesh, settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = lastValidVert( mesh );
    const int numPolygons = numValidFaces( mesh );

    out << "OFF\n" << numPoints << ' ' << numPolygons << " 0\n\n";
    int numSaved = 0;
    for ( VertId i{ 0 }; i <= lastVertId; ++i )
    {
        if ( settings.onlyValidPoints && !hasVert( mesh, i ) )
            continue;
        auto saveVertex = [&]( auto && p )
        {
            out << fmt::format( "{} {} {}\n", p.x, p.y, p.z );
        };
        if ( settings.xf )
            saveVertex( applyDouble( settings.xf, mesh.points[i] ) );
        else
            saveVertex( mesh.points[i] );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / numPoints * 0.5f ) )
            return unexpectedOperationCanceled();
    }
    out << '\n';

    const FaceId lastFaceId = lastValidFace( mesh );
    const float facesNum = float( lastFaceId + 1 );
    for ( FaceId f{ 0 }; f <= lastFaceId; ++f )
    {
        if ( settings.progress && !( ( (int)f + 1 ) & 0x3FF ) && !settings.progress( float( (int)f + 1 ) / facesNum * 0.5f + 0.5f ) )
            return unexpectedOperationCanceled();
        if ( !hasFace( mesh, f ) )
            continue;

        const auto [a, b, c] = getTriVerts( mesh, f );
        out << fmt::format( "3 {} {} {}\n", vertRenumber( a ), vertRenumber( b ), vertRenumber( c ) );
    }

    if ( !out )
        return unexpected( std::string( "Error saving in OFF-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toOff( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    return toOffImpl( mesh, file, settings );
}

Expected<void> toOff( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    return toOffImpl( mesh, out, settings );
}

Expected<void> toOff( const TriMesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toOffImpl( TriMeshPart( mesh, *region ), file, settings ) : toOffImpl( mesh, file, settings );
}

Expected<void> toOff( const TriMesh & mesh, std::ostream & out, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toOffImpl( TriMeshPart( mesh, *region ), out, settings ) : toOffImpl( mesh, out, settings );
}

// several materials (Texture0, Texture1, ...) are saved in .obj instead of one (Texture)
static bool hasManyTextures( const SaveSettings & settings )
{
    return settings.textures && settings.textures->size() > 1 && settings.texturePerFace && !settings.texturePerFace->empty();
}

#ifndef __EMSCRIPTEN__ // .mtl files are not written there
// writes the solid color (if any) in the current material of .mtl file
static void writeMtlSolidColor( std::ostream & ofMtl, const SaveSettings & settings )
{
    if ( !settings.solidColor )
        return;
    const auto c = Vector4f( *settings.solidColor );
    ofMtl << fmt::format( "Kd {} {} {}\n", c.x, c.y, c.z );
    if ( settings.solidColor->a != 255 )
        ofMtl << fmt::format( "d {}\n", c.w );
}
#endif

template<typename MeshType>
static Expected<void> toObjImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings, int firstVertId );

template<typename MeshType>
static Expected<void> toObjImpl( const MeshType & mesh, const std::filesystem::path & file, const SaveSettings & settings, int firstVertId )
{
    // although .obj is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

#ifndef __EMSCRIPTEN__
    // it is hard to handle several files output for browser, so for now it is under ifdef,
    // anyway later it may be reworked to save simple zip and taken out of ifdef
    const bool saveTexture = settings.uvMap && settings.texture && !settings.texture->pixels.empty();
    if ( settings.uvMap && hasManyTextures( settings ) )
    {
        if ( auto pngSaver = ImageSave::getImageSaver( "*.png" ) )
        {
            auto mtlPath = file.parent_path() / asU8String( settings.materialName + ".mtl" );
            std::ofstream ofMtl( mtlPath, std::ofstream::binary );
            for ( TextureId t{ 0 }; ofMtl && t < settings.textures->size(); ++t )
            {
                ofMtl << fmt::format( "newmtl Texture{}\n", int( t ) );
                writeMtlSolidColor( ofMtl, settings );
                const auto& texture = ( *settings.textures )[t];
                const auto textureName = fmt::format( "{}_{}.png", settings.materialName, int( t ) );
                if ( !texture.pixels.empty() && pngSaver( texture, file.parent_path() / asU8String( textureName ) ).has_value() )
                    ofMtl << fmt::format( "map_Kd {}\n", textureName );
            }
        }
    }
    else if ( saveTexture || settings.solidColor )
    {
        auto mtlPath = file.parent_path() / asU8String( settings.materialName + ".mtl" );
        std::ofstream ofMtl( mtlPath, std::ofstream::binary );
        if ( ofMtl )
        {
            ofMtl << "newmtl Texture\n";
            writeMtlSolidColor( ofMtl, settings );
            if ( saveTexture )
                if ( auto pngSaver = ImageSave::getImageSaver( "*.png" ) )
                    if ( pngSaver( *settings.texture, file.parent_path() / asU8String( settings.materialName + ".png" ) ).has_value() )
                        ofMtl << fmt::format( "map_Kd {}\n", settings.materialName + ".png" );
        }
    }
#endif
    return toObjImpl( mesh, out, settings, firstVertId );
}

template<typename MeshType>
static Expected<void> toObjImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings, int firstVertId )
{
    MR_TIMER;
    const VertId lastVertId = lastValidVert( mesh );
    out << "# MeshInspector.com\n";
    bool saveUV = settings.uvMap && !settings.uvMap->empty();
    bool saveColors = settings.colors && !settings.colors->empty();
    const bool useMtl = saveUV || settings.solidColor;

    if ( useMtl )
        out << fmt::format( "mtllib {}.mtl\n", settings.materialName );

    const auto vertRenumber = getVertRenumber( mesh, settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();

    int numSaved = 0;
    auto sb = subprogress( settings.progress, 0.0f, settings.uvMap ? 0.35f : 0.5f );
    for ( VertId i{ 0 }; i <= lastVertId; ++i )
    {
        if ( settings.onlyValidPoints && !hasVert( mesh, i ) )
            continue;

        auto saveVertex = [&]( auto && p )
        {
            if ( saveColors )
            {
                const auto c = ( Vector4f )getAt( *settings.colors, i );
                out << fmt::format( "v {} {} {} {} {} {}\n", p.x, p.y, p.z, c[0], c[1], c[2] );
            }
            else
            {
                out << fmt::format( "v {} {} {}\n", p.x, p.y, p.z );
            }
        };
        if ( settings.xf )
            saveVertex( applyDouble( settings.xf, mesh.points[i] ) );
        else
            saveVertex( mesh.points[i] );
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !sb( float( numSaved ) / numPoints ) )
            return unexpectedOperationCanceled();
    }

    if ( saveUV )
    {
        numSaved = 0;
        sb = subprogress( settings.progress, 0.35f, 0.7f );
        for ( VertId i{ 0 }; i <= lastVertId; ++i )
        {
            if ( settings.onlyValidPoints && !hasVert( mesh, i ) )
                continue;
            const auto uv = getAt( *settings.uvMap, i );
            out << fmt::format( "vt {} {}\n", uv.x, uv.y );
            ++numSaved;
            if ( settings.progress && !( numSaved & 0x3FF ) && !sb( float( numSaved ) / numPoints ) )
                return unexpectedOperationCanceled();
        }
    }
    const bool manyTextures = saveUV && hasManyTextures( settings );
    if ( useMtl && !manyTextures )
        out << "usemtl Texture\n";

    const auto writeFace = [&]( FaceId f )
    {
        const auto [a, b, c] = getTriVerts( mesh, f );
        Vector3i values( vertRenumber( a ) + firstVertId, vertRenumber( b ) + firstVertId, vertRenumber( c ) + firstVertId );
        if ( saveUV )
            out << fmt::format( "f {}/{} {}/{} {}/{}\n",
                values.x, values.x,
                values.y, values.y,
                values.z, values.z );
        else
            out << fmt::format( "f {} {} {}\n",
                values.x, values.y, values.z );
    };

    sb = subprogress( settings.progress, saveUV ? 0.7f : 0.5f, 1.0f );
    const FaceId lastFaceId = lastValidFace( mesh );
    const auto faceTexture = [&]( FaceId f )
    {
        const auto t = getAt( *settings.texturePerFace, f, TextureId( 0 ) );
        return t.valid() && t < settings.textures->endId() ? t : TextureId( 0 );
    };
    if ( !manyTextures || !settings.packPrimitives )
    {
        // the order of faces is kept, and with several textures usemtl is written whenever the texture changes
        const float facesNum = float( lastFaceId + 1 );
        TextureId curTexture;
        for ( FaceId f{ 0 }; f <= lastFaceId; ++f )
        {
            if ( settings.progress && !( ( (int)f + 1 ) & 0x3FF ) && !sb( float( (int)f + 1 ) / facesNum ) )
                return unexpectedOperationCanceled();
            if ( !hasFace( mesh, f ) )
                continue;
            if ( manyTextures )
            {
                if ( const auto t = faceTexture( f ); t != curTexture )
                {
                    curTexture = t;
                    out << fmt::format( "usemtl Texture{}\n", int( t ) );
                }
            }
            writeFace( f );
        }
    }
    else
    {
        // faces are grouped by texture to write one usemtl per material;
        // faces of texture t are sortedFaces[texStart[t]..texStart[t+1])
        Vector<int, TextureId> texStart( settings.textures->size() + 1 );
        for ( FaceId f{ 0 }; f <= lastFaceId; ++f )
            if ( hasFace( mesh, f ) )
                ++texStart[faceTexture( f ) + 1];
        std::partial_sum( texStart.begin(), texStart.end(), texStart.begin() );
        Vector<FaceId> sortedFaces( texStart.back() );
        auto texNext = texStart;
        for ( FaceId f{ 0 }; f <= lastFaceId; ++f )
            if ( hasFace( mesh, f ) )
                sortedFaces[texNext[faceTexture( f )]++] = f;

        const float facesNum = float( sortedFaces.size() );
        for ( TextureId t{ 0 }; t < settings.textures->endId(); ++t )
        {
            if ( texStart[t] == texStart[t + 1] )
                continue;
            out << fmt::format( "usemtl Texture{}\n", int( t ) );
            for ( int i = texStart[t]; i < texStart[t + 1]; ++i )
            {
                if ( settings.progress && !( ( i + 1 ) & 0x3FF ) && !sb( float( i + 1 ) / facesNum ) )
                    return unexpectedOperationCanceled();
                writeFace( sortedFaces[i] );
            }
        }
    }

    if ( !out )
        return unexpected( std::string( "Error saving in OBJ-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toObj( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, int firstVertId )
{
    return toObjImpl( mesh, file, settings, firstVertId );
}

Expected<void> toObj( const Mesh & mesh, std::ostream & out, const SaveSettings & settings, int firstVertId )
{
    return toObjImpl( mesh, out, settings, firstVertId );
}

Expected<void> toObj( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings& settings )
{
    return toObjImpl( mesh, file, settings, 1 );
}

Expected<void> toObj( const Mesh& mesh, std::ostream& out, const SaveSettings& settings )
{
    return toObjImpl( mesh, out, settings, 1 );
}

Expected<void> toObj( const TriMesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, int firstVertId, const FaceBitSet * region )
{
    return region ? toObjImpl( TriMeshPart( mesh, *region ), file, settings, firstVertId ) : toObjImpl( mesh, file, settings, firstVertId );
}

Expected<void> toObj( const TriMesh & mesh, std::ostream & out, const SaveSettings & settings, int firstVertId, const FaceBitSet * region )
{
    return region ? toObjImpl( TriMeshPart( mesh, *region ), out, settings, firstVertId ) : toObjImpl( mesh, out, settings, firstVertId );
}

Expected<void> toObj( const TriMesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toObjImpl( TriMeshPart( mesh, *region ), file, settings, 1 ) : toObjImpl( mesh, file, settings, 1 );
}

Expected<void> toObj( const TriMesh & mesh, std::ostream & out, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toObjImpl( TriMeshPart( mesh, *region ), out, settings, 1 ) : toObjImpl( mesh, out, settings, 1 );
}

template<typename MeshType>
static FaceBitSet getNotDegenTris( const MeshType & mesh )
{
    MR_TIMER;
    FaceBitSet notDegenTris = validFaces( mesh );
    BitSetParallelFor( notDegenTris, [&]( FaceId f )
    {
        const auto [a, b, c] = getTriVerts( mesh, f );
        assert( a.valid() && b.valid() && c.valid() );

        const Vector3f& ap = mesh.points[a];
        const Vector3f& bp = mesh.points[b];
        const Vector3f& cp = mesh.points[c];
        if ( ap == bp || bp == cp || cp == ap )
            notDegenTris.reset( f );
    } );
    return notDegenTris;
}

template<typename MeshType>
static Expected<void> toBinaryStlImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings );

template<typename MeshType>
static Expected<void> toBinaryStlImpl( const MeshType & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toBinaryStlImpl( mesh, out, settings );
}

template<typename MeshType>
static Expected<void> toBinaryStlImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;

    auto notDegenTris = getNotDegenTris( mesh );
    auto numTris = (std::uint32_t)notDegenTris.count();
    BinaryStlSaver saver( out, settings, numTris );

    const float trisNum = float( numTris );
    int trisIndex = 0;
    for ( auto f : notDegenTris )
    {
        saver.writeTri( getTriPoints( mesh, f ) );
        if ( settings.progress && !( trisIndex & 0x3FF ) && !settings.progress( trisIndex / trisNum ) )
            return unexpectedOperationCanceled();
        ++trisIndex;
    }

    if ( !saver.updateHeadCounter() )
        return unexpected( std::string( "Error saving in binary STL-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toBinaryStl( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    return toBinaryStlImpl( mesh, file, settings );
}

Expected<void> toBinaryStl( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    return toBinaryStlImpl( mesh, out, settings );
}

Expected<void> toBinaryStl( const TriMesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toBinaryStlImpl( TriMeshPart( mesh, *region ), file, settings ) : toBinaryStlImpl( mesh, file, settings );
}

Expected<void> toBinaryStl( const TriMesh & mesh, std::ostream & out, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toBinaryStlImpl( TriMeshPart( mesh, *region ), out, settings ) : toBinaryStlImpl( mesh, out, settings );
}

BinaryStlSaver::BinaryStlSaver( std::ostream & out, const SaveSettings & settings, std::uint32_t expectedNumTris )
    : out_( out )
    , settings_( settings )
{
    char header[80] = "MeshInspector.com";
    if ( const auto & c = settings.solidColor )
    {
        // object color by Materialise Magics convention
        const char color[] = { ' ', 'C', 'O', 'L', 'O', 'R', '=', char( c->r ), char( c->g ), char( c->b ), char( c->a ) };
        std::memcpy( header + 17, color, sizeof( color ) );
        // bit 15 means "use object color" for Magics, and "valid color" with 5-bit red, green, blue for VisCAM/SolidView,
        // so both conventions show the same color
        triAttr_ = std::uint16_t( 0x8000 | ( ( c->r >> 3 ) << 10 ) | ( ( c->g >> 3 ) << 5 ) | ( c->b >> 3 ) );
    }
    out.write( header, 80 );

    numTrisPos_ = out.tellp();
    out.write( ( const char* )&expectedNumTris, 4 );
    headNumTris_ = expectedNumTris;
}

bool BinaryStlSaver::writeTri( const Triangle3f& tri )
{
    // perform normal computation in double-precision to get exactly the same single-precision result on all platforms
    const Vector3d ad = applyDouble( settings_.xf, tri[0] );
    const Vector3d bd = applyDouble( settings_.xf, tri[1] );
    const Vector3d cd = applyDouble( settings_.xf, tri[2] );
    const Vector3f normal( cross( bd - ad, cd - ad ).normalized() );
    const Vector3f ap( ad );
    const Vector3f bp( bd );
    const Vector3f cp( cd );

    out_.write( (const char*)&normal, 12 );
    out_.write( (const char*)&ap, 12 );
    out_.write( (const char*)&bp, 12 );
    out_.write( (const char*)&cp, 12 );
    out_.write( ( const char* )&triAttr_, 2 );
    ++savedNumTris_;
    return (bool)out_;
}

bool BinaryStlSaver::updateHeadCounter()
{
    if ( savedNumTris_ != headNumTris_ )
    {
        const auto curr = out_.tellp();
        out_.seekp( numTrisPos_ );
        out_.write( ( const char* )&savedNumTris_, 4 );
        headNumTris_ = savedNumTris_;
        out_.seekp( curr );
    }
    return (bool)out_;
}

BinaryStlSaver::~BinaryStlSaver()
{
    updateHeadCounter();
}

template<typename MeshType>
static Expected<void> toAsciiStlImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings );

template<typename MeshType>
static Expected<void> toAsciiStlImpl( const MeshType & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toAsciiStlImpl( mesh, out, settings );
}

template<typename MeshType>
static Expected<void> toAsciiStlImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;

    static const char* solid_name = "MeshInspector.com";
    out << "solid " << solid_name << "\n";
    auto notDegenTris = getNotDegenTris( mesh );
    const float trisNum = float( notDegenTris.count() );
    int trisIndex = 0;
    for ( auto f : notDegenTris )
    {
        const auto [a, b, c] = getTriVerts( mesh, f );
        assert( a.valid() && b.valid() && c.valid() );
        auto saveVertex = [&]( auto && ap, auto && bp, auto && cp )
        {
            const auto normal = cross( bp - ap, cp - ap ).normalized();
            out << "" << fmt::format( "facet normal {} {} {}\n", normal.x, normal.y, normal.z );
            out << "outer loop\n";
            for ( const auto & p : { ap, bp, cp } )
                out << fmt::format( "vertex {} {} {}\n", p.x, p.y, p.z );
        };
        if ( settings.xf )
            saveVertex( applyDouble( settings.xf, mesh.points[a] ),
                        applyDouble( settings.xf, mesh.points[b] ),
                        applyDouble( settings.xf, mesh.points[c] ) );
        else
            saveVertex( mesh.points[a], mesh.points[b], mesh.points[c] );
        out << "endloop\n";
        out << "endfacet\n";
        if ( settings.progress && !( trisIndex & 0x3FF ) && !settings.progress( trisIndex / trisNum ) )
            return unexpectedOperationCanceled();
        ++trisIndex;
    }
    out << "endsolid " << solid_name << "\n";

    if ( !out )
        return unexpected( std::string( "Error saving in ascii STL-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toAsciiStl( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    return toAsciiStlImpl( mesh, file, settings );
}

Expected<void> toAsciiStl( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    return toAsciiStlImpl( mesh, out, settings );
}

Expected<void> toAsciiStl( const TriMesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toAsciiStlImpl( TriMeshPart( mesh, *region ), file, settings ) : toAsciiStlImpl( mesh, file, settings );
}

Expected<void> toAsciiStl( const TriMesh & mesh, std::ostream & out, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toAsciiStlImpl( TriMeshPart( mesh, *region ), out, settings ) : toAsciiStlImpl( mesh, out, settings );
}

template<typename MeshType>
static Expected<void> toPlyImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings );

template<typename MeshType>
static Expected<void> toPlyImpl( const MeshType & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

#ifndef __EMSCRIPTEN__
    // it is hard to handle several files output for browser, so for now it is under ifdef,
    // anyway later it may be reworked to save simple zip and taken out of ifdef
    if ( settings.texture && !settings.texture->pixels.empty() )
    {
        if ( auto texSaver = ImageSave::getImageSaver( "*.jpg" ) ) // MeshLab cannot open textures from .PNG
            (void)texSaver( *settings.texture, file.parent_path() / asU8String( settings.materialName + ".jpg" ) );
    }
#endif
    return toPlyImpl( mesh, out, settings );
}

template<typename MeshType>
static Expected<void> toPlyImpl( const MeshType & mesh, std::ostream & out, const SaveSettings & settings )
{
    MR_TIMER;

    const auto vertRenumber = getVertRenumber( mesh, settings.onlyValidPoints );
    const int numPoints = vertRenumber.sizeVerts();
    const VertId lastVertId = lastValidVert( mesh );

    bool saveTexture = settings.texture && !settings.texture->pixels.empty();
    bool saveUV = settings.uvMap && !settings.uvMap->empty();
    bool saveColors = settings.colors && !settings.colors->empty();
    bool saveFaceColors = settings.primitiveColors && !settings.primitiveColors->empty();

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshInspector.com\n";
    if ( saveUV && saveTexture )
        out << "comment TextureFile " <<  settings.materialName << ".jpg\n";
    out << "element vertex " << numPoints << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( saveColors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    if ( saveUV && !settings.saveTriCornerUVCoords )
        out << "property float texture_u\nproperty float texture_v\n";

    const auto fLast = lastValidFace( mesh );
    const auto numSaveFaces = settings.packPrimitives ? numValidFaces( mesh ) : int( fLast + 1 );
    out << "element face " << numSaveFaces << "\nproperty list uchar int vertex_indices\n";
    if ( saveFaceColors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    if ( saveUV && settings.saveTriCornerUVCoords )
        out << "property list uchar float texcoord\n";
    out << "end_header\n";

    static_assert( sizeof( UVCoord ) == 8, "wrong size of UVCoord" );
    static_assert( sizeof( Vector3f ) == 12, "wrong size of Vector3f" );
    struct PlyColor
    {
        unsigned char r = 0, g = 0, b = 0;
    };
    static_assert( sizeof( PlyColor ) == 3, "check your padding" );

    // write vertices
    int numSaved = 0;
    for ( VertId i{ 0 }; i <= lastVertId; ++i )
    {
        if ( settings.onlyValidPoints && !hasVert( mesh, i ) )
            continue;
        const Vector3f p = applyFloat( settings.xf, mesh.points[i] );
        out.write( ( const char* )&p, 12 );
        if ( saveColors )
        {
            const auto c = getAt( *settings.colors, i );
            PlyColor pc{ .r = c.r, .g = c.g, .b = c.b };
            out.write( ( const char* )&pc, 3 );
        }
        if ( saveUV && !settings.saveTriCornerUVCoords )
        {
            const auto uv = getAt( *settings.uvMap, i );
            out.write( ( const char* )&uv, sizeof( UVCoord ) );
        }
        ++numSaved;
        if ( settings.progress && !( numSaved & 0x3FF ) && !settings.progress( float( numSaved ) / numPoints * 0.5f ) )
            return unexpectedOperationCanceled();
    }

    // write triangles
    Vector3i tri;
    int savedFaces = 0;
    for ( FaceId f{0}; f <= fLast; ++f )
    {
        ThreeVertIds vs;
        if ( hasFace( mesh, f ) )
        {
            vs = getTriVerts( mesh, f );
            for ( int i = 0; i < 3; ++i )
                tri[i] = vertRenumber( vs[i] );
        }
        else if ( !settings.packPrimitives )
            tri[0] = tri[1] = tri[2] = 0;
        else
            continue;
        out.put( char( 3 ) );
        static_assert( sizeof( Vector3i ) == 12 );
        out.write( (const char *)&tri, sizeof( Vector3i ) );
        if ( saveFaceColors )
        {
            const auto c = getAt( *settings.primitiveColors, f );
            PlyColor pc{ .r = c.r, .g = c.g, .b = c.b };
            out.write( ( const char* )&pc, 3 );
        }
        if ( saveUV && settings.saveTriCornerUVCoords )
        {
            UVCoord uvs[3];
            for ( int i = 0; i < 3; ++i )
                uvs[i] = getAt( *settings.uvMap, vs[i] );
            out.put( char( 6 ) );
            static_assert( sizeof( uvs ) == 3 * 2 * 4 );
            out.write( (const char *)uvs, sizeof( uvs ) );
        }
        ++savedFaces;
        if ( settings.progress && !( savedFaces & 0x3FF ) && !settings.progress( float( savedFaces ) / numSaveFaces * 0.5f + 0.5f ) )
            return unexpectedOperationCanceled();
    }

    if ( !out )
        return unexpected( std::string( "Error saving in PLY-format" ) );

    reportProgress( settings.progress, 1.f );
    return {};
}

Expected<void> toPly( const Mesh & mesh, const std::filesystem::path & file, const SaveSettings & settings )
{
    return toPlyImpl( mesh, file, settings );
}

Expected<void> toPly( const Mesh & mesh, std::ostream & out, const SaveSettings & settings )
{
    return toPlyImpl( mesh, out, settings );
}

Expected<void> toPly( const TriMesh & mesh, const std::filesystem::path & file, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toPlyImpl( TriMeshPart( mesh, *region ), file, settings ) : toPlyImpl( mesh, file, settings );
}

Expected<void> toPly( const TriMesh & mesh, std::ostream & out, const SaveSettings & settings, const FaceBitSet * region )
{
    return region ? toPlyImpl( TriMeshPart( mesh, *region ), out, settings ) : toPlyImpl( mesh, out, settings );
}

static void telemetryLogSize( const Mesh& mesh )
{
    if ( int logFaces = intLog2( mesh.topology.numValidFaces() ) )
        TelemetrySignal( "Save Mesh Log Tris " + std::to_string( logFaces ) );
    else if ( int logPoints = intLog2( mesh.points.size() ) ) // if saved mesh contains no triangles but only points
        TelemetrySignal( "Save Mesh Log Pnts " + std::to_string( logPoints ) );
}

static void telemetrySaveMesh( const std::string& ext, const Mesh& mesh, const SaveSettings& settings )
{
    if ( !settings.telemetrySignal )
        return;

    std::string signalString = "Save " + ext;

    if ( auto lv = mesh.points.size() ) // not lv = mesh.topology.lastValidVert(), since topology can be empty
    {
        signalString += " VP";
        if ( settings.colors && settings.colors->size() >= lv )
            signalString += 'C';
        if ( settings.uvMap && settings.uvMap->size() >= lv )
            signalString += "UV";
    }

    if ( auto lf = mesh.topology.lastValidFace() )
    {
        signalString += " TRI";
        if ( settings.primitiveColors && settings.primitiveColors->size() >= lf )
            signalString += 'C';
    }

    if ( settings.texture && !settings.texture->pixels.empty() )
        signalString += " TEX";

    if ( settings.solidColor )
        signalString += " SOLIDC";

    if ( settings.lengthUnit )
        signalString += " UNITS";

    if ( settings.xf && *settings.xf != AffineXf3d{} )
        signalString += " XF";

    TelemetrySignal( signalString );
    telemetryLogSize( mesh );
}

Expected<void> toAnySupportedFormat( const Mesh& mesh, const std::filesystem::path& file, const SaveSettings & settings )
{
    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char) tolower( c );
    ext = "*" + ext;

    auto saver = getMeshSaver( ext );
    if ( !saver.fileSave )
        return unexpectedUnsupportedFileExtension();

    auto res = saver.fileSave( mesh, file, settings );
    if ( res )
        telemetrySaveMesh( ext, mesh, settings );
    return res;
}

Expected<void> toAnySupportedFormat( const Mesh& mesh, const std::string& extension, std::ostream& out, const SaveSettings & settings )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    auto saver = getMeshSaver( ext );
    if ( !saver.streamSave )
        return unexpected( std::string( "unsupported stream extension" ) );

    auto res = saver.streamSave( mesh, out, settings );
    if ( res )
        telemetrySaveMesh( ext, mesh, settings );
    return res;
}

/// One can call this function from VS interpreter window during debuging
/// ( { ,,MRMesh.dll }MR::MeshSave::debugSaveMesh_ )( ( void* )meshPtr,( void* )faceBitSet, "E:/debug.mru" )
void debugSaveMesh_( void* mesh, void* faces, const char* path )
{
    using namespace MR;
    auto pp = pathFromUtf8( path );
    if ( utf8string( pp.extension() ) == ".mru" || faces )
    {
        // isolation is needed so it does not steal actuall work during debug saving
        tbb::this_task_arena::isolate( [&] ()
        {
            ( void )serializeMesh( *( const Mesh* )mesh, pp, ( const FaceBitSet* )faces );
        } );
    }
    else
    {
        ( void )toAnySupportedFormat( *( const Mesh* )mesh, pp );
    }
}

MR_ADD_MESH_SAVER_WITH_PRIORITY( IOFilter( "MrMesh (.mrmesh)", "*.mrmesh" ), toMrmesh, {}, -1 )
MR_ADD_MESH_SAVER( IOFilter( "Binary STL (.stl)", "*.stl"   ), toBinaryStl, {} )
MR_ADD_MESH_SAVER( IOFilter( "OFF (.off)",        "*.off"   ), toOff, {} )
MR_ADD_MESH_SAVER( IOFilter( "OBJ (.obj)",        "*.obj"   ), toObj, { .storesVertexColors = true } )
MR_ADD_MESH_SAVER( IOFilter( "PLY (.ply)",        "*.ply"   ), toPly, { .storesVertexColors = true } )

} //namespace MeshSave

} //namespace MR
