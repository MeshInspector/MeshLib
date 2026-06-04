#include "MRViewportCornerController.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRVector.h"
#include "MRColorTheme.h"
#include "MRI18n.h"
#include "MRRibbonFontManager.h"
#include "MRViewer.h"
#include "MRViewerSignals.h"
#include "MRViewport.h"
#include "MRMesh/MRSystemPath.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRImageLoad.h"
#include "MRMesh/MR2DContoursTriangulation.h"
#include "MRMesh/MR2to3.h"
#include "MRMesh/MRStringConvert.h"
#include "MRViewer/MRMouseController.h"

namespace MR
{

namespace
{

constexpr float cControllerCubeFontSize = 22.f;

ImFont* loadControllerCubeFont( float fontSize )
{
    loadCustomFont( SystemPath::getFontsDirectory() / "NotoSans-Bold.ttf", fontSize );
    return loadCustomFont( SystemPath::getFontsDirectory() / "NotoSansCJK-Regular.ttc", fontSize, { .mergeMode = true } );
}

void copyTexture( int w, int h, const ImTextureData* tex, int tx0, int ty0, Image& img, int ix0, int iy0 )
{
    for ( auto y = 0; y < h; ++y )
    {
        for ( auto x = 0; x < w; ++x )
        {
            const auto* src = tex->Pixels + ( ( ty0 + y ) * tex->Width + ( tx0 + x ) ) * tex->BytesPerPixel;
            auto& dst = img.pixels[( iy0 + y ) * img.resolution.x + ( ix0 + x )];
            switch ( tex->Format )
            {
            case ImTextureFormat_RGBA32:
                for ( auto i = 0; i < tex->BytesPerPixel; ++i )
                    dst[i] = src[i];
                break;
            case ImTextureFormat_Alpha8:
                dst = { *src, *src, *src };
                break;
            }
        }
    }
}

void copyImage( int w, int h, const Image& src, int sx0, int sy0, Image& dst, int dx0, int dy0 )
{
    for ( auto y = 0; y < h; ++y )
        for ( auto x = 0; x < w; ++x )
            dst.pixels[( dy0 + y ) * dst.resolution.x + ( dx0 + x )] = src.pixels[( sy0 + y ) * src.resolution.x + ( sx0 + x )];
}

void flipVertically( Image& img )
{
    const auto w = img.resolution.x, h = img.resolution.y;
    for ( auto y = 0; y < h / 2; ++y )
        for ( auto x = 0; x < w; ++x )
            std::swap( img.pixels[y * w + x], img.pixels[( h - y - 1 ) * w + x] );
}

Expected<Image> renderControllerSideText( const Vector2i& resolution )
{
    // TODO: disconnect from ImGui
    static auto* font = loadControllerCubeFont( cControllerCubeFontSize );
    if ( !font )
        return unexpected( "Could not load font" );

    auto* baked = font->GetFontBaked( cControllerCubeFontSize );
    if ( !baked )
        return unexpected( "Could not load font" );

    const auto* tex = ImGui::GetIO().Fonts->TexData;

    constexpr std::array cSideTexts = {
        _t( "RIGHT" ),  _t( "LEFT" ),
        _t( "TOP" ),    _t( "BOTTOM" ),
        _t( "FRONT" ),  _t( "BACK" ),
    };

    Image image;
    image.resolution = resolution;
    image.pixels.resize( image.resolution.x * image.resolution.y, { 255, 255, 255, 0 } );

    for ( auto i = 0; i < cSideTexts.size(); ++i )
    {
        const auto text = s_tr( cSideTexts[i] );

        Image block;
        block.resolution = {
            image.resolution.x / 2,
            image.resolution.y / 3,
        };
        block.pixels.resize( block.resolution.x * block.resolution.y, { 255, 255, 255, 0 } );

        int minx = INT_MAX, miny = INT_MAX;
        int maxx = 0, maxy = 0;
        float penx = 0.f;
        for ( const auto ch : utf8ToUtf32( text ) )
        {
            auto* glyph = baked->FindGlyph( (ImWchar)ch );
            if ( !glyph )
                return unexpected( fmt::format( "Could not load glyph for code point {}", (size_t)ch ) );

            if ( block.resolution.x < (int)std::ceil( penx + glyph->X1 ) || block.resolution.y < (int)std::ceil( glyph->Y1 ) )
            {
                // TODO: adjust font size
                assert( false );
                break;
            }

            const auto
                w = int( glyph->X1 - glyph->X0 ),
                h = int( glyph->Y1 - glyph->Y0 );
            const auto
                tx0 = int( glyph->U0 * tex->Width ),
                ty0 = int( glyph->V0 * tex->Height );
            copyTexture( w, h, tex, tx0, ty0, block, std::floor( penx + glyph->X0 ), std::floor( glyph->Y0 ) );

            if ( penx == 0.f )
                minx = (int)std::floor( glyph->X0 );
            miny = std::min( miny, (int)std::floor( glyph->Y0 ) );
            maxx = (int)std::ceil( penx + glyph->X1 );
            maxy = std::max( maxy, (int)std::ceil( glyph->Y1 ) );
            penx += glyph->AdvanceX;
        }
        if ( maxx < minx || maxy < miny )
            continue;

        int offsetx = ( i % 2 ) * block.resolution.x;
        int offsety = ( i / 2 ) * block.resolution.y;
        offsetx += ( block.resolution.x - ( maxx - minx ) ) / 2;
        offsety += ( block.resolution.y - ( maxy - miny ) ) / 2;
        copyImage( maxx - minx, maxy - miny, block, minx, miny, image, offsetx, offsety );
    }

    flipVertically( image );
    return image;
}

} // namespace

Mesh makeCornerControllerMesh( float size, float cornerRatio /*= 0.15f */ )
{
    Mesh outMesh;
    outMesh.points.resize( 4 * 6 + 2 * 12 * 2 + 8 * 3 ); // 4x6 - verts on each side, 2x12 - verts on 2-rank corners (x2 to have disconnected edges), 8 - verts on 3-rank corners (x3 to have disconnected corners)
    Triangulation t;
    t.resize( 2 * 6 + 4 * 12 + 6 * 8 ); // 2x6 - faces on each side, 4x12 - faces on 2-rank corners, 6x8 - faces on 3-rank corners

    // 4x6 - verts on each side
    for ( int i = 0; i < 6; ++i )
    {
        int axis = i / 2;
        float sign = ( i % 2 ) * 2.0f - 1.0f;

        int ax2 = ( axis + 1 ) % 3;
        int ax3 = ( axis + 2 ) % 3;
        if ( sign > 0 )
            std::swap( ax2, ax3 );
        for ( int j = 0; j < 4; ++j )
        {
            auto& pt = outMesh.points[VertId( i * 4 + j )];
            pt[axis] = sign * 0.5f * size;
            pt[ax2] = ( ( j / 2 ) * 2.0f - 1.0f ) * ( ( 0.5f - cornerRatio ) * size );
            pt[ax3] = ( ( j % 2 ) * 2.0f - 1.0f ) * ( ( 0.5f - cornerRatio ) * size );
        }
        // add faces
        t[FaceId( i * 2 )] = { VertId( i * 4 ),VertId( i * 4 + 1 ),VertId( i * 4 + 3 ) };
        t[FaceId( i * 2 + 1 )] = { VertId( i * 4 ),VertId( i * 4 + 3 ),VertId( i * 4 + 2 ) };
    }

    int vOffset = 4 * 6;
    int fOffset = 2 * 6;
    // 2x12 - verts on 2-rank corners (x2 to have disconnected edges)
    for ( int i = 0; i < 12; ++i )
    {
        int baseSide = i / 4; // 0 - (xy,x-y,-xy,-x-y), 1 - (yz,y-z,-yz,-y-z), 2 - (zx,z-x,-zx,-z-x)
        int baseSide2 = ( baseSide + 1 ) % 3;
        int signsId = i % 4;
        int signsId1 = signsId / 2;
        int signsId2 = signsId % 2;
        float sign = signsId1 * 2.0f - 1.0f;
        float sign2 = signsId2 * 2.0f - 1.0f;
        for ( int k = 0; k < 2; ++k )
        {
            for ( int j = 0; j < 2; ++j )
            {
                auto& pt = outMesh.points[VertId( vOffset + i * 4 + k * 2 + j )];
                pt[baseSide] = sign * 0.5f * size;
                pt[baseSide2] = sign2 * 0.5f * size;
                pt[( baseSide2 + 1 ) % 3] = ( ( j % 2 ) * 2.0f - 1.0f ) * ( ( 0.5f - cornerRatio ) * size );
            }
        }
        // add faces
        for ( int j = 0; j < 2; ++j )
        {
            auto faceBaseSide = j == 0 ? baseSide : baseSide2;
            auto faceSign = j == 0 ? signsId1 : signsId2;
            auto otherSign = j == 0 ? signsId2 : signsId1;

            VertId baseVert = VertId( faceBaseSide * 8 + faceSign * 4 );
            VertId v0, v1;

            bool inversed = false;
            if ( j == 0 )
            {
                inversed = faceSign != otherSign;
                v0 = baseVert + ( ( 3 * otherSign + 2 * faceSign ) % 4 );
                v1 = baseVert + 1 + 1 * otherSign + ( inversed ? -1 : 1 ) * faceSign;
            }
            else
            {
                inversed = faceSign == otherSign;
                v0 = baseVert + ( inversed ? -1 : 1 ) * faceSign + 3 * otherSign;
                v1 = baseVert + ( ( 2 + 2 * faceSign + 3 * otherSign ) % 4 );
            }
            if ( j != 0 )
                std::swap( v0, v1 );
            VertId edgeVert0 = VertId( vOffset + i * 4 + j * 2 );
            VertId edgeVert1 = edgeVert0 + 1;
            t[FaceId( fOffset + i * 4 + j * 2 + 0 )] = { edgeVert0, v1, v0 };
            t[FaceId( fOffset + i * 4 + j * 2 + 1 )] = { edgeVert0,inversed ? v0 : edgeVert1,inversed ? edgeVert1 : v1 };
        }
    }

    int vOffset2 = 12 * 2 * 2;
    int fOffset2 = 12 * 4;
    // 8 - verts on 3-rank corners (x3 to have disconnected corners)
    for ( int i = 0; i < 8; ++i )
    {
        Vector3i sign = Vector3i();
        for ( int j = 0; j < 3; ++j )
            sign[j] = bool( i & ( 1 << j ) ) ? 1 : 0;

        auto centerV = VertId( vOffset + vOffset2 + i * 3 );
        for ( int k = 0; k < 3; ++k )
        {
            auto& pt = outMesh.points[centerV + k];
            for ( int j = 0; j < 3; ++j )
                pt[j] = ( sign[j] * 2.0f - 1.0f ) * 0.5f * size;
        }
        // add faces
        std::array<VertId, 9> ringVerts;
        for ( int j = 0; j < 9; ++j )
        {
            int mainAxis = j / 3;
            int nextAxis = ( mainAxis + 1 ) % 3;
            int otherAxis = ( mainAxis + 2 ) % 3;
            if ( j % 3 == 0 )
            {
                // lower corner vert
                ringVerts[j] = VertId( vOffset + 2 * ( otherAxis * 8 + sign[otherAxis] * 4 + sign[mainAxis] * 2 + 1 ) + sign[nextAxis] );
            }
            else if ( j % 3 == 1 )
            {
                // inner vert
                ringVerts[j] = VertId( mainAxis * 8 + sign[mainAxis] * 4 );
                ringVerts[j] += ( sign[mainAxis] == 0 ? 2 : 1 ) * sign[nextAxis] + ( sign[mainAxis] == 0 ? 1 : 2 ) * sign[otherAxis];
            }
            else
            {
                // upper corner vert
                ringVerts[j] = VertId( vOffset + 2 * ( mainAxis * 8 + sign[mainAxis] * 4 + sign[nextAxis] * 2 ) + sign[otherAxis] );
            }
        }
        for ( int j = 0; j < 6; ++j )
        {
            int ind = ( j / 2 ) * 3 + ( j % 2 );
            VertId nextV = ringVerts[ind];
            VertId next2V = ringVerts[ind + 1];
            if ( ( sign[0] + sign[1] + sign[2] ) % 2 == 0 )
                std::swap( nextV, next2V );
            t[FaceId( fOffset + fOffset2 + i * 6 + j )] = { VertId( centerV + j / 2 ), nextV, next2V };
        }
    }

    outMesh.topology = MeshBuilder::fromTriangles( t );
    return outMesh;
}

Mesh makeCornerControllerRotationArrowMesh( float size, const Vector2f& shift, bool ccw )
{
    Contours2d conts;
    auto& cont = conts.emplace_back();
    const int cAngleSteps = 10;
    cont.resize( 2 * cAngleSteps + 4 );
    double r1 = 1.2 * size;
    double r2 = 1.4 * size;
    double r3 = 0.9 * size;
    Vector2d center = Vector2d( shift ) - Vector2d( r1, 0.0 );

    double currentAngle = 0.0f;

    double angleStep = PI * 0.25 / float( cAngleSteps - 1 );    
    for ( int i = 0; i < cAngleSteps; ++i )
    {
        currentAngle = i * angleStep;
        cont[i] = center + Vector2d( r1 * std::cos( currentAngle ), r1 * std::sin( currentAngle ) );
    }
    cont[cAngleSteps] = center + Vector2d( r2 * std::cos( currentAngle ), r2 * std::sin( currentAngle ) );
    auto arrowR = ( r1 + r3 + 0.15 * size ) * 0.5;
    auto arrowAng = PI / 2.5;
    cont[cAngleSteps + 1] = center + Vector2d( arrowR * std::cos( arrowAng ), arrowR * std::sin( arrowAng ) );
    cont[cAngleSteps + 2] = center + Vector2d( ( r3 - r2 + r1 ) * std::cos( currentAngle ), ( r3 - r2 + r1 ) * std::sin( currentAngle ) );
    for ( int i = 0; i < cAngleSteps; ++i )
    {
        currentAngle = ( cAngleSteps - i - 1 ) * angleStep;
        cont[cAngleSteps + 3 + i] = center + Vector2d( r3 * std::cos( currentAngle ), r3 * std::sin( currentAngle ) );
    }
    cont.back() = cont.front();
    Mesh mesh = PlanarTriangulation::triangulateContours( conts );
    for ( auto& p : mesh.points )
        p.z = -2 * size;

    if ( ccw )
        return mesh;

    for ( auto& p : mesh.points )
        p.y = 2 * shift.y - p.y;
    mesh.topology.flipOrientation();
    return mesh;
}

VertUVCoords makeCornerControllerUVCoords( float cornerRatio /*= 0.2f */ )
{
    VertUVCoords uvs( 4 * 6 + 2 * 12 * 2 + 8 * 3 ); // 4x6 - verts on each side, 2x12 - verts on 2-rank corners (x2 to have disconnected edges), 8 - verts on 3-rank corners (x3 to have disconnected corners);
    VertId vOffset = VertId( 4 * 6 );
    VertId vOffset2 = VertId( 2 * 12 * 2 );

    constexpr std::array<UVCoord, 4 * 6> cBaseUvs
    {
        UVCoord( 1.0f,2.f / 3.f ),UVCoord( 1.0f,1.0f ),UVCoord( 0.5f,2.f / 3.f ),UVCoord( 0.5f,1.0f ),//-+ -- ++ +-
        UVCoord( 0.0f,2.f / 3.f ),UVCoord( 0.5f,2.f / 3.f ),UVCoord( 0.0f,1.0f ),UVCoord( 0.5f,1.0f ),//++ -+ +- --
        
        UVCoord( 0.0f,0.0f ),UVCoord( 0.5f,0.0f ),UVCoord( 0.0f,1.f / 3.f ),UVCoord( 0.5f,1.f / 3.f ),//++ -+ +- --
        UVCoord( 1.0f,0.0f ),UVCoord( 1.0f,1.f / 3.f ),UVCoord( 0.5f,0.0f ),UVCoord( 0.5f,1.f / 3.f ),//-+ -- ++ +-

        UVCoord( 1.0f,1.f / 3.f ),UVCoord( 1.0f,2.f / 3.f ),UVCoord( 0.5f,1.f / 3.f ),UVCoord( 0.5f,2.f / 3.f ),//-+ -- ++ +-
        UVCoord( 0.0f,1.f / 3.f ),UVCoord( 0.5f,1.f / 3.f ),UVCoord( 0.0f,2.f / 3.f ),UVCoord( 0.5f,2.f / 3.f ),//++ -+ +- --
    };

    constexpr std::array<bool, 2 * 4 * 6> cBaseSign
    {
        false,true,  false,false,  true,true,   true,false, //-+ -- ++ +-
        true,true,   false,true,   true,false,  false,false,//++ -+ +- --

        true,true,   false,true,   true,false,  false,false,//++ -+ +- --
        false,true,  false,false,  true,true,   true,false, //-+ -- ++ +-

        false,true,  false,false,  true,true,   true,false, //-+ -- ++ +-
        true,true,   false,true,   true,false,  false,false,//++ -+ +- --
    };

    // 4x6 - verts on each side
    for ( int i = 0; i < 6; ++i )
    {
        const UVCoord* cornerUVs = &cBaseUvs[i * 4];

        for ( int j = 0; j < 4; ++j )
        {
            auto& uv = uvs[VertId( i * 4 + j )];
            uv = cornerUVs[j];

            auto sign0 = ( cBaseSign[i * 8 + j * 2] ? 1.0f : -1.0f );
            auto sign1 = ( cBaseSign[i * 8 + j * 2 + 1] ? 1.0f : -1.0f );
            uv.x += sign0 * 0.5f * cornerRatio;
            uv.y += sign1 * ( 1.f / 3.f ) * cornerRatio;
        }
    }

    // 2x12 - verts on 2-rank corners (x2 to have disconnected edges)
    for ( int i = 0; i < 12; ++i )
    {
        int baseSide = i / 4; // 0 - (xy,x-y,-xy,-x-y), 1 - (yz,y-z,-yz,-y-z), 2 - (zx,z-x,-zx,-z-x)
        int baseSide2 = ( baseSide + 1 ) % 3;
        int signsId = i % 4;
        int signsId1 = signsId / 2;
        int signsId2 = signsId % 2;
        // add faces
        for ( int j = 0; j < 2; ++j )
        {
            auto faceBaseSide = j == 0 ? baseSide : baseSide2;
            auto faceSign = j == 0 ? signsId1 : signsId2;
            auto otherSign = j == 0 ? signsId2 : signsId1;

            VertId baseVert = VertId( faceBaseSide * 8 + faceSign * 4 );
            VertId v0, v1;

            bool inversed = false;
            if ( j == 0 )
            {
                inversed = faceSign != otherSign;
                v0 = baseVert + ( ( 3 * otherSign + 2 * faceSign ) % 4 );
                v1 = baseVert + 1 + 1 * otherSign + ( inversed ? -1 : 1 ) * faceSign;
            }
            else
            {
                inversed = faceSign == otherSign;
                v0 = baseVert + ( inversed ? -1 : 1 ) * faceSign + 3 * otherSign;
                v1 = baseVert + ( ( 2 + 2 * faceSign + 3 * otherSign ) % 4 );
            }
            if ( signsId1 != signsId2 )
                std::swap( v0, v1 );
            VertId edgeVert0 = VertId( vOffset + i * 4 + j * 2 );
            VertId edgeVert1 = edgeVert0 + 1;
            uvs[edgeVert0] = cBaseUvs[v0];
            uvs[edgeVert1] = cBaseUvs[v1];

            auto updateCoord = ( baseSide == 0 || ( baseSide == 2 && j == 0 ) ) ? 1 : 0;
            uvs[edgeVert0][updateCoord] += ( cBaseSign[2 * v0 + int( updateCoord )] ? 1.0f : -1.0f ) * ( updateCoord == 0 ? 0.5f : 1.f / 3.f ) * cornerRatio;
            uvs[edgeVert1][updateCoord] += ( cBaseSign[2 * v1 + int( updateCoord )] ? 1.0f : -1.0f ) * ( updateCoord == 0 ? 0.5f : 1.f / 3.f ) * cornerRatio;
        }
    }

    // 8 - verts on 3-rank corners (x3 to have disconnected corners)
    for ( int i = 0; i < 8; ++i )
    {
        Vector3i sign = Vector3i();
        auto centerV = VertId( vOffset + vOffset2 + i * 3 );
        for ( int j = 0; j < 3; ++j )
            sign[j] = bool( i & ( 1 << j ) ) ? 1 : 0;
        for ( int j = 0; j < 3; ++j )
        {
            int mainAxis = j;
            int nextAxis = ( mainAxis + 1 ) % 3;
            int otherAxis = ( mainAxis + 2 ) % 3;
            //if ( sign[mainAxis] > 0 )
            //    std::swap( nextAxis, otherAxis );

            auto innerV = VertId( mainAxis * 8 + sign[mainAxis] * 4 );
            innerV += ( sign[mainAxis] == 0 ? 2 : 1 ) * sign[nextAxis] + ( sign[mainAxis] == 0 ? 1 : 2 ) * sign[otherAxis];
            uvs[centerV + j] = cBaseUvs[innerV];
        }
    }

    return uvs;
}

Vector<MeshTexture, TextureId> loadCornerControllerTextures()
{
    const auto path = SystemPath::getResourcesDirectory() / "resource" / "textures";
    const std::array<std::filesystem::path, 3> cTexturePaths = {
        path / "controller_cube_default.png",
        path / "controller_cube_hover.png",
        path / "controller_cube_edges.png",
    };

    Vector<MeshTexture, TextureId> res;
    res.reserve( cTexturePaths.size() );
    for ( const auto& texPath : cTexturePaths )
    {
        auto loaded = ImageLoad::fromAnySupportedFormat( texPath );
        if ( !loaded.has_value() )
            return {};

        MeshTexture tex;
        tex.pixels = std::move( loaded->pixels );
        tex.resolution = loaded->resolution;
        tex.filter = FilterType::Linear;
        res.emplace_back( std::move( tex ) );
    }

    if ( const auto textImage = renderControllerSideText( res.front().resolution ) )
    {
        const std::array<Color, 3> textColors = {
            Color { 133, 139, 147 },
            Color { 133, 139, 147 },
            Color { 255, 255, 255 },
        };
        for ( auto i = 0; i < 3; ++i )
        {
            auto& tex = res.vec_[i];
            const auto& col = textColors[i];
            assert( tex.pixels.size() == textImage->pixels.size() );
            for ( auto px = 0; px < tex.pixels.size(); ++px )
                tex.pixels[px] = blend( { col.r, col.g, col.b, textImage->pixels[px].a }, tex.pixels[px] );
        }
    }

    return res;
}

const TexturePerFace& getCornerControllerTextureMap()
{
    static TexturePerFace textures;
    if ( textures.empty() )
    {
        const int f2RankOffset = 2 * 6;
        const int f2RankSize = 12 * 4;
        const int f3RankOffset = f2RankOffset + f2RankSize;

        textures.resize( f3RankOffset + 8 * 6, TextureId( 0 ) );
    }
    return textures;
}

RegionId getCornerControllerRegionByFace( FaceId face )
{
    static Face2RegionMap map;
    if ( map.empty() )
    {
        RegionId currentRegion;
        map.resize( 2 * 6 + 4 * 12 + 6 * 8 );
        for ( int i = 0; i < 2 * 6; ++i )
        {
            if ( i % 2 == 0 )
                ++currentRegion;
            map[FaceId( i )] = currentRegion;
        }
        auto f2RankOffset = 2 * 6;
        for ( int i = 0; i < 4 * 12; ++i )
        {
            if ( i % 4 == 0 )
                ++currentRegion;
            map[FaceId( f2RankOffset + i )] = currentRegion;
        }
        auto f3RankOffset = f2RankOffset + 4 * 12;
        for ( int i = 0; i < 6 * 8; ++i )
        {
            if ( i % 6 == 0 )
                ++currentRegion;
            map[FaceId( f3RankOffset + i )] = currentRegion;
        }
    }    
    return map[face];
}

TexturePerFace getCornerControllerHoveredTextureMap( RegionId rId )
{
    auto textures = getCornerControllerTextureMap();
    const int fOffset = 2 * 6;
    const int fOffset2 = 4 * 12;
    if ( rId < 6 )
    {
        FaceId shift = FaceId( rId * 2 );
        for ( FaceId f( shift ); f < shift + 2; ++f )
            textures[f] = TextureId( 1 );

        // hover edges
        int baseInd = rId / 2;
        int otherInd = ( baseInd + 2 ) % 3;
        textures[FaceId( fOffset + baseInd * 16 + ( rId % 2 ) * 8 + 0 )] = TextureId( 1 );
        textures[FaceId( fOffset + baseInd * 16 + ( rId % 2 ) * 8 + 1 )] = TextureId( 1 );
        textures[FaceId( fOffset + baseInd * 16 + ( rId % 2 ) * 8 + 4 )] = TextureId( 1 );
        textures[FaceId( fOffset + baseInd * 16 + ( rId % 2 ) * 8 + 5 )] = TextureId( 1 );
        textures[FaceId( fOffset + otherInd * 16 + 0 + ( rId % 2 ) * 4 + 2 )] = TextureId( 1 );
        textures[FaceId( fOffset + otherInd * 16 + 0 + ( rId % 2 ) * 4 + 3 )] = TextureId( 1 );
        textures[FaceId( fOffset + otherInd * 16 + 8 + ( rId % 2 ) * 4 + 2 )] = TextureId( 1 );
        textures[FaceId( fOffset + otherInd * 16 + 8 + ( rId % 2 ) * 4 + 3 )] = TextureId( 1 );

        // hover corners
        for ( int i = 0; i < 8; ++i )
        {
            Vector3i sign = Vector3i();
            for ( int j = 0; j < 3; ++j )
                sign[j] = bool( i & ( 1 << j ) ) ? 1 : 0;

            if ( sign[baseInd] != ( rId % 2 ) )
                continue;

            textures[FaceId( fOffset + fOffset2 + i * 6 + baseInd * 2 + 0 )] = TextureId( 1 );
            textures[FaceId( fOffset + fOffset2 + i * 6 + baseInd * 2 + 1 )] = TextureId( 1 );
        }
    }
    else if ( rId < 6 + 12 )
    {
        FaceId shift = FaceId( fOffset + ( rId - 6 ) * 4 );
        for ( FaceId f( shift ); f < shift + 4; ++f )
            textures[f] = TextureId( 2 );
    }
    else
    {
        FaceId shift = FaceId( fOffset + fOffset2 + ( rId - 6 - 12 ) * 6 );
        for ( FaceId f( shift ); f < shift + 6; ++f )
            textures[f] = TextureId( 2 );
    }
    return textures;
}

void updateCurrentViewByControllerRegion( CornerControllerObject::PickedIds pickedId )
{
    if ( !pickedId.rId || !pickedId.vId )
        return;
    Viewport& vp = getViewerInstance().viewport( pickedId.vId );
    auto rId = pickedId.rId;
    switch ( int( rId ) )
    {
    // sides
    case 0: //from  -x
        vp.cameraLookAlong( { 1,0,0 }, { 0,0,1 } );
        break;
    case 1: //from  x
        vp.cameraLookAlong( { -1,0,0 }, { 0,0,1 } );
        break;
    case 2: //from  -y
        vp.cameraLookAlong( { 0,1,0 }, { 0,0,1 } );
        break;
    case 3: //from  y
        vp.cameraLookAlong( { 0,-1,0 }, { 0,0,1 } );
        break;
    case 4: //from  -z
        vp.cameraLookAlong( { 0,0,1 }, { 0,1,0 } );
        break;
    case 5: //from  z
        vp.cameraLookAlong( { 0,0,-1 }, { 0,1,0 } );
        break;

    // 2 rank corners
    case 6 + 0: //from  -x -y
        vp.cameraLookAlong( { 1,1,0 }, { 0,0,1 } );
        break;
    case 6 + 1: //from  -x y
        vp.cameraLookAlong( { 1,-1,0 }, { 0,0,1 } );
        break;
    case 6 + 2: //from  x -y
        vp.cameraLookAlong( { -1,1,0 }, { 0,0,1 } );
        break;
    case 6 + 3: //from  x y
        vp.cameraLookAlong( { -1,-1,0 }, { 0,0,1 } );
        break;
    case 6 + 4: //from  -y -z
        vp.cameraLookAlong( { 0,1,1 }, { 0,-1,1 } );
        break;
    case 6 + 5: //from  -y z
        vp.cameraLookAlong( { 0,1,-1 }, { 0,1,1 } );
        break;
    case 6 + 6: //from  y -z
        vp.cameraLookAlong( { 0,-1,1 }, { 0,1,1 } );
        break;
    case 6 + 7: //from  y z
        vp.cameraLookAlong( { 0,-1,-1 }, { 0,-1,1 } );
        break;
    case 6 + 8: //from  -z -x
        vp.cameraLookAlong( { 1,0,1 }, { -1,0,1 } );
        break;
    case 6 + 9: //from  -z x
        vp.cameraLookAlong( { -1,0,1 }, { 1,0,1 } );
        break;
    case 6 + 10: //from  z -x
        vp.cameraLookAlong( { 1,0,-1 }, { 1,0,1 } );
        break;
    case 6 + 11: //from  z x
        vp.cameraLookAlong( { -1,0,-1 }, { -1,0,1 } );
        break;

    // 3 rank corners
    case 6 + 12 + 0: //from -x -y -z
        vp.cameraLookAlong( { 1,1,1 }, { -1,-1,2 } );
        break;
    case 6 + 12 + 1: //from x -y -z
        vp.cameraLookAlong( { -1,1,1 }, { 1,-1,2 } );
        break;
    case 6 + 12 + 2: //from -x y -z
        vp.cameraLookAlong( { 1,-1,1 }, { -1,1,2 } );
        break;
    case 6 + 12 + 3: //from x y -z
        vp.cameraLookAlong( { -1,-1,1 }, { 1,1,2 } );
        break;
    case 6 + 12 + 4: //from -x -y z
        vp.cameraLookAlong( { 1,1,-1 }, { 1,1,2 } );
        break;
    case 6 + 12 + 5: //from x -y z
        vp.cameraLookAlong( { -1,1,-1 }, { -1,1,2 } );
        break;
    case 6 + 12 + 6: //from -x y z
        vp.cameraLookAlong( { 1,-1,-1 }, { 1,-1,2 } );
        break;
    case 6 + 12 + 7: //from x y z
        vp.cameraLookAlong( { -1,-1,-1 }, { -1,-1,2 } );
        break;

    // side axes rotation
    case int( SideRegions::CCWArrow ): // rotation
        vp.transformView( AffineXf3f::xfAround( Matrix3f::rotation( vp.getBackwardDirection(), PI2_F ), vp.getCameraPoint() ) );
        break;
    case int( SideRegions::CWArrow ) : // rotation other way
        vp.transformView( AffineXf3f::xfAround( Matrix3f::rotation( vp.getBackwardDirection(), -PI2_F ), vp.getCameraPoint() ) );
        break;
    default:
        return;
    }

    vp.preciseFitDataToScreenBorder( { 0.9f } );
}

void CornerControllerObject::initDefault()
{
    std::shared_ptr<Mesh> arrowMeshCCW = std::make_shared<Mesh>( makeCornerControllerRotationArrowMesh( 0.4f, Vector2f( 1.1f, 0.1f ), true ) );
    std::shared_ptr<Mesh> arrowMeshCW = std::make_shared<Mesh>( makeCornerControllerRotationArrowMesh( 0.4f, Vector2f( 1.1f, 0.0f ), false ) );
    std::shared_ptr<Mesh> basisControllerMesh = std::make_shared<Mesh>( makeCornerControllerMesh( 0.8f ) );

    auto basisViewControllerHoverable = std::make_shared<ObjectMesh>();
    auto basisViewControllerNonHoverable = std::make_shared<ObjectMesh>();
    auto arrowCCW = std::make_shared<ObjectMesh>();
    auto arrowCW = std::make_shared<ObjectMesh>();

    auto textures = loadCornerControllerTextures();

    auto setupCube = [&] ( std::shared_ptr<ObjectMesh> obj, bool hoverable )
    {
        obj->setMesh( basisControllerMesh );
        obj->setName( hoverable ? "CVC Hoverable" : "CVC Non-Hoverable" );
        obj->setFlatShading( true );
        obj->setVisualizeProperty( true, MeshVisualizePropertyType::BordersHighlight, ViewportMask::all() );
        obj->setVisualizeProperty( true, MeshVisualizePropertyType::PolygonOffsetFromCamera, ViewportMask::all() );
        obj->setVisualizeProperty( false, MeshVisualizePropertyType::EnableShading, ViewportMask::all() );

        obj->setUVCoords( makeCornerControllerUVCoords() );
        obj->setEdgeWidth( 0.2f );

        if ( hoverable )
        {
            obj->setTextures( textures );
        }
        else
        {
            if ( !textures.empty() )
                obj->setTextures( { textures.front() } );
        }

        if ( !obj->getTextures().empty() )
        {
            if ( hoverable )
                obj->setTexturePerFace( getCornerControllerTextureMap() );
            obj->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
        }
    };

    auto setupArrow = [&] ( std::shared_ptr<ObjectMesh> obj, bool cw )
    {
        obj->setMesh( cw ? arrowMeshCW : arrowMeshCCW );
        obj->setName( cw ? "CW" : "CCW" );
        obj->setFlatShading( true );
        obj->setVisualizeProperty( true, MeshVisualizePropertyType::BordersHighlight, ViewportMask::all() );
        obj->setVisualizeProperty( true, MeshVisualizePropertyType::PolygonOffsetFromCamera, ViewportMask::all() );
        obj->setVisualizeProperty( false, MeshVisualizePropertyType::EnableShading, ViewportMask::all() );
        obj->setEdgeWidth( 0.3f );
    };

    setupCube( basisViewControllerHoverable, true );
    setupCube( basisViewControllerNonHoverable, false );
    basisViewControllerHoverable->setVisibilityMask( ViewportMask() );
    setupArrow( arrowCW, true );
    setupArrow( arrowCCW, false );

    connections_.push_back( ColorTheme::instance().onChanged( [this] ()
    {
        if ( !rootObj_ )
            return;
        const Color& colorBg = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Background );
        const Color& colorBorder = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnDisableStart );
        for ( auto child : rootObj_->children() )
        {
            if ( auto visObj = child->asType<ObjectMesh>() )
            {
                visObj->setFrontColor( colorBg, true );
                visObj->setFrontColor( colorBg, false );
                visObj->setBordersColor( colorBorder );
            }
        }
    } ) );

    connections_.push_back( getViewerInstance().signals().preDrawSignal.connect( [this] ()
    {
        if ( !rootObj_ )
            return;
        if ( !getViewerInstance().mouseController().isPressedAny() )
            hover_( Vector2f( getViewerInstance().mouseController().getMousePos() ) );
    } ) );

    // 5th group: we want cornerControllerMouseDown_ signal be caught before tools but after menu
    connections_.push_back( getViewerInstance().signals().mouseDownSignal.connect( 5, [this] ( MouseButton btn, int mod )->bool
    {
        if ( !rootObj_ )
            return false;
        if ( btn == MouseButton::Left && mod == 0 && press_( Vector2f( getViewerInstance().mouseController().getMousePos() ) ) )
            return true;
        return false;
    } ) );

    rootObj_ = std::make_shared<Object>();
    rootObj_->addChild( basisViewControllerNonHoverable );
    rootObj_->addChild( arrowCCW );
    rootObj_->addChild( arrowCW );
    rootObj_->addChild( basisViewControllerHoverable );
}

void CornerControllerObject::enable( ViewportMask mask )
{
    if ( !rootObj_ )
        return;
    rootObj_->setVisibilityMask( mask );
}

void CornerControllerObject::draw( const Viewport& vp, const AffineXf3f& rotXf, const AffineXf3f& vpInvXf )
{
    if ( !rootObj_ || !rootObj_->isVisible( vp.id ) )
        return;
    const auto& childern = rootObj_->children();
    auto arrowsXf = rotXf * vpInvXf;
    for ( int i = 0; i < childern.size(); ++i )
    {
        const auto& xf = i == 0 || i == 3 ? rotXf : arrowsXf;
        childern[i]->setXf( xf, vp.id );
        if ( !childern[i]->isVisible( vp.id ) )
            continue;
        if ( auto visObj = childern[i]->asType<VisualObject>() )
            vp.drawOrthoFixedPos( *visObj, xf, DepthFunction::Always );
    }
    // second pass
    for ( const auto& child : childern )
    {
        if ( !child->isVisible( vp.id ) )
            continue;
        if ( auto visObj = child->asType<VisualObject>() )
            vp.drawOrthoFixedPos( *visObj, visObj->xf( vp.id ) );
    }
}

bool CornerControllerObject::getRedrawFlag( ViewportMask mask ) const
{
    if ( !rootObj_ )
        return false;
    return rootObj_->getRedrawFlag( mask );
}

void CornerControllerObject::resetRedrawFlag()
{
    if ( !rootObj_ )
        return;
    rootObj_->resetRedrawFlag();
}

CornerControllerObject::PickedIds CornerControllerObject::pick_( const Vector2f& mousePos ) const
{
    if ( !rootObj_ )
        return {};

    auto hId = getViewerInstance().getHoveredViewportId();
    if ( !hId )
        return {};

    if ( !rootObj_->isVisible( hId ) )
        return {};

    const auto& vp = getViewerInstance().viewport( hId );
    auto screenPos = to2dim( getViewerInstance().viewportToScreen( to3dim( vp.getAxesPosition() ), hId ) );

    if ( distanceSq( mousePos, screenPos ) > sqr( vp.getAxesSize() * 2.0f ) )
        return {};

    const auto& children = rootObj_->children();
    auto staticRenderParams = vp.getBaseRenderParamsOrthoFixedPos();
    auto [obj, pick] = vp.pickRenderObject( { {
            static_cast< VisualObject* >( children[0].get() ),
            static_cast< VisualObject* >( children[1].get() ),
            static_cast< VisualObject* >( children[2].get() )
        } }, { .baseRenderParams = &staticRenderParams } );
    if ( !obj )
        return {};

    if ( obj == children[0] )
        return { hId, getCornerControllerRegionByFace( pick.face ) };
    else if ( obj == children[1] )
        return { hId, RegionId( int( SideRegions::CCWArrow ) ) };
    else
        return { hId, RegionId( int( SideRegions::CWArrow ) ) };
}

void CornerControllerObject::hover_( const Vector2f& mousePos )
{
    auto curPick = pick_( mousePos );
    if ( curPick == pickedId_ )
        return;

    getViewerInstance().setSceneDirty();

    if ( pickedId_.rId )
    {
        if ( pickedId_.rId >= RegionId( int( SideRegions::CCWArrow ) ) )
        {
            const Color& colorBg = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Background );
            const Color& colorBorder = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnDisableStart );
            auto obj = rootObj_->children()[pickedId_.rId - int( SideRegions::CCWArrow ) + 1];
            if ( auto objMesh = obj->asType<ObjectMesh>() )
            {
                objMesh->setFrontColor( colorBg, true, pickedId_.vId );
                objMesh->setFrontColor( colorBg, false, pickedId_.vId );
                objMesh->setBordersColor( colorBorder, pickedId_.vId );
            }
        }
        else
        {
            rootObj_->children()[0]->setVisible( true, pickedId_.vId ); // enable non-hovarable
            rootObj_->children()[3]->setVisible( false, pickedId_.vId ); // disable hovarable
        }
    }
    // now all unhovered
    pickedId_ = curPick;
    if ( !pickedId_.rId )
        return;

    if ( pickedId_.rId >= RegionId( int( SideRegions::CCWArrow ) ) )
    {
        const Color& colorBg = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnStart );
        const Color& colorBorder = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActive );
        auto obj = rootObj_->children()[pickedId_.rId - int( SideRegions::CCWArrow ) + 1];
        if ( auto objMesh = obj->asType<ObjectMesh>() )
        {
            objMesh->setFrontColor( colorBg, true, pickedId_.vId );
            objMesh->setFrontColor( colorBg, false, pickedId_.vId );
            objMesh->setBordersColor( colorBorder, pickedId_.vId );
        }
    }
    else
    {
        rootObj_->children()[0]->setVisible( false, pickedId_.vId ); // disable non-hovarable
        auto obj = rootObj_->children()[3];
        obj->setVisible( true, pickedId_.vId ); // enable hovarable
        if ( auto objMesh = obj->asType<ObjectMesh>() )
            objMesh->setTexturePerFace( getCornerControllerHoveredTextureMap( pickedId_.rId ) );
    }
}

bool CornerControllerObject::press_( const Vector2f& mousePos )
{
    if ( !pickedId_.rId )
        return false;

    auto curPick = pick_( mousePos );
    if ( !curPick.rId )
        return false;

    updateCurrentViewByControllerRegion( curPick );
    return true;
}

}
