#include "MRSymbolMesh.h"
#ifndef MRMESH_NO_LABEL
#include "MRMesh.h"
#include "MRVector2.h"
#include "MRBox.h"
#include "MRMeshBuilder.h"
#include "MRMeshFillHole.h"
#include "MRStringConvert.h"
#include "MR2DContoursTriangulation.h"
#include "MRGTest.h"
#include "MRMeshSave.h"
#include "MRPolyline.h"
#include "MRDistanceMap.h"
#include "MRTimer.h"
#include "MRBuffer.h"
#include "MRPch/MRSpdlog.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

namespace MR
{

struct OutlineDecomposer
{
    OutlineDecomposer( unsigned bezierSteps ) :
        bezierSteps( bezierSteps )
    {
    }
    void decompose( FT_Outline* outline, Vector2d offset = {} );
    void clearLast();

    unsigned bezierSteps;
    Contours2d contours;
    Vector2d offset;
};

int MoveToCb( const FT_Vector* to, void* user )
{
    auto decomposer = static_cast<OutlineDecomposer*>( user );
    decomposer->contours.push_back( { Vector2d( ( double )to->x,( double )to->y ) + decomposer->offset } );
    return 0;
}

int LineToCb( const FT_Vector* to, void* user )
{
    auto decomposer = static_cast<OutlineDecomposer*>( user );
    decomposer->contours.back().push_back( Vector2d( ( double )to->x, ( double )to->y ) + decomposer->offset );
    return 0;
}

int ConicToCb( const FT_Vector* control, const FT_Vector* to, void* user )
{
    auto decomposer = static_cast<OutlineDecomposer*>( user );
    const Vector2d from = decomposer->contours.back().back();
    Vector2d control2d = { ( double )control->x + decomposer->offset.x ,( double )control->y + decomposer->offset.y };
    Vector2d to2d = { ( double )to->x + decomposer->offset.x,( double )to->y + decomposer->offset.y };
    auto& contour = decomposer->contours.back();
    for ( unsigned i = 0; i < decomposer->bezierSteps; ++i )
    {
        double t = double( i + 1 ) / double( decomposer->bezierSteps );
        contour.push_back( ( 1 - t )*( ( 1 - t )*from + t * control2d ) + t * ( ( 1 - t )*control2d + t * to2d ) );
    }
    return 0;
}

int CubicToCb( const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user )
{
    auto decomposer = static_cast<OutlineDecomposer*>( user );
    const Vector2d from = decomposer->contours.back().back();
    Vector2d control12d = { ( double )control1->x + decomposer->offset.x,( double )control1->y + decomposer->offset.y };
    Vector2d control22d = { ( double )control2->x + decomposer->offset.x,( double )control2->y + decomposer->offset.y };
    Vector2d to2d = { ( double )to->x + decomposer->offset.x,( double )to->y + decomposer->offset.y };
    auto& contour = decomposer->contours.back();
    for ( unsigned i = 0; i < decomposer->bezierSteps; ++i )
    {
        double t = double( i + 1 ) / double( decomposer->bezierSteps );
        const Vector2d p0 = ( 1 - t ) * ( ( 1 - t ) * from + t * control12d ) + t * ( ( 1 - t ) * control12d + t * control22d );
        const Vector2d p1 = ( 1 - t ) * ( ( 1 - t ) * control12d + t * control22d ) + t * ( ( 1 - t ) * control22d + t * to2d );
        contour.push_back( ( 1 - t ) * p0 + t * p1 );
    }
    return 0;
}

void OutlineDecomposer::decompose( FT_Outline* outline, Vector2d offsetP /* = {} */)
{
    FT_Outline_Funcs funcs;
    funcs.move_to = MoveToCb;
    funcs.line_to = LineToCb;
    funcs.conic_to = ConicToCb;
    funcs.cubic_to = CubicToCb;
    funcs.shift = 0;
    funcs.delta = 0;
    offset = offsetP;
    FT_Outline_Decompose( outline, &funcs, this );
}

void OutlineDecomposer::clearLast()
{
    for ( auto& contour : contours )
    {
        contour.erase( contour.end() - 1 );
    }
}

Contours2d createSymbolContours( const SymbolMeshParams& params )
{
    MR_TIMER
    // Begin
    FT_Library library;
    FT_Face face;
    FT_Init_FreeType( &library );
#ifdef _WIN32
    // on Windows, FT_New_Face cannot open files with Unicode names
    std::error_code ec;
    const auto fileSize = std::filesystem::file_size( params.pathToFontFile, ec );
    Buffer<char> buffer( fileSize );
    std::ifstream in( params.pathToFontFile, std::ifstream::binary );
    in.read( buffer.data(), buffer.size() );
    assert( in );
    FT_New_Memory_Face( library, (const FT_Byte *)buffer.data(), (FT_Long)buffer.size(), 0, &face );
#else
    FT_New_Face( library, utf8string( params.pathToFontFile ).c_str(), 0, &face );
#endif

    FT_Set_Char_Size( face, 128 << 6, 128 << 6, 72, 72 );
    OutlineDecomposer decomposer( params.fontDetalization );

    std::wstring wideStr = utf8ToWide( params.text.c_str() );

    // Find space width
    const std::wstring spaceSymbol = L" ";
    FT_UInt index = FT_Get_Char_Index( face, spaceSymbol[0] );
    [[maybe_unused]] auto loadError = FT_Load_Glyph( face, index, FT_LOAD_NO_BITMAP );
    assert( !loadError );
    auto addOffsetX = FT_Pos( params.symbolsDistanceAdditionalOffset.x * float( face->glyph->advance.x ) );
    auto offsetY = FT_Pos( ( 128 << 6 ) * ( 1.0f + params.symbolsDistanceAdditionalOffset.y ) );


    // <the last contour index (before '\n') to xOffset of a line>
    std::vector<std::pair<size_t, double>> contourId2width;
    double maxLineWidth = -1;
    size_t contoursPrevSize = 0;
    int currentLineLength = 0;
    Vector2i numSymbols{ 0,1 };
    auto updateContourSizeAndWidth = [&]() {
        bool isInitialized = false;
        double minX = 0.0f;
        double maxX = 0.0f;
        for (size_t j = contoursPrevSize; j < decomposer.contours.size(); ++j)
        {
            for ( const auto& p : decomposer.contours[j] )
            {
                if (!isInitialized)
                {
                    minX = p.x;
                    maxX = p.x;
                    isInitialized = true;
                }
                minX = std::min(minX, p.x);
                maxX = std::max(maxX, p.x);
            }
        }
        contourId2width.emplace_back( decomposer.contours.size() - 1, maxX - minX );
        maxLineWidth = std::max( maxLineWidth, maxX - minX );
        contoursPrevSize = decomposer.contours.size();
    };

    // Body
    FT_Pos xOffset{ 0 };
    FT_Pos yOffset{ 0 };
    FT_UInt previous = 0;
    FT_Bool kerning = FT_HAS_KERNING( face );
    for ( int i = 0; i < wideStr.length(); ++i )
    {
        if ( wideStr[i] == '\n' )
        {
            updateContourSizeAndWidth();
            xOffset = 0;
            yOffset -= offsetY;
            numSymbols.x = std::max( numSymbols.x, currentLineLength );
            currentLineLength = 0;
            ++numSymbols.y;
            continue;
        }

        index = FT_Get_Char_Index( face, wideStr[i] );
        if ( kerning && previous && index )
        {
            FT_Vector delta;
            FT_Get_Kerning( face, previous, index, FT_KERNING_DEFAULT, &delta );
            xOffset += delta.x;
        }
        if ( FT_Load_Glyph( face, index, FT_LOAD_NO_BITMAP ) )
            continue;

        // decompose
        // y offset is needed to resolve degenerate intersections of some fonts (YN sequence of Times New Roman for example)
        decomposer.decompose( &face->glyph->outline,
                              { double( xOffset ), ( i % 2 == 0 ) ? yOffset + 0.0 : yOffset + 0.5 } );

        ++currentLineLength;
        xOffset += ( face->glyph->advance.x + addOffsetX );
        previous = index;
    }
    numSymbols.x = std::max( numSymbols.x, currentLineLength );
    updateContourSizeAndWidth();
    decomposer.clearLast();

    if ( params.align != AlignType::Left )
    {
        auto lineContourIt = contourId2width.begin();
        auto currentLineWidth = lineContourIt->second;
        auto shift = params.align == AlignType::Right ? maxLineWidth - currentLineWidth : (maxLineWidth - currentLineWidth) / 2;
        for (size_t i = 0; i < decomposer.contours.size(); ++i)
        {
            for ( auto& p : decomposer.contours[i] )
                    p.x += shift;

            if (i == lineContourIt->first && next(lineContourIt) != contourId2width.end() )
            {
                ++lineContourIt;
                currentLineWidth = lineContourIt->second;
                shift = params.align == AlignType::Right ? maxLineWidth - currentLineWidth : (maxLineWidth - currentLineWidth) / 2;
            }
        }
    }

    // End
    if ( face )
    {
        FT_Done_Face( face );
    }
    FT_Done_FreeType( library );

    Box2d box;
    for ( auto& c : decomposer.contours )
    {
        for ( auto& p : c )
        {
            p *= 1e-3;
            box.include( p );
        }
        if ( c.size() > 2 )
            c.push_back( c.front() );
    }

    if ( params.symbolsThicknessOffsetModifier != 0.0f )
    {
        auto size = box.size();
        auto absOffset = size.y / double( numSymbols.y ) * params.symbolsThicknessOffsetModifier;
        Polyline2 polyline;
        polyline.topology.buildFromContours( decomposer.contours,
            [&points = polyline.points]( size_t sz )
        {
            points.reserve( sz );
        },
            [&points = polyline.points]( const Vector2d& p )
        {
            points.emplace_back( float( p.x ), float( p.y ) );
            return points.backId();
        }
        );
        ContourToDistanceMapParams dmParams;
        dmParams.pixelSize = Vector2f::diagonal( float( size.y ) / float( numSymbols.y ) / 72.0f );
        dmParams.orgPoint = Vector2f( box.min );
        dmParams.resolution.x = int( std::ceil( size.x / dmParams.pixelSize.x ) ) + 1;
        dmParams.resolution.y = int( std::ceil( size.y / dmParams.pixelSize.y ) ) + 1;
        dmParams.withSign = true;
        if ( absOffset > 0.0f )
        {
            int numPixelsOffset = int( std::ceil( absOffset / dmParams.pixelSize.x ) + 2 );
            dmParams.orgPoint -= Vector2f::diagonal( numPixelsOffset * dmParams.pixelSize.x );
            dmParams.resolution += Vector2i::diagonal( numPixelsOffset * 2 );
        }

        auto distanceMap = distanceMapFromContours( polyline, dmParams );
        auto offsettedContours = distanceMapTo2DIsoPolyline( distanceMap, dmParams, float( absOffset ) ).contours();
        decomposer.contours.resize( offsettedContours.size() );
        for ( int c = 0; c < offsettedContours.size(); ++c )
        {
            auto& doubleCount = decomposer.contours[c];
            const auto& floatCount = offsettedContours[c];
            doubleCount.resize( floatCount.size() );
            for ( int i = 0; i < floatCount.size(); ++i )
                doubleCount[i] = Vector2d( floatCount[i] );
        }
    }

    return std::move( decomposer.contours );
}

Mesh triangulateSymbolContours( const SymbolMeshParams& params )
{
    MR_TIMER
    return PlanarTriangulation::triangulateContours( createSymbolContours( params ) );
}

void addBaseToPlanarMesh( Mesh & mesh, float zOffset )
{
    MR_TIMER

    if ( zOffset <= 0.0f )
    {
        spdlog::warn( "addBaseToPlanarMesh zOffset should be > 0, and it is {}", zOffset );
        zOffset = -zOffset;
    }

    mesh.pack(); // for some hard fonts with duplicated points (if triangulated contours have same points, duplicates are not used)
    // it's important to have all vertices valid:
    // first half is upper points of text and second half is lower points of text

    Mesh mesh2 = mesh;
    for ( auto& p : mesh2.points )
        p.z -= zOffset;

    mesh2.topology.flipOrientation();

    mesh.addPart( mesh2 );

    auto edges = mesh.topology.findHoleRepresentiveEdges();
    for ( int bi = 0; bi < edges.size() / 2; ++bi )
    {
        StitchHolesParams stitchParams;
        stitchParams.metric = getVerticalStitchMetric( mesh, Vector3f::plusZ() );
        buildCylinderBetweenTwoHoles( mesh, edges[bi], edges[edges.size() / 2 + bi], stitchParams );
    }
}

Mesh createSymbolsMesh( const SymbolMeshParams& params )
{
    MR_TIMER
    Mesh mesh = triangulateSymbolContours( params );
    addBaseToPlanarMesh( mesh );
    return mesh;
}

}
#endif
