#include "MRSymbolMesh.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MR2DContoursTriangulation.h"
#include "MRMesh/MRGTest.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MROffsetContours.h"
#include "MRMesh/MRAlignContoursToMesh.h"
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

Expected<Contours2f> createSymbolContours( const SymbolMeshParams& params )
{
    MR_TIMER;

    std::error_code ec;
    if ( !std::filesystem::is_regular_file( params.pathToFontFile, ec ) )
        return unexpected( "Cannot find file with font" );

    // Begin
    FT_Library library;
    FT_Face face;
    FT_Init_FreeType( &library );
#ifdef _WIN32
    // on Windows, FT_New_Face cannot open files with Unicode names
    const auto fileSize = std::filesystem::file_size( params.pathToFontFile, ec );
    Buffer<char> buffer( fileSize );
    std::ifstream in( params.pathToFontFile, std::ifstream::binary );
    in.read( buffer.data(), buffer.size() );
    assert( in );
    auto error = FT_New_Memory_Face( library, (const FT_Byte *)buffer.data(), (FT_Long)buffer.size(), 0, &face );
    if ( error != 0 )
        return unexpected( "Font file is not valid" );
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
        else if(index == 0)
        {
            return unexpected( "Font does not contain symbol at position " + std::to_string( i ) );
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

    if ( decomposer.contours.empty() )
        return {};

    auto res = PlanarTriangulation::getOutline( std::move( decomposer.contours ), { .baseParams = {.allowMerge = true,.innerType = PlanarTriangulation::WindingMode::NonZero} } );

    if ( params.symbolsThicknessOffsetModifier != 0.0f )
        res = offsetContours( res, float( box.size().y / double( numSymbols.y ) * params.symbolsThicknessOffsetModifier ) );

    return res;
}

Expected<Mesh> triangulateSymbolContours( const SymbolMeshParams& params )
{
    MR_TIMER;
    auto contours = createSymbolContours( params );
    if ( !contours.has_value() )
    {
        return unexpected( std::move( contours.error() ) );
    }

    return PlanarTriangulation::triangulateContours( contours.value() );
}

Expected<Mesh> createSymbolsMesh( const SymbolMeshParams& params )
{
    MR_TIMER;
    auto mesh = triangulateSymbolContours( params );
    if( !mesh.has_value() )
    {
        return unexpected( std::move( mesh.error() ) );
    }
    addBaseToPlanarMesh( mesh.value(), -1.0f );
    return mesh.value();
}

}
