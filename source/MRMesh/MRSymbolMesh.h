#pragma once

#ifndef MRMESH_NO_LABEL
#include "MRSystem.h"
#include <string>
#include <filesystem>

namespace MR
{

enum AlignType {
    Left,
    Center,
    Right,
};

struct SymbolMeshParams
{
    // Text that will be made mesh
    std::string text;
    // Detailization of Bezier curves on font glyphs
    int fontDetalization{5};
    // Additional offset between symbols (in symbol size: 1.0f adds one "space", 0.5 adds half "space")
    // should be >= 0.0f
    float symbolsDistanceAdditionalOffset{ 0.0f };
    // Symbols thickness will be modified by this value (newThickness = modifier*baseSymbolHeight + defaultThickness)
    // note: changing this to non-zero values cause costly calculations
    float symbolsThicknessOffsetModifier{ 0.0f };
    // alignment of the text inside bbox
    AlignType align{AlignType::Left};
#ifdef _WIN32
    // Path to font file
    std::filesystem::path pathToFontFile = GetWindowsInstallDirectory() / "Fonts" / "Consola.ttf";
#else
    // Path to font file
    std::filesystem::path pathToFontFile;
#endif // _WIN32

    // max font size with 128 << 6 FT_F26Dot6 font size
    static constexpr float MaxGeneratedFontHeight = 5826.0f * 1e-3f;
};

// converts text string into set of contours
MRMESH_API Contours2d createSymbolContours( const SymbolMeshParams& params );

// given a planar mesh with boundary on input located in plane XY, packs and extends it along Z on zOffset to make a volumetric closed mesh
// zOffset should be > 0
MRMESH_API void addBaseToPlanarMesh( Mesh& mesh, float zOffset = 1.0f );

// converts text string into Z-facing symbol mesh
MRMESH_API Mesh createSymbolsMesh( const SymbolMeshParams& params );

}
#endif
