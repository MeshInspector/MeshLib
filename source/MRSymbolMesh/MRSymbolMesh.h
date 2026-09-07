#pragma once

#include "MRSymbolMeshFwd.h"

#include "MRMesh/MRVector2.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRExpected.h"

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
    // Additional offset between symbols
    // X: In symbol size: 1.0f adds one "space", 0.5 adds half "space".
    // Y: In symbol size: 1.0f adds one base height, 0.5 adds half base height
    Vector2f symbolsDistanceAdditionalOffset{ 0.0f, 0.0f };
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
MRSYMBOLMESH_API Expected<Contours2f> createSymbolContours( const SymbolMeshParams& params );

// converts text string into Z-facing symbol mesh
MRSYMBOLMESH_API Expected<Mesh> createSymbolsMesh( const SymbolMeshParams& params );

}
