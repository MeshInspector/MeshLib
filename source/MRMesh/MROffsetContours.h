#pragma once
#include "MRMeshFwd.h"
#include "MRConstants.h"
#include "MRExpected.h"
#include <functional>
#include <string>

namespace MR
{

struct ContoursVertId
{
    // -1 means unknown index
    int contourId{ -1 };
    // -1 means unknown index
    int vertId{ -1 };
    bool operator==( const ContoursVertId& other ) const = default;
};
using ContoursVertMap = std::vector<ContoursVertId>;
using ContoursVertMaps = std::vector<ContoursVertMap>;

struct OffsetContoursParams
{
    /// type of offset
    enum class Type
    {
        Offset, ///< One-side signed offset, requires closed contours
        Shell ///< Two-side offset
    } type{ Type::Offset };

    /// type of offsetting on ends of non-closed contours
    enum class EndType
    {
        Round, ///< creates round ends (use `minAnglePrecision`)
        Cut ///< creates sharp end (same as Round with `minAnglePrecision` < 180 deg)
    } endType{ EndType::Round };

    /// type of positive offset curve in corners
    enum class CornerType
    {
        Round, ///< creates round corners (use `minAnglePrecision`)
        Sharp ///< creates sharp connected corner (use `maxSharpAngle` as limit)
    } cornerType{ CornerType::Round };

    /// precision of round corners and ends
    float minAnglePrecision = PI_F / 9.0f; // 20 deg
    /// limit for sharp corners connection
    float maxSharpAngle = PI_F * 2.0f / 3.0f; // 120 deg

    /// optional output that maps result contour ids to input contour ids
    ContoursVertMaps* indicesMap = nullptr;
};

/// offsets 2d contours in plane
[[nodiscard]] MRMESH_API Expected<Contours2f> offsetContours( const Contours2f& contours, float offset,
    const OffsetContoursParams& params = {} );


using ContoursVariableOffset = std::function<float( int, int )>;
/// offsets 2d contours in plane
[[nodiscard]] MRMESH_API Expected<Contours2f> offsetContours( const Contours2f& contours, 
    ContoursVariableOffset offset, const OffsetContoursParams& params = {} );

}