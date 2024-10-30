#pragma once
#include "MRMeshFwd.h"
#include "MRConstants.h"
#include "MRExpected.h"
#include <functional>
#include <string>

namespace MR
{

struct OffsetContourIndex
{
    // -1 means unknown index
    int contourId{ -1 };
    // -1 means unknown index
    int vertId{ -1 };
    bool valid() const { return contourId >= 0 && vertId >= 0; }
};

struct OffsetContoursOrigins
{
    // Should be always valid
    // index of lower corresponding origin point on input contour
    OffsetContourIndex lOrg;
    // index of lower corresponding destination point on input contour
    OffsetContourIndex lDest;
    // index of upper corresponding origin point on input contour
    OffsetContourIndex uOrg;
    // index of upper corresponding destination point on input contour
    OffsetContourIndex uDest;

    // ratio of intersection point on lOrg->lDest segment
    // 0.0 -> lOrg
    // 1.0 -> lDest
    float lRatio{ 0.0f };
    // ratio of intersection point on uOrg->uDest segment
    // 0.0 -> uOrg
    // 1.0 -> uDest
    float uRatio{ 0.0f };

    bool valid() const { return lOrg.valid(); }
    bool isIntersection() const { return lDest.valid(); }
};
using OffsetContoursVertMap = std::vector<OffsetContoursOrigins>;
using OffsetContoursVertMaps = std::vector<OffsetContoursVertMap>;

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
    OffsetContoursVertMaps* indicesMap = nullptr;
};

/// offsets 2d contours in plane
[[nodiscard]] MRMESH_API Contours2f offsetContours( const Contours2f& contours, float offset,
    const OffsetContoursParams& params = {} );


using ContoursVariableOffset = std::function<float( int, int )>;
/// offsets 2d contours in plane
[[nodiscard]] MRMESH_API Contours2f offsetContours( const Contours2f& contours, 
    ContoursVariableOffset offset, const OffsetContoursParams& params = {} );

/// Parameters of restoring Z coordinate of XY offset 3d contours
struct OffsetContoursRestoreZParams
{
    /// if callback is set it is called to restore Z value
    /// please note that this callback may be called in parallel
    using OriginZCallback = std::function<float( const Contours2f& offsetCont, const OffsetContourIndex& offsetIndex, const OffsetContoursOrigins& origingContourMapoing)>;
    OriginZCallback zCallback;
    /// if > 0 z coordinate will be relaxed this many iterations
    int relaxIterations = 1;
};

/// offsets 3d contours in XY plane
[[nodiscard]] MRMESH_API Contours3f offsetContours( const Contours3f& contours, float offset,
    const OffsetContoursParams& params = {}, const OffsetContoursRestoreZParams& zParmas = {} );

/// offsets 3d contours in XY plane
[[nodiscard]] MRMESH_API Contours3f offsetContours( const Contours3f& contours,
    ContoursVariableOffset offset, const OffsetContoursParams& params = {}, const OffsetContoursRestoreZParams& zParmas = {} );

}