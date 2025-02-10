#pragma once
#include "MRViewerFwd.h"

namespace MR
{

/// <summary>
/// Makes cube mesh with specified face structure for each 3-rank corner, each 2-rank corner and each side:\n
/// </summary>
/// <param name="size">full side length of the cume</param>
/// <param name="cornerRatio">ratio of side length that is used for corners</param>
/// <returns>Cube mesh with specified face structure</returns>
MRVIEWER_API Mesh makeCornerControllerMesh( float size, float cornerRatio = 0.2f );

/// returns color map for each part\n
/// x side - red\n
/// y side - green\n
/// z side - blue\n
/// xy - mixed red + green\n
/// etc.
MRVIEWER_API const FaceColors& getCornerControllerColorMap();

/// returns region id of corner controller by its face
MRVIEWER_API RegionId getCornerControllerRegionByFace( FaceId face );

/// returns color map with region faces hovered
MRVIEWER_API FaceColors getCornerControllerHoveredColorMap( RegionId rId );

/// setup camera for selected viewport by corner controller region
MRVIEWER_API void updateCurrentViewByControllerRegion( RegionId rId );

}