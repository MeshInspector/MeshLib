#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRViewportId.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRSignal.h"

namespace MR
{

enum class SideRegions
{
    CCWArrow = 26,
    CWArrow = 27
};

/// class that holds and manages corner controller object
class MRVIEWER_CLASS CornerControllerObject
{
public:
    struct PickedIds
    {
        ViewportId vId;
        RegionId rId;
        bool operator==( const PickedIds& other ) const = default;
    };

    CornerControllerObject() = default;

    /// initialize this object with default settings
    MRVIEWER_API void initDefault();

    /// enables or disables this object by provided mask
    MRVIEWER_API void enable( ViewportMask mask );

    /// draw this object in given viewport in position of basis axes
    MRVIEWER_API void draw( const Viewport& vp, const AffineXf3f& rotXf, const AffineXf3f& vpInvXf );

    /// returns true if objects requires redraw
    MRVIEWER_API bool getRedrawFlag( ViewportMask mask ) const;

    /// marks this object as drawn
    MRVIEWER_API void resetRedrawFlag();
private:
    std::shared_ptr<Object> rootObj_;
    std::vector<boost::signals2::scoped_connection> connections_;

    PickedIds pick_( const Vector2f& mousePos ) const;
    void hover_( const Vector2f& mousePos );
    bool press_( const Vector2f& mousePos );
    PickedIds pickedId_;
};


/// <summary>
/// Makes cube mesh with specified face structure for each 3-rank corner, each 2-rank corner and each side:\n
/// </summary>
/// <param name="size">full side length of the cume</param>
/// <param name="cornerRatio">ratio of side length that is used for corners</param>
/// <returns>Cube mesh with specified face structure</returns>
MRVIEWER_API Mesh makeCornerControllerMesh( float size, float cornerRatio = 0.2f );

/// <summary>
/// Makes planar arrow mesh that will be used for controlling in plane rotation in corner near cube controller
/// </summary>
/// <param name="size">vertical length projection of Arrow</param>
/// <param name="shift">shift in XY plane in world units (unis same as size)</param>
/// <param name="ccw">direction of arrow</param>
/// <returns><Planar arrow mesh/returns>
MRVIEWER_API Mesh makeCornerControllerRotationArrowMesh( float size, const Vector2f& shift, bool ccw );

/// <summary>
/// Creates UV coordinates for `makeCornerControllerMesh` output mesh for texture like:\n
/// "Right"" Left "\n
/// " Top ""Bottom"\n
/// "Front"" Back "
/// </summary>
/// <param name="cornerRatio">ratio of side length that is used for corners, should be the same that was used in `makeCornerControllerMesh`</param>
/// <returns>UV coordinates for each vertex of corner controller mesh</returns>
MRVIEWER_API VertUVCoords makeCornerControllerUVCoords( float cornerRatio = 0.2f );

/// <summary>
/// Loads 3 textures for corner controller: default, side hovered, corner hovered
/// </summary>
/// <returns>3 textures, or empty vector on error</returns>
MRVIEWER_API Vector<MeshTexture, TextureId> loadCornerControllerTextures();

/// returns textures map for each part\n
/// actually all zeros
MRVIEWER_API const TexturePerFace& getCornerControllerTexureMap();

/// returns region id of corner controller by its face
MRVIEWER_API RegionId getCornerControllerRegionByFace( FaceId face );

/// returns textures map with region faces hovered
MRVIEWER_API TexturePerFace getCornerControllerHoveredTextureMap( RegionId rId );

/// setup camera for selected viewport by corner controller region
MRVIEWER_API void updateCurrentViewByControllerRegion( CornerControllerObject::PickedIds pickedId );

}