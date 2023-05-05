#pragma once
#include "exports.h"
#include <MRMesh/MRColor.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRVoxelPath.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRRenderGLHelpers.h"

namespace MR
{

// Simple ImGui Image
// create GL texture in constructor, free it in destructor
// cant be moved(for now) or copied(forever)
class ImGuiImage
{
public:
    MRVIEWER_API ImGuiImage();
    MRVIEWER_API virtual ~ImGuiImage();

    // Sets image to texture
    MRVIEWER_API void update( const MeshTexture& texture );

    // Returns void* for ImGui::Image( getImTextureId(), ... )
    void* getImTextureId() const { return (void*) (intptr_t) glTex_.getId(); }

    // Returns gl texture id
    unsigned getId() const { return glTex_.getId(); }

    // Returns current MeshTexture
    const MeshTexture& getMeshTexture() const { return texture_; }

    int getImageWidth() const { return texture_.resolution.x; }
    int getImageHeight() const { return texture_.resolution.y; }

private:
    GlTexture2 glTex_;
    MeshTexture texture_;
         
    void bind_();
};

#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
// Class for easy controlling slice with seeds marks
class MarkedVoxelSlice : public ImGuiImage
{
public:
    struct Mark
    {
        Color color;
        VoxelBitSet mask;
    };

    MRVIEWER_API MarkedVoxelSlice( const ObjectVoxels& voxels );
    
    enum MaskType { Inside, Outside, Segment, Count };

    // Needed to avoid VoxelBitSet copy
    // ensure using forceUpdate() after chaging this reference
    VoxelBitSet& getMask( MaskType type ) { return params_.marks[type].mask; }
    // Returns mask(VoxelBitSet of whole voxel object) of given type
    const VoxelBitSet& getMask( MaskType type ) const { return params_.marks[type].mask; }
    // Sets mask(VoxelBitSet of whole voxel object) of given type, updates texture
    void setMask( const VoxelBitSet& mask, MaskType type ) { params_.marks[type].mask = mask; forceUpdate(); }
    
    // Colors of slice marks controls, setters update texture
    const Color& getColor( MaskType type ) const { return params_.marks[type].color; }
    void setColor( const Color& color, MaskType type ) { params_.marks[type].color = color; forceUpdate(); }


    // Needed to avoid VoxelBitSet copy
    // ensure using forceUpdate() after chaging this reference
    Mark& getMark( MaskType type ) { return params_.marks[type]; }
    // Returns color and mask(VoxelBitSet of whole voxel object) of given type
    const Mark& getMark( MaskType type ) const { return params_.marks[type]; }
    // Sets color and mask(VoxelBitSet of whole voxel object) of given type, updates texture
    void setMark( const Mark& mark, MaskType type ) { params_.marks[type] = mark; forceUpdate(); }

    // Needed to avoid VoxelBitSet copy
    // ensure using forceUpdate() after chaging this reference
    // returns background colors and masks(VoxelBitSet of whole voxel object)
    std::vector<Mark>& getCustomBackgroundMarks() { return params_.customBackgroundMarks; }
    const std::vector<Mark>& getCustomBackgroundMarks() const { return params_.customBackgroundMarks; }
    // Sets background colors and masks(VoxelBitSet of whole voxel object) of given type, updates texture
    void setCustomBackgroundMarks( const std::vector<Mark>& backgroundMarks ) { params_.customBackgroundMarks = backgroundMarks; forceUpdate(); }

    // Needed to avoid VoxelBitSet copy
    // ensure using forceUpdate() after chaging this reference
    // returns foreground colors and masks(VoxelBitSet of whole voxel object)
    std::vector<Mark>& getCustomForegroundMarks() { return params_.customForegroundMarks; }    
    const std::vector<Mark>& getCustomForegroundMarks() const { return params_.customForegroundMarks; }
    // Sets foreground colors and masks(VoxelBitSet of whole voxel object) of given type, updates texture
    void setCustomForegroundMarks( const std::vector<Mark>& foregroundMarks ) { params_.customForegroundMarks = foregroundMarks; forceUpdate(); }


    // Active plane (YZ, ZX or XY) controls, setters update texture
    SlicePlane getActivePlane() const { return params_.activePlane; }
    void setActivePlane( SlicePlane plane ) { params_.activePlane = plane; forceUpdate(); }

    const Vector3i& getActiveVoxel() const { return params_.activeVoxel; }
    void setActiveVoxel( const Vector3i& voxel ) { params_.activeVoxel = voxel; forceUpdate(); }

    // Slice normalization parameters, setters update texture
    float getMin() const { return params_.min; }
    void setMin( float min ) { params_.min = min; forceUpdate(); }
    float getMax() const { return params_.max; }
    void setMax( float max ) { params_.max = max; forceUpdate(); }

    // Returns current active box of slice
    const Box3i& getActiveBox() const { return params_.activeBox; }
    // Updates active box of slice, do not affect ObjectVoxels, updates texture
    void setActiveBox( const Box3i& box ) { params_.activeBox = box; forceUpdate(); }

    // Parameters of slice
    struct Parameters
    {
        // Base marks
        std::array<Mark, size_t( MaskType::Count )> marks = {Mark{Color::red()},Mark{Color::blue()},Mark{Color::yellow()}};
        std::vector<Mark> customBackgroundMarks;
        std::vector<Mark> customForegroundMarks;
        // Current voxel 
        Vector3i activeVoxel;
        // Active box, set as ObjectVoxels active box in constructor
        Box3i activeBox;
        // Minimum dense to show black
        float min{0.0f};
        // Maximum dense to show white
        float max{0.0f};
        // Slice plane
        SlicePlane activePlane{XY};
    };

    // Get all parameters as one structure
    const Parameters& getParameters() const { return params_; }
    // Set all parameters as one structure, updates texture
    void setParameters( const Parameters& params ) { params_ = params; forceUpdate(); }

    // Set current slice with marks to texture, do not abuse this
    MRVIEWER_API void forceUpdate();

private:
    FloatGrid grid_;
    Vector3i dims_;

    Parameters params_;

};
#endif

}
