#pragma once
#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRMeshTexture.h>
#include "MRRenderGLHelpers.h"

namespace MR
{

// Simple ImGui Image
// create GL texture in constructor, free it in destructor
// cant be moved(for now) or copied(forever)
class MRVIEWER_CLASS ImGuiImage
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

} //namespace MR
