#pragma once

#include <MRMesh/MRIRenderObject.h>
#include <MRMesh/MRMeshTexture.h>
#include <MRMesh/MRId.h>

namespace MR
{
class RenderLinesObject : public IRenderObject
{
public:
    RenderLinesObject( const VisualObject& visObj );
    ~RenderLinesObject();

    virtual void render( const RenderParams& params ) const override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const override;

    // requested line width clamped to the range of hardware supported values
    float actualLineWidth() const;

private:
    const ObjectLinesHolder* objLines_ = nullptr;

    // need this to use per corner rendering (this is not simple copy of mesh vertices etc.)
    mutable std::vector<Vector3f> vertPosBufferObj_;
    mutable std::vector<Vector3f> vertNormalsBufferObj_;
    mutable std::vector<Color> vertColorsBufferObj_;
    mutable std::vector<UVCoord> vertUVBufferObj_;
    mutable std::vector<Vector2i> linesIndicesBufferObj_;

    typedef unsigned int GLuint;

    GLuint linesArrayObjId_{ 0 };
    GLuint linesPickerArrayObjId_{ 0 };

    GLuint vertPosBufferObjId_{ 0 };
    GLuint vertUVBufferObjId_{ 0 };
    GLuint vertNormalsBufferObjId_{ 0 };
    GLuint vertColorsBufferObjId_{ 0 };

    GLuint lineIndicesBufferObjId_{ 0 };
    GLuint texture_{ 0 };

    GLuint pointsSelectionTex_{ 0 };
    GLuint lineColorsTex_{ 0 };

    void bindLines_() const;
    void bindLinesPicker_() const;

    void drawPoints_( const RenderParams& params ) const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_() const;

    // Marks dirty buffers that need to be uploaded to OpenGL
    mutable uint32_t dirty_;
};

}