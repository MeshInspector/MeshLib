#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRBuffer.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

namespace MR
{
class RenderMeshObject : public IRenderObject
{
public:
    RenderMeshObject( const VisualObject& visObj );
    ~RenderMeshObject();

    virtual void render( const RenderParams& params ) override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) override;
    virtual size_t heapBytes() const override;

private:
    const ObjectMeshHolder* objMesh_;

    // memory buffer for objects that about to be loaded to GPU, shared among different data types
    RenderObjectBuffer bufferObj_;
    int vertPosSize_{ 0 };
    int vertNormalsSize_{ 0 };
    int vertColorsSize_{ 0 };
    int vertUVSize_{ 0 };
    int faceIndicesSize_{ 0 };
    int edgeIndicesSize_{ 0 };
    Vector2i faceSelectionTextureSize_;
    Vector2i faceNormalsTextureSize_;
    int borderHighlightPointsSize_{ 0 };
    int selectedEdgePointsSize_{ 0 };

    RenderBufferRef<Vector3f> loadVertPosBuffer_();
    RenderBufferRef<Vector3f> loadVertNormalsBuffer_();
    RenderBufferRef<Color> loadVertColorsBuffer_();
    RenderBufferRef<UVCoord> loadVertUVBuffer_();
    RenderBufferRef<Vector3i> loadFaceIndicesBuffer_();
    RenderBufferRef<Vector2i> loadEdgeIndicesBuffer_();
    RenderBufferRef<unsigned> loadFaceSelectionTextureBuffer_();
    RenderBufferRef<Vector4f> loadFaceNormalsTextureBuffer_();
    RenderBufferRef<Vector3f> loadBorderHighlightPointsBuffer_();
    RenderBufferRef<Vector3f> loadSelectedEdgePointsBuffer_();

    typedef unsigned int GLuint;

    GLuint borderArrayObjId_{ 0 };
    GlBuffer borderBuffer_;

    GLuint selectedEdgesArrayObjId_{ 0 };
    GlBuffer selectedEdgesBuffer_;

    GLuint meshArrayObjId_{ 0 };
    GLuint meshPickerArrayObjId_{ 0 };

    GlBuffer vertPosBuffer_;
    GlBuffer vertUVBuffer_;
    GlBuffer vertNormalsBuffer_;
    GlBuffer vertColorsBuffer_;

    GlBuffer facesIndicesBuffer_;
    GlBuffer edgesIndicesBuffer_;
    GLuint texture_{ 0 };

    GLuint faceSelectionTex_{ 0 };

    GLuint faceColorsTex_{ 0 };

    GLuint facesNormalsTex_{ 0 };

    int maxTexSize_{ 0 };

    void renderEdges_( const RenderParams& parameters, GLuint vao, GlBuffer & vbo, const Color& color, uint32_t dirtyFlag );

    void renderMeshEdges_( const RenderParams& parameters );

    void bindMesh_( bool alphaSort );
    void bindMeshPicker_();

    void drawMesh_( bool solid, ViewportId viewportId, bool picker = false ) const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_( ViewportId id );

    // Marks dirty buffers that need to be uploaded to OpenGL
    uint32_t dirty_{ 0 };
    // this is needed to fix case of missing normals bind (can happen if `renderPicker` before first `render` with flat shading)
    bool normalsBound_{ false };
    // ...
    bool dirtyEdges_{ false };
};

}
